import sys
sys.path.append("")
import os
import warnings
import json
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image, ImageEnhance
import importlib

import torch
import torch.nn.functional as F
from torchvision import transforms

import core.util as Util
from core.praser import parse_engine
from models.model import MaLiang
from utils.img_proc.image_process import shape_to_mask, crop_image, points2mask
import os
import pdb

class MEngine():
    def __init__(self, opt, img_sz=256):
        if not isinstance(opt, dict) and os.path.exists(opt):
            opt = parse_engine(opt)      
        self.opt = opt
        # gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        self.device = 'cpu'
        assert len(self.opt['gpu_ids']) <= 1, 'in inference, we only need one gpu'
        if len(self.opt['gpu_ids']) >= 1:
            gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            self.device = 'cuda:' + str(self.opt['gpu_ids'][0])
        if 'local_rank' not in opt:
            self.opt['local_rank'] = self.opt['global_rank'] = self.opt['gpu_ids'][0]
            self.opt['device'] = self.device

        '''set seed and and cuDNN environment '''
        torch.backends.cudnn.enabled = False
        if torch.backends.cudnn.enabled:
            warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
        Util.set_seed(self.opt['seed'])

        '''network'''
        net_opt = self.opt['model']['which_networks'][0].get('args', {})
        name = self.opt['model']['which_networks'][0].get('name', {})[1]
        file_name = self.opt['model']['which_networks'][0].get('name', {})[0]

        # assert name == 'MLDM', "Name must be MLDM!"

        module = importlib.import_module(file_name)

        net = getattr(module, name)(**net_opt)
        net.__name__ = net.__class__.__name__

        self.model = MaLiang(networks=[net],
                             losses=["mse_loss"],
                             sample_num=8,
                             task='inpainting',
                             optimizers=[{"lr": 5e-5, "weight_decay": 0}],
                             ema_scheduler=None,
                             opt=self.opt,
                             phase_loader=None,
                             val_loader=None,
                             metrics=None,
                             logger=None,
                             writer=None
                             )
        self.__name__ = self.__class__.__name__

        self.im_crop_sz = img_sz
        ## defect name and its index, 0 is left to UNK label
        self.defect2eng = {'noneless':'noneless', 'lp':1, 'ng':2, 
                            'huashang':'scratch', 'pengshang':'bruise', 'yise':'color variation',
                            'aokeng':'dent', 'heidian':'black spot', 'shahenyin':'sanding marks', 
                            'bianxing':'deformation', 'tabian':'collapse edge', 'molie':'film cracking', 
                            'gubao':'bulge', 'yiwu':'foreign object', 'guashang':'scrape', 
                            'caizhixian':'material line', 'liewen':'crack', 'daowen':'Knife mark', 
                            'zhanya':'adsorption crushing', 'aotuhen':'recessed and raised Marks', 'cashang':'abrasion', 
                            'yashang':'crushing', 'madian':'pitting spots', 'youmo':'ink stain', 'zangwu':'stain', 
                            'baidian':'white spot', 'maoxu':'lint', 'keli':'particles', 'quepeng':'edge bruise', 
                            'maoci':'burrs', 'queliao':'missing material', 'quepenghua':'bruise-scratch',                          
                            'wuluowen':'threadless', 'zhanliao':'residue', 'liuwen':'flow marks', 
                            'aotu':'concave-convex', 'juchi':'serrated edge', 'qipao':'air bubble', 
                            'zanghua':'blemis', 'kailie':'crack', 'xianweimao':'fiber hair', 'nzgs':'unknow', 
                            'jiaobuliang':'poor adhesion', 'aotudian':'concave and convex spot'}

        if "extra_class" in opt and opt["extra_class"] is not None:
            for name in opt["extra_class"]:
                if name in self.defect2eng:
                    continue
                self.defect2eng[name] = opt["extra_class"][name]

    def get_prompt(self, img, mask, label, task='generation'): 
        try:     
            pos = np.argwhere(mask > 0)
            (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
        except:
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
        area_prompt = '<{:d}><{:d}><{:d}><{:d}>'.format(x1, y1, x2, y2)

        if task == 'inpaint':
            prompt = '[inpaint] please inpaint the area: ' + area_prompt     
        elif task == 'generation':
            if label not in self.defect2eng:
                prompt = '[generation] there is a defect in the area: ' + area_prompt
            else:
                prompt = '[generation] there is a {:s} defect in the area: '.format(self.defect2eng[label]) + area_prompt
        elif task =='synthesis':
            if label not in self.defect2eng:
                prompt = '[sythesis] please synthesize a defect in the area: ' + area_prompt
            else:
                prompt = '[sythesis] please synthesize a {:s} defect in the area: '.format(self.defect2eng[label]) + area_prompt
        else:
            print("Oops, we just support generation, inpaint and synthsis tasks.")

        mask = np.expand_dims(mask, -1)
        if task == 'synthesis':
            im_cond = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        else:
            im_cond = img * (1 - mask)
            im_cond = torch.from_numpy(cv2.cvtColor(im_cond, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        im_cond = (im_cond / 127.5) - 1 

        return im_cond, prompt
        
    def process_(self, img, points, shape_type, label, mask_type='rect', task='generation'):
        '''
        task: generation | inpaint | synthesis
              generate means ng/lp --> ng
              inpaint means ng/lp --> lp
              synthesis means ng --> ng
        '''
        ret_img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        ret_img = (ret_img / 127.5) - 1 

        ret_mask = points2mask(points, shape_type, label=label, mask_type=mask_type, MIN_PIX=32, im_crop_sz=self.im_crop_sz)  # rect | poly
        ret_mask = torch.from_numpy(ret_mask).float()
        
        # image_shape = [self.im_crop_sz, self.im_crop_sz, 3]
        # label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
        # label_inst = np.where(label_inst == True, 1.0, 0).astype('uint8')
        # label_inst = np.expand_dims(label_inst, 0)
        # ret_inst = torch.from_numpy(label_inst).float()

        ret_im_cond, ret_prompt = self.get_prompt(img=img, 
                                                mask=ret_mask.squeeze(0).numpy(),
                                                label=label,
                                                task=task)
        
        return ret_img, ret_mask, ret_im_cond, ret_prompt
    
    @torch.no_grad()
    def edit(self, 
                  imp, 
                  jsp, 
                  defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                  gid_need_gen=[1000, 2000],
                  ratio=1.0,
                  gd_w=0.0, 
                  prob_syn=1.0,
                  defect_aug_ratio=None,
                  SHOW=False,  
                  return_visuals=False,
                  mask_type='rect',
                  use_rectangle_labelme=False,
                  task='generation'):
        
        img = cv2.imread(imp)
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)

        IMG_CHANGED, new_img, new_info = self.edit_(img, 
                                                     info, 
                                                     defect_need_gen=defect_need_gen,
                                                     gid_need_gen=gid_need_gen,
                                                     ratio=ratio,
                                                     gd_w=gd_w, 
                                                     prob_syn=prob_syn,
                                                     defect_aug_ratio=defect_aug_ratio,
                                                     SHOW=SHOW,  
                                                     return_visuals=return_visuals,
                                                     mask_type=mask_type,
                                                     use_rectangle_labelme=use_rectangle_labelme,
                                                     task=task)
        return IMG_CHANGED, new_img, new_info
    
    @torch.no_grad()
    def edit_(self, 
                   img, 
                   info, 
                   defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                   gid_need_gen=[1000, 2000],
                   ratio=1.0,
                   gd_w=0.0, 
                   prob_syn=1.0,
                   defect_aug_ratio=None,
                   SHOW=False,  
                   return_visuals=False,
                   mask_type='rect',
                   use_rectangle_labelme=False,
                   task=None):
        
        IMG_CHANGED = False
        new_info = deepcopy(info)
        new_info["shapes"] = list()

        for _shape in info['shapes']:
            label = _shape.get("label")

            ## check: whether does the defect need to edit 
            if label not in defect_need_gen:
                new_info["shapes"].append(_shape)
                continue 

            ##  
            if defect_aug_ratio is not None:
                if label in defect_aug_ratio:
                    ratio = defect_aug_ratio[label]                  

            group_id = _shape.get("group_id")
            if len(gid_need_gen) > 0 and group_id not in gid_need_gen:
                new_info["shapes"].append(_shape)
                continue

            shape_type = _shape.get("shape_type")
            points = _shape.get("points")
            ret, crop_img, new_points, real_bbox = crop_image(img, points, self.im_crop_sz)
            if ret == 0:
                if len(gid_need_gen) <= 0:
                    new_info["shapes"].append(_shape)
                continue

            if np.random.rand() > prob_syn:
                new_info["shapes"].append(_shape)
                continue
      
            if SHOW or return_visuals:
                img_show = crop_img.copy()

            ret_img, ret_mask, ret_im_cond, ret_prompt = self.process_(img=crop_img, 
                                                                        points=new_points, 
                                                                        shape_type=shape_type,
                                                                        mask_type=mask_type,
                                                                        label=label,
                                                                        task=task)
            print(ret_prompt)
            
            # if the area of mask is too large or NULL, we will skip 
            if ret_mask.sum() <= 0 or ret_mask.sum() / (self.im_crop_sz * self.im_crop_sz) > 0.5:
                new_info["shapes"].append(_shape)
                continue

            ## set to device
            gt_image = self.model.set_device(ret_img.unsqueeze(0))
            cond_image = self.model.set_device(ret_im_cond.unsqueeze(0))
            mask = self.model.set_device(ret_mask.unsqueeze(0))
            text = [ret_prompt]
            
            if return_visuals:
                output, visuals = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=ratio, gd_w=gd_w, return_visuals=True)
            else:
                output = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=ratio, gd_w=gd_w)
            output = Util.tensor2img(output.cpu())
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
            if use_rectangle_labelme:
                _shape["shape_type"] = 'rectangle'            
                label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
            else:
                # _shape["shape_type"] = 'polygon'
                # label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                # pos = np.argwhere(label_mask > 0)
                # (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                # _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1],
                #                 [float(x2) + boxx_x1, float(y1) + boxx_y1],
                #                 [float(x2) + boxx_x1, float(y2) + boxx_y1], 
                #                 [float(x1) + boxx_x1, float(y2) + boxx_y1]]
                pass

            if task is not None:
                _shape["group_id"] = task

            new_info["shapes"].append(_shape) 
         
            img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = output[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()
            IMG_CHANGED = True

            if SHOW:
                img_cond = Util.tensor2img(cond_image.cpu())
                img_cond = cv2.cvtColor(img_cond, cv2.COLOR_RGB2BGR)
                img_res = output.copy()
                img_diff = np.abs(img_res - img_show)
                label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                img_res = np.concatenate([img_show, img_res, img_diff, img_cond], 1)
                
                cv2.imshow("res", img_res)
                cv2.waitKey(0)
            if return_visuals:
                img_mask = Util.tensor2img(2 * mask.cpu() - 1)
                img_r = Util.tensor2img(visuals[0, :, :, :].cpu())
                img_m = Util.tensor2img(visuals[7, :, :, :].cpu())
                # pdb.set_trace()
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
                img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
                img_m = cv2.cvtColor(img_m, cv2.COLOR_RGB2BGR)
                img_res = output.copy()

                img_res = np.concatenate([img_show, img_mask, img_r, img_m, img_res], 1)
                index = 0
                save_name = './test/exp_paper/exp_' + str(index) + '.jpg'
                while os.path.exists(save_name):
                    index += 1
                    save_name = './test/exp_paper/exp_' + str(index) + '.jpg'
                cv2.imwrite(save_name, img_res)
                
                cv2.imshow("res", img_res)
                cv2.waitKey(0)


        return IMG_CHANGED, img, new_info  
    
    def defect_transfer(self, imp, jsp, 
                        SHOW=False,
                        mask_type='rect'):
        
        img = cv2.imread(imp)
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)

        new_img, new_info = self.defect_selfaug_(img, info,
                                                 aug_type_prob={'re-generate':1.0,'defect2lp':0.0,'xy-shift':0.0,'flip':0.0},
                                                 ratio=1.0,
                                                 gd_w=0.0, 
                                                 SHOW=SHOW,
                                                 mask_type=mask_type,
                                                 task='synthesis',
                                                 use_rectangle_labelme=False) 
        
        return new_img, new_info
    
    def defect_selfaug(self, imp, jsp, 
                        aug_type_prob={'re-generate':1.0,'defect2lp':0.0,'xy-shift':0.0,'flip':0.0},
                        ratio=1.0,
                        gd_w=0.0, 
                        SHOW=False,
                        mask_type='rect',
                        use_rectangle_labelme=False):
        img = cv2.imread(imp)
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)

        new_img, new_info = self.defect_selfaug_(img, info,
                                                 aug_type_prob=aug_type_prob,
                                                 ratio=ratio,
                                                 gd_w=gd_w,
                                                 SHOW=SHOW,
                                                 mask_type=mask_type,
                                                 use_rectangle_labelme=use_rectangle_labelme)
        
        return new_img, new_info
    
    def defect_selfaug_(self, img, info, 
                        aug_type_prob={'re-generate':1.0,'defect2lp':0.0,'xy-shift':0.0,'flip':0.0},
                        ratio=1.0,
                        gd_w=0.0, 
                        SHOW=False,
                        mask_type='rect',
                        task='generation',
                        use_rectangle_labelme=False):
        '''
        aug_type: re-generate | defect2lp | xy-shift | flip
        '''
        assert np.array(list(aug_type_prob.values())).sum() == 1.0, "Oops, the sum of prob does not equal to 1.0."
        new_info = deepcopy(info)
        new_info["shapes"] = list()

        for _shape in info['shapes']:
            # if not _shape.get("need_aug"):
            #     new_info["shapes"].append(_shape)
            #     continue

            crop_shape = deepcopy(_shape)
            points = _shape.get("points")
            ret, crop_img, new_points, real_bbox = crop_image(img, points, self.im_crop_sz)               
            if ret == 0:
                new_info["shapes"].append(_shape)
                continue

            if SHOW:
                img_show = crop_img.copy()

            crop_shape["points"] = new_points
            op_aug = np.random.choice(list(aug_type_prob.keys()), p=list(aug_type_prob.values()))

            if op_aug == 're-generate':     
                img_regen, mask = self.regenerate_(crop_img, crop_shape, ratio=ratio, mask_type=mask_type, task=task)    
                boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
                img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = img_regen[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()

                _shape['group_id'] = 're-generate'
                if use_rectangle_labelme:
                    _shape["shape_type"] = 'rectangle'            
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
                else:
                    # _shape["shape_type"] = 'polygon'
                    # label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    # pos = np.argwhere(label_mask > 0)
                    # (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    # _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1],
                    #                 [float(x2) + boxx_x1, float(y1) + boxx_y1],
                    #                 [float(x2) + boxx_x1, float(y2) + boxx_y1], 
                    #                 [float(x1) + boxx_x1, float(y2) + boxx_y1]]
                    pass     
                new_info["shapes"].append(_shape)   

                if SHOW:
                    img_res = img_regen.copy()
                    img_diff = np.abs(img_res - img_show)
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                    cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                    cv2.putText(img_show, _shape["label"], (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                    img_res = np.concatenate([img_show, img_res, img_diff], 1)
                    
                    cv2.imshow("re-generate", img_res)
                    cv2.waitKey(0)

                continue

            ## defect2lp
            img_d2l, mask = self.defect2lp_(crop_img, crop_shape, mask_type=mask_type)

            if op_aug == 'defect2lp':
                boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
                img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = img_d2l[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()

                if SHOW:
                    img_res = img_d2l.copy()
                    img_diff = np.abs(img_res - img_show)
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                    cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                    cv2.putText(img_show, _shape["label"], (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                    img_res = np.concatenate([img_show, img_res, img_diff], 1)
                    
                    cv2.imshow("defect2lp", img_res)
                    cv2.waitKey(0)

                continue

            if op_aug == 'xy-shift':
                img_shift, mask, shape_shift = self.xyshift_(crop_img, crop_shape, img_d2l, ratio=ratio, mask_type=mask_type,)
                boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
                img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = img_shift[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()
                _shape["points"] = [[pp[0]+boxx_x1, pp[1]+boxx_y1] for pp in shape_shift["points"]]

                if SHOW:
                    img_res = img_shift.copy()
                    img_diff = np.abs(img_res - img_show)
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                    cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                    cv2.putText(img_show, _shape["label"], (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                    img_res = np.concatenate([img_show, img_res, img_diff], 1)
                    
                    cv2.imshow("xy-shift", img_res)
                    cv2.waitKey(0)

                continue

            if op_aug == 'flip':
                img_flip, mask, shape_flip = self.flip_(crop_img, crop_shape, img_d2l, ratio=ratio, mask_type=mask_type,)
                boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
                img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = img_flip[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()
                _shape["points"] = [[pp[0]+boxx_x1, pp[1]+boxx_y1] for pp in shape_flip["points"]]

                if SHOW:
                    img_res = img_flip.copy()
                    img_diff = np.abs(img_res - img_show)
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                    cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                    cv2.putText(img_show, _shape["label"], (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                    img_res = np.concatenate([img_show, img_res, img_diff], 1)
                    
                    cv2.imshow("flip", img_res)
                    cv2.waitKey(0)

                continue

        return img, new_info
    
    def flip_(self, img, shape, img_lp, ratio=0.5, flip_type=None, mask_type='rect'):
        points, shape_type, label = shape["points"], shape["shape_type"], shape["label"]
        mask_thin = points2mask(points, shape_type, label=label, mask_type='rect', MIN_PIX=16, line_width=8, point_size=8)
        mask_poly = points2mask(points, shape_type, label=label, mask_type='poly', MIN_PIX=8, line_width=8, point_size=8)

        label_mask = mask_thin[0, :, :]
        pos = np.argwhere(label_mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
        img_iou = img[y1:y2, x1:x2, :].copy()
        mask_flip = np.zeros_like(mask_poly)

        if flip_type is None:
            if x2 - x1 >= y2 - y1:
                flip_type = 'y'
            else:
                flip_type = 'x'

        if flip_type == 'x':
            img_iou = np.flip(img_iou, 1)
            shape["points"] = [[img_iou.shape[1] - (pp[0] - x1) + x1, pp[1]] for pp in points]
            mask_flip = np.flip(mask_poly, 2)
        else:
            img_iou = np.flip(img_iou, 0)
            shape["points"] = [[pp[0], img_iou.shape[0] - (pp[1] - y1) + y1] for pp in points]
            mask_flip = np.flip(mask_poly, 1)
        img_lp[y1:y2, x1:x2, :] = img_iou

        ret_img, ret_mask, ret_im_cond, ret_prompt = self.process_(img=img_lp, 
                                                                    points=shape["points"], 
                                                                    shape_type=shape["shape_type"],
                                                                    mask_type=mask_type,
                                                                    label=shape["label"],
                                                                    task='generation')
        print(ret_prompt)

        ## set to device
        gt_image = self.model.set_device(ret_img.unsqueeze(0))
        cond_image = self.model.set_device(ret_im_cond.unsqueeze(0))
        mask = self.model.set_device(ret_mask.unsqueeze(0))
        text = [ret_prompt]

        ## inpaint to lp
        output_shift = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=ratio, gd_w=0.0)
        img_flip = Util.tensor2img(output_shift.cpu())
        img_flip = cv2.cvtColor(img_flip, cv2.COLOR_RGB2BGR) 

        mask_flip = np.transpose(mask_poly, [1, 2, 0]) 
        img_flip = img_lp * (1 - mask_flip) + img_flip * mask_flip
        img_flip = img_flip.astype('uint8')

        return img_flip, mask, shape

    def xyshift_(self, img, shape, img_lp, ratio=0.5, shift_x=None, shift_y=None, mask_type='rect'):
        points, shape_type, label = shape["points"], shape["shape_type"], shape["label"]
        mask_thin = points2mask(points, shape_type, label=label, mask_type='rect', MIN_PIX=16, line_width=8, point_size=8)
        mask_poly = points2mask(points, shape_type, label=label, mask_type='poly', MIN_PIX=8, line_width=8, point_size=8)

        label_mask = mask_thin[0, :, :]
        pos = np.argwhere(label_mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1

        if shift_x is None or shift_y is None:
            shift_x = np.random.randint(-16, 17) #np.random.choice([-1, 1, 0])
            shift_y = np.random.randint(-16, 17) #np.random.choice([-1, 1, 0])
            if shift_x > 0:
                while True:
                    if x2 + shift_x <= img.shape[1]:
                        break
                    else:
                        shift_x -= 1
            if shift_x < 0:
                while True:
                    if x1 + shift_x >= 0:
                        break
                    else:
                        shift_x += 1
            if shift_y > 0:
                while True:
                    if y2 + shift_y <= img.shape[0]:
                        break
                    else:
                        shift_y -= 1
            if shift_y < 0:
                while True:
                    if y1 + shift_y >= 0:
                        break
                    else:
                        shift_y += 1
                    
        shape["points"] = [[p[0]+shift_x, p[1]+shift_y] for p in points]
        img_lp[y1+shift_y:y2+shift_y, x1+shift_x:x2+shift_x, :] = img[y1:y2, x1:x2, :].copy()
        mask_shift = np.zeros_like(mask_poly)
        mask_shift[:, y1+shift_y:y2+shift_y, x1+shift_x:x2+shift_x] = mask_poly[:, y1:y2, x1:x2]

        ret_img, ret_mask, ret_im_cond, ret_prompt = self.process_(img=img_lp, 
                                                                    points=shape["points"], 
                                                                    shape_type=shape["shape_type"],
                                                                    mask_type=mask_type,
                                                                    label=shape["label"],
                                                                    task='generation')
        print(ret_prompt)

        ## set to device
        gt_image = self.model.set_device(ret_img.unsqueeze(0))
        cond_image = self.model.set_device(ret_im_cond.unsqueeze(0))
        mask = self.model.set_device(ret_mask.unsqueeze(0))
        text = [ret_prompt]

        ## inpaint to lp
        output_shift = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=ratio, gd_w=0.0)
        img_shift = Util.tensor2img(output_shift.cpu())
        img_shift = cv2.cvtColor(img_shift, cv2.COLOR_RGB2BGR)   

        mask_shift = np.transpose(mask_poly, [1, 2, 0]) 
        img_shift = img_lp * (1 - mask_shift) + img_shift * mask_shift
        img_shift = img_shift.astype('uint8')

        return img_shift, mask, shape

    def regenerate_(self, img, shape, ratio=1.0, mask_type='rect', task='generation'):  
        '''
        NOTE: ratio is between 0 to 1
              when ratio = 1.0, this function degrads as generation
        '''
        ret_img, ret_mask, ret_im_cond, ret_prompt = self.process_(img=img, 
                                                                    points=shape["points"], 
                                                                    shape_type=shape["shape_type"],
                                                                    mask_type=mask_type,
                                                                    label=shape["label"],
                                                                    task=task)
        print(ret_prompt)

        ## set to device
        gt_image = self.model.set_device(ret_img.unsqueeze(0))
        cond_image = self.model.set_device(ret_im_cond.unsqueeze(0))
        mask = self.model.set_device(ret_mask.unsqueeze(0))
        text = [ret_prompt]

        ## inpaint to lp
        output_regen = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=ratio, gd_w=0.0)
        img_regen = Util.tensor2img(output_regen.cpu())
        img_regen = cv2.cvtColor(img_regen, cv2.COLOR_RGB2BGR)    

        return img_regen, mask

    def defect2lp_(self, img, shape, mask_type='rect'):  
        ret_img, ret_mask, ret_im_cond, ret_prompt = self.process_(img=img, 
                                                                    points=shape["points"], 
                                                                    shape_type=shape["shape_type"],
                                                                    mask_type=mask_type,
                                                                    label=shape["label"],
                                                                    task='inpaint')
        print(ret_prompt)

        ## set to device
        gt_image = self.model.set_device(ret_img.unsqueeze(0))
        cond_image = self.model.set_device(ret_im_cond.unsqueeze(0))
        mask = self.model.set_device(ret_mask.unsqueeze(0))
        text = [ret_prompt]

        ## inpaint to lp
        output_d2l = self.model.inpaint_generater(gt_image, cond_image, mask, text=text, ratio=1.0, gd_w=0.0)
        img_d2l = Util.tensor2img(output_d2l.cpu())
        img_d2l = cv2.cvtColor(img_d2l, cv2.COLOR_RGB2BGR)    
        mask_poly = points2mask(shape["points"], shape["shape_type"], label=shape["label"], mask_type='poly', MIN_PIX=8, line_width=8, point_size=8)        
        alpha = mask_poly.transpose(1, 2, 0)
        img_d2l = img * (1 - alpha) + img_d2l * alpha
        img_d2l = img_d2l.astype('uint8')

        return img_d2l, mask

    
if __name__ == "__main__":
    config = './config/infer/infer_inpainting_mldm_ch256.json'    
    # config = './config/infer/infer_inpainting_tldm_ch256.json'  
    engine = MEngine(config) 

    ## test for base
    # import glob
    # for imp in glob.glob('./test/' + '*.jpg'):
    #     jsp = imp.replace('.jpg', '.json')
    #     ret, img, new_info = engine.edit(imp, 
    #                                     jsp, 
    #                                     defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
    #                                     gid_need_gen=[],
    #                                     task='generation',
    #                                     gd_w=0.0,
    #                                     ratio=0.2,
    #                                     SHOW=True)

    ## test for self-aug
    src_dir = '/home/data0/temp/'
    src_imgs_list = list()
    for filepath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if '.json' in filename:
                continue
            imp = os.path.join(filepath, filename)
            jsp = imp.replace('.jpg', '.json')               
            if not os.path.exists(jsp):
                continue 
            src_imgs_list.append(imp)
    for imp in src_imgs_list:
        jsp = imp.replace('.jpg', '.json')
        new_img, new_info = engine.defect_selfaug(imp=imp,
                                                  jsp=jsp,
                                                  aug_type_prob={'re-generate':0.0,'defect2lp':1.0,'xy-shift':0.0,'flip':0.0},
                                                  ratio=0.2,
                                                  gd_w=0.0, 
                                                  SHOW=True,
                                                  mask_type='rect',
                                                  use_rectangle_labelme=False)
        
        # new_img, new_info = engine.defect_transfer(imp=imp,
        #                                           jsp=jsp,
        #                                           SHOW=True,
        #                                           mask_type='rect')

    




        