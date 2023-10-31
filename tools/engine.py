import sys
sys.path.append("")
import os
import warnings
import json
from collections import OrderedDict
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image, ImageDraw
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

class Engine():
    def __init__(self, opt, img_sz=256):
        if not isinstance(opt, dict) and os.path.exists(opt):
            opt = parse_engine(opt)      
        self.opt = opt
        # gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        self.device = 'cpu'
        assert len(self.opt['gpu_ids']) <= 1, 'in inference, we only need one gpu'
        if len(self.opt['gpu_ids']) >= 1:
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

        assert name == 'DDPM', "Name must be DDPM!"

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

        self.tfs = transforms.Compose([
            transforms.Resize((img_sz, img_sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.im_crop_sz = img_sz
        ## defect name and its index, 0 is left to UNK label
        self.class_dict = {'lp':1, 'ng':2, 'huashang':3, 'pengshang':4, 'yise':5, 
                           'aokeng':6, 'heidian':7, 'shahenyin':8, 'bianxing':9, 
                           'tabian':10, 'molie':11, 'gubao':12, 'yiwu':13, 'guashang':14, 
                           'caizhixian':15, 'liewen':16, 'daowen':17, 'zhanya':18, 'aotuhen':19,
                           'cashang':20, 'yashang':21, 'madian':22, 'youmo':23,
                           'zangwu':24, 'baidian':25, 'maoxu':26, 'keli':27,
                           'quepeng':28, 'maoci':29, 'queliao':30, 'quepenghua':31,
                           'wuluowen':32, 'zhanliao':33, 'liuwen':34, 'aotu':35,
                           'juchi':36, 'qipao':37, 'zanghua':38, 'kailie':39, 'xianweimao':40,
                           'nzgs':41, 'jiaobuliang':42, 'aotudian':43}

        if "class_name" in opt and opt["class_name"] is not None:
            id_codec = 44
            for name in opt["class_name"]:
                if name in self.class_dict:
                    continue
                self.class_dict[name] = id_codec
                id_codec += 1
  
    def set_input(self, cv_img, cond_img=None, mask=None, label=None, text=None):
        assert cv_img.shape[0] == self.im_crop_sz and cv_img.shape[1] == self.im_crop_sz, "The image's shape is mismatch." 
        pil_crop_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        gt_image = self.tfs(pil_crop_img)
        if mask is None:
            mask = torch.ones(1, self.im_crop_sz, self.im_crop_sz)    
        else:
            mask = torch.from_numpy(mask).float()
        if cond_img is None:
            cond_image = torch.ones(1, self.im_crop_sz, self.im_crop_sz)    
        else:
            cond_image = torch.from_numpy(cond_img).float()
        index = 2 # 2 means ng label
        if label is not None and label in self.class_dict:
            index = self.class_dict[label]
        text = text  ## TODO
        
        ## to GPU
        self.gt_image = self.model.set_device(gt_image.unsqueeze(0))
        self.cond_image = self.model.set_device(cond_image.unsqueeze(0))
        self.mask_image = self.model.set_device(mask.unsqueeze(0))
        self.label = self.model.set_device(torch.tensor([index]))

    @torch.no_grad()
    def restoration_(self, 
                    task='inpaint',
                    y_0=None, 
                    y_cond=None,                             
                    mask=None,
                    label=None,
                    text=None,
                    sample_num=8, 
                    eta=0.0, 
                    ratio=1.0, 
                    gd_w=0.0,
                    seed=None,
                    return_visuals=False):
        '''
        task: inpaint | generate | synthesis
        all data must be tensor, 
        and all data must have been pushed to device
        '''
        
        ## check the inputs' valid
        if task == 'inpaint':
            assert y_0 is not None, 'work in inpait mode, we must have a gt iamge.'
            assert y_cond is not None, 'work in inpait mode, we must have a cond iamge.'
            assert mask is not None, 'work in inpait mode, we must have a mask iamge.'
        
            output, visuals = self.model.netG.inpaint_restoration(y_0=y_0, 
                                                                  y_cond=y_cond,                             
                                                                  mask=mask,
                                                                  label=label,
                                                                  text=text,
                                                                  sample_num=sample_num, 
                                                                  eta=eta, 
                                                                  ratio=ratio, 
                                                                  gd_w=gd_w)
        elif task == 'genetate':
            output, visuals = self.model.netG.generate_restoration(label=label,
                                                                   text=text,
                                                                   sample_num=sample_num, 
                                                                   eta=eta, 
                                                                   gd_w=gd_w,
                                                                   im_sz=self.im_crop_sz,
                                                                   seed=seed,
                                                                   device=self.device)
        
        if return_visuals:
            return output, visuals
        return output

    @torch.no_grad()
    def inpaint_(self, 
                 crop_image, 
                 cond_image,
                 mask,
                 label=None,
                 text=None,
                 ratio=1.0,
                 gd_w=0.0):        
            
        gt_image = self.tfs(Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)))
        cond_image = torch.from_numpy(cond_image).float()
        mask = torch.from_numpy(mask).float()
        
        index = 2 # means ng
        if label is not None and label in self.class_dict:
            index = self.class_dict[label]

        ## to gpu
        y_0 = self.model.set_device(gt_image.unsqueeze(0))
        y_cond = self.model.set_device(cond_image.unsqueeze(0))
        mask = self.model.set_device(mask.unsqueeze(0))
        label = self.model.set_device(torch.tensor([index]))

        output = self.restoration_(task='inpaint',
                                   y_0=y_0, 
                                   y_cond = y_cond, 
                                   mask=mask, 
                                   label=label, 
                                   text=text,
                                   ratio=ratio, 
                                   gd_w=gd_w)
        output = Util.tensor2img(output.cpu())
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output
    
    @torch.no_grad()
    def rnd_sample_(self,                     
                    label=None,
                    text=None,
                    ratio=1.0,
                    gd_w=0.0,
                    seed=None):       
        
        index = 2 # means ng
        if label is not None and label in self.class_dict:
            index = self.class_dict[label]

        ## to gpu
        label = self.model.set_device(torch.tensor([index]))
        ## TODO text

        output = self.restoration_(task='genetate',
                                   label=label, 
                                   text=text,
                                   ratio=ratio, 
                                   gd_w=gd_w,
                                   seed=seed)
        output = Util.tensor2img(output.cpu())
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output

    @torch.no_grad()
    def image2latent(self, 
                     gt_image, 
                     mask_image=None, 
                     cond_image=None, 
                     label=None, 
                     text=None,
                     inference_step=100,
                     ldm_scale_factor=0.31723,
                     return_invert_sample=False):
        '''
        all data is put into gpu, tensor B x C x H x W
        Actually, mask_image and cond_image must be None
        '''
        if mask_image is None:
            mask_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(gt_image.device)
        if cond_image is None:
            cond_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(gt_image.device)

        y_0 = gt_image
        mask = mask_image
        y_cond = cond_image        
        
        ## use_ldm
        if self.model.netG.first_stage_fn is not None:
            self.model.netG.first_stage_fn.eval()
            with torch.no_grad():
                y_0 = self.model.netG.first_stage_fn.encode(gt_image).mode()
                y_0 = y_0 * ldm_scale_factor
                
            scale_factor = 1.0 * y_0.shape[2] / mask_image.shape[2]
            mask = F.interpolate(mask_image, scale_factor=scale_factor, mode="bilinear")
            mask = mask.detach()

            y_cond = F.interpolate(cond_image, scale_factor=scale_factor, mode="bilinear")
            y_cond = y_cond.detach()
        
        context = None
        if self.model.netG.cond_fn is not None:            
            context = self.model.netG.cond_fn(text)

        y_t = y_0   

        if self.model.netG.sample_type == 'ddim': 
            invert_latent_code, ret_arr = self.model.netG.sampler.sample_pro(y_t=y_t,
                                                                         y_cond=y_cond, 
                                                                         y_0=y_0, 
                                                                         mask=mask, 
                                                                         label=label, 
                                                                         context=context,
                                                                         sample_num=1, 
                                                                         ddim_num_steps=self.model.netG.ddim_timesteps,
                                                                         inference_step=inference_step,
                                                                         flag='invert')
            # pdb.set_trace()
        if return_invert_sample:
            y_t = deepcopy(invert_latent_code)            
            y_t, ret_arr = self.model.netG.sampler.sample_pro(y_t=y_t,
                                                              y_cond=y_cond, 
                                                              y_0=y_0, 
                                                              mask=mask, 
                                                              label=label, 
                                                              context=context,
                                                              sample_num=1, 
                                                              ddim_num_steps=self.model.netG.ddim_timesteps,
                                                              inference_step=inference_step,
                                                              flag='sample')
            if self.model.netG.first_stage_fn is not None:
                with torch.no_grad():
                    y_t = 1.0 / ldm_scale_factor * y_t
                    output = self.model.netG.first_stage_fn.decode(y_t)
            output = Util.tensor2img(output.cpu())
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return invert_latent_code, output
        return invert_latent_code
    
    @torch.no_grad()
    def latent2image(self, 
                     latent, 
                     mask_image=None, 
                     cond_image=None, 
                     label=None, 
                     text=None,
                     inference_step=100,
                     ldm_scale_factor=0.31723):
        '''
        all data is put into gpu, tensor B x C x H x W
        Actually, mask_image and cond_image must be None
        '''
        if mask_image is None:
            mask_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(latent.device)
        if cond_image is None:
            cond_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(latent.device)

        y_0 = latent
        mask = mask_image
        y_cond = cond_image        
        
        ## use_ldm
        if self.model.netG.first_stage_fn is not None:                
            scale_factor = 1.0 * y_0.shape[2] / mask_image.shape[2]
            mask = F.interpolate(mask_image, scale_factor=scale_factor, mode="bilinear")
            mask = mask.detach()

            y_cond = F.interpolate(cond_image, scale_factor=scale_factor, mode="bilinear")
            y_cond = y_cond.detach()
        
        context = None
        if self.model.netG.cond_fn is not None:            
            context = self.model.netG.cond_fn(text)

        y_t = y_0   
        y_t, ret_arr = self.model.netG.sampler.sample_pro(y_t=y_t,
                                                          y_cond=y_cond, 
                                                          y_0=y_0, 
                                                          mask=mask, 
                                                          label=label, 
                                                          context=context,
                                                          sample_num=1, 
                                                          gd_w=0.0,
                                                          ddim_num_steps=self.model.netG.ddim_timesteps,
                                                          inference_step=inference_step,
                                                          flag='sample')
        if self.model.netG.first_stage_fn is not None:
            with torch.no_grad():
                y_t = 1.0 / ldm_scale_factor * y_t
                output = self.model.netG.first_stage_fn.decode(y_t)
        output = Util.tensor2img(output.cpu())
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output   
    
    @torch.no_grad()
    def synthesis(self, 
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
                  task=None):
        
        img = cv2.imread(imp)
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)

        IMG_CHANGED, new_img, new_info  = self.synthesis_(img, 
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
    def synthesis_(self, 
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
            if label not in defect_need_gen:
                new_info["shapes"].append(_shape)
                continue    
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
            
            pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            gt_image = self.tfs(pil_crop_img)

            mask = points2mask(new_points, shape_type, label=label, mask_type=mask_type, MIN_PIX=16)  # rect | poly
            mask = torch.from_numpy(mask).float()

            # if the area of mask is too large or NULL, we will skip 
            if mask.sum() <= 0 or mask.sum() / (self.im_crop_sz * self.im_crop_sz) > 0.5:
                new_info["shapes"].append(_shape)
                continue

            cond_image = deepcopy(mask)
            index = 2 # default is 'ng'
            if label in self.class_dict:
                index = self.class_dict[label] 
            if task == 'inpaint':
                label = 'lp'
                index = self.class_dict[label]  # l is lp
                ratio = 1.0 

            ## 
            gt_image = self.model.set_device(gt_image.unsqueeze(0))
            cond_image = self.model.set_device(cond_image.unsqueeze(0))
            mask = self.model.set_device(mask.unsqueeze(0))
            index = self.model.set_device(torch.tensor([index]))
            
            if return_visuals:
                output, visuals = self.model.inpaint_generater(gt_image, cond_image, mask, index, ratio=ratio, gd_w=gd_w, return_visuals=True)
            else:
                output = self.model.inpaint_generater(gt_image, cond_image, mask, index, ratio=ratio, gd_w=gd_w)
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
                img_res = output.copy()
                img_diff = np.abs(img_res - img_show)
                label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                img_res = np.concatenate([img_show, img_res, img_diff], 1)
                
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

    @torch.no_grad()
    def defect_selfaug(self, 
                       imp, 
                       jsp, 
                       defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                       aug_ops=['xy-shift'],
                       ratio=1.0,
                       gd_w=0.0, 
                       prob_syn=1.0,
                       defect_aug_ratio=None,
                       prob_defect2lp=0.2,
                       SHOW=False,
                       mask_type='rect',
                       use_rectangle_labelme=False):
        
        img = cv2.imread(imp)
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)

        IMG_CHANGED, new_img, new_info  = self.defect_selfaug_(img, 
                                                               info, 
                                                               defect_need_gen=defect_need_gen,
                                                               aug_ops=aug_ops,
                                                               ratio=ratio,
                                                               gd_w=gd_w, 
                                                               prob_syn=prob_syn,
                                                               defect_aug_ratio=defect_aug_ratio,
                                                               prob_defect2lp=prob_defect2lp,
                                                               SHOW=SHOW,  
                                                               mask_type=mask_type,
                                                               use_rectangle_labelme=use_rectangle_labelme)
        return IMG_CHANGED, new_img, new_info    
    
    @torch.no_grad()
    def defect_selfaug_(self, 
                        img, 
                        info, 
                        defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                        aug_ops=['xy-shift'],
                        ratio=1.0,
                        gd_w=0.0, 
                        prob_syn=1.0,
                        defect_aug_ratio=None,
                        prob_defect2lp=0.2,
                        SHOW=False,
                        mask_type='rect',
                        use_rectangle_labelme=False):
        
        IMG_CHANGED = False
        new_info = deepcopy(info)
        new_info["shapes"] = list()

        whole_defect_mask = np.zeros((img.shape[0], img.shape[1]))
        for shape in info['shapes']:
            points = shape["points"]
            shape_type = shape["shape_type"]
            image_shape = [img.shape[0], img.shape[1], 3]
            label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=8, point_size=8)
            label_mask = np.where(label_mask == True, 1.0, 0)
            whole_defect_mask += label_mask

        for _shape in info['shapes']:
            label = _shape.get("label")  
            if label not in defect_need_gen:
                new_info["shapes"].append(_shape)
                continue    
            if defect_aug_ratio is not None:
                if label in defect_aug_ratio:
                    ratio = defect_aug_ratio[label]                  

            shape_type = _shape.get("shape_type")
            points = _shape.get("points")
            ret, crop_img, new_points, real_bbox = crop_image(img, points, self.im_crop_sz)                        
            if ret == 0:
                continue

            boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox
            crop_mask = np.zeros((crop_img.shape[0], crop_img.shape[1]))
            crop_mask[0:boxx_y2+1-boxx_y1, 0:boxx_x2+1-boxx_x1] = deepcopy(whole_defect_mask[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1])

            if np.random.rand() > prob_syn:
                new_info["shapes"].append(_shape)
                continue
      
            if SHOW:
                img_show = crop_img.copy()
            
            pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))            
            gt_image = self.tfs(pil_crop_img)
            
            ## preprocess the mask 
            mask = points2mask(new_points, shape_type, label=label, mask_type=mask_type, MIN_PIX=32)  # rect | poly
            if mask.sum() <= 0 or mask.sum() / (self.im_crop_sz * self.im_crop_sz) > 0.5:
                new_info["shapes"].append(_shape)
                continue
            mask_thin = points2mask(new_points, shape_type, label=label, mask_type=mask_type, MIN_PIX=16, line_width=8, point_size=8)
            mask_poly = points2mask(new_points, shape_type, label=label, mask_type='poly', MIN_PIX=8, line_width=8, point_size=8)
            mask = torch.from_numpy(mask).float()
            mask_thin = torch.from_numpy(mask_thin).float()
            mask_poly = torch.from_numpy(mask_poly).float()
            
            ## to device
            cond_image = deepcopy(mask)
            gt_image = self.model.set_device(gt_image.unsqueeze(0))
            cond_image = self.model.set_device(cond_image.unsqueeze(0))
            mask = self.model.set_device(mask.unsqueeze(0))
            mask_thin = self.model.set_device(mask_thin.unsqueeze(0))
            mask_poly = self.model.set_device(mask_poly.unsqueeze(0))
            for p in torch.argwhere(mask_thin > 0):
                crop_mask[p[2], p[3]] = 0.0
            
            ## mask for latent
            mask_interp = F.interpolate(mask, (crop_img.shape[0]//8, crop_img.shape[1]//8), mode="bilinear")
            mask_thin_interp = F.interpolate(mask_thin, (crop_img.shape[0]//8, crop_img.shape[1]//8), mode="bilinear")
            mask_poly_interp = F.interpolate(mask_poly, (crop_img.shape[0]//8, crop_img.shape[1]//8), mode="bilinear")

            ## first, we need transfer defect to ok(lp)
            index = self.model.set_device(torch.tensor([1]))  # index equal to 1 means lp
            output_d2l = self.model.inpaint_generater(gt_image, cond_image, mask, index, ratio=1.0, gd_w=gd_w)
            img_d2l = Util.tensor2img(output_d2l.cpu())
            img_d2l = cv2.cvtColor(img_d2l, cv2.COLOR_RGB2BGR)            
            alpha = mask_poly.squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
            img_d2l = crop_img * (1 - alpha) + img_d2l * alpha
            img_d2l = img_d2l.astype('uint8')
            if np.random.rand() <= prob_defect2lp:                
                img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = img_d2l[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()
                IMG_CHANGED = True

                if SHOW:
                    img_res = img_d2l.copy()
                    img_diff = np.abs(img_res - img_show)
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                    cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                    cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                    img_res = np.concatenate([img_show, img_res, img_diff], 1)
                    
                    cv2.imshow("res0", img_res)
                    cv2.waitKey(0)

                continue

            ## defect latents
            index = 2 # default is 'ng'
            if label in self.class_dict:
                index = self.class_dict[label]
            index = self.model.set_device(torch.tensor([index]))       
            
            invert_steps = int(ratio * self.model.netG.ddim_timesteps)
            lat_defect = self.image2latent(gt_image=gt_image, 
                                           label=index,
                                           mask_image=mask,
                                           cond_image=cond_image,
                                           inference_step=invert_steps, 
                                           return_invert_sample=False)
            
            lat_def2lp = self.image2latent(gt_image=output_d2l, 
                                           label=index,
                                           mask_image=mask,
                                           cond_image=cond_image,
                                           inference_step=invert_steps, 
                                           return_invert_sample=False)

            aug_op = np.random.choice(aug_ops)
            ## random shift
            if aug_op == 'xy-shift': 
                label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                shift_x = np.random.randint(-16, 17) #np.random.choice([-1, 1, 0])
                shift_y = np.random.randint(-16, 17) #np.random.choice([-1, 1, 0])
                if shift_x > 0:
                    while True:
                        if x2 + shift_x <= crop_img.shape[1]:
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
                        if y2 + shift_y <= crop_img.shape[0]:
                            break
                        else:
                            shift_y -= 1
                if shift_y < 0:
                    while True:
                        if y1 + shift_y >= 0:
                            break
                        else:
                            shift_y += 1

                mask_shift = torch.zeros_like(mask)
                mask_poly_shift = torch.zeros_like(mask_poly)
                for p in torch.argwhere(mask > 0):
                    mask_shift[:, :, p[2]+shift_y, p[3]+shift_x] = mask[:, :, p[2], p[3]]
                for p in torch.argwhere(mask_poly > 0):
                    mask_poly_shift[:, :, p[2]+shift_y, p[3]+shift_x] = mask_poly[:, :, p[2], p[3]]
                mask_shift_interp = F.interpolate(mask_shift, (lat_defect.shape[2], lat_defect.shape[3]), mode="bilinear")                
                label_mask = mask_shift_interp.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1_shift, x1_shift), (y2_shift, x2_shift) = pos.min(0), pos.max(0) + 1
                label_mask = mask_interp.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                # label_mask = mask_thin_interp.squeeze(0).squeeze(0).cpu().numpy()
                # pos = np.argwhere(label_mask > 0)
                # (y1_thin, x1_thin), (y2_thin, x2_thin) = pos.min(0), pos.max(0) + 1

                lat_syn = deepcopy(lat_def2lp)

                shift_x_lat = x1_shift - x1
                shift_y_lat = y1_shift - y1
                for p in torch.argwhere(mask_thin_interp > 0):
                    py, px = p[2]+shift_y_lat, p[3]+shift_x_lat
                    px = min(lat_defect.shape[3], max(0, px))
                    py = min(lat_defect.shape[2], max(0, py))
                    lat_syn[:, :, py, px] = lat_defect[:, :, p[2], p[3]]

                

                mask = mask_shift
                cond_image = mask
                
                output = self.latent2image(latent=lat_syn,                                            
                                           label=index, 
                                           mask_image=mask,
                                           cond_image=cond_image,
                                           inference_step=invert_steps)
                alpha = mask_poly_shift.squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
                output = img_d2l * (1 - alpha) + output * alpha
                output = output.astype('uint8')

                if use_rectangle_labelme:
                    _shape["shape_type"] = 'rectangle'            
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
                else:
                    # pass 
                    _shape["points"] = [[p[0]+shift_x, p[1]+shift_y] for p in points]
                    _shape['group_id'] = 'aug_xyshift'
            elif aug_op == 'flip':
                lat_syn = deepcopy(lat_def2lp)

                label_mask = mask_thin.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1_thin, x1_thin), (y2_thin, x2_thin) = pos.min(0), pos.max(0) + 1
                
                label_mask = mask_thin_interp.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                iou_lat = deepcopy(lat_defect[:, :, y1:y2, x1:x2])
                # iou_mask_lat = deepcopy(mask_poly_interp[:, :, y1:y2, x1:x2])
                if x2 - x1 >= y2 - y1:                    
                    iou_lat = iou_lat.flip(dims=[3])
                    # iou_mask_lat = iou_mask_lat.flip(dims=[3])                    
                    w = x2_thin - x1_thin
                    new_points = [[w - (pp[0] - x1_thin - boxx_x1) + x1_thin + boxx_x1, pp[1]] for pp in points]
                else:
                    iou_lat = iou_lat.flip(dims=[2])
                    # iou_mask_lat = iou_mask_lat.flip(dims=[2])
                    h = y2_thin - y1_thin
                    new_points = [[pp[0], h - (pp[1] - y1_thin - boxx_y1) + y1_thin + boxx_y1] for pp in points]

                # lat_syn[:, :, y1:y2, x1:x2] = lat_syn[:, :, y1:y2, x1:x2] * (1 - iou_mask_lat) + iou_lat * iou_mask_lat
                lat_syn[:, :, y1:y2, x1:x2] = iou_lat

                output = self.latent2image(latent=lat_syn, 
                                           mask_image=mask,
                                           cond_image=cond_image,
                                           label=index, 
                                           inference_step=invert_steps)
                alpha = mask_thin.squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
                output = img_d2l * (1 - alpha) + output * alpha
                output = output.astype('uint8')

                if use_rectangle_labelme:
                    _shape["shape_type"] = 'rectangle'            
                    label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                    pos = np.argwhere(label_mask > 0)
                    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                    _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
                else:
                    # pass 
                    _shape["points"] = new_points
                    _shape['group_id'] = 'aug_flip'

            else:
                assert aug_op in ['xy-shift', 'flip', 'resize', 'rotate'], 'Oops, we do not support this aug optate!!!'


            new_info["shapes"].append(_shape) 
         
            img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = output[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()
            IMG_CHANGED = True

            if SHOW:
                img_res = output.copy()
                img_diff = np.abs(img_res - img_show)
                label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
                cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                img_res = np.concatenate([img_show, img_res, img_diff], 1)
                
                cv2.imshow("res", img_res)
                cv2.waitKey(0)

        return IMG_CHANGED, img, new_info    
    
if __name__ == "__main__":
    config = './config/infer/infer_inpainting_ldm_ch192.json'    
    engine = Engine(config) 

    ## random sample
    for seed in range(10):
        output1 = engine.rnd_sample_(label='lp', seed=seed, gd_w=0.0)
        output2 = engine.rnd_sample_(label='ng', seed=seed, gd_w=0.0)
        output3 = engine.rnd_sample_(label='pengshang', seed=seed, gd_w=0.0)
        output = np.concatenate([output1, output2, output3], 1)
        cv2.imshow('rand sample', output)
        cv2.waitKey(0)

    ## inpaint
    imp = './test/aokeng_0_damian_0372-0008-01.jpg'
    import glob
    for imp in glob.glob('./test/' + '*.jpg'):
        jsp = imp.replace('.jpg', '.json')
        cv_img = cv2.imread(imp)
        with open(jsp, 'r', encoding='utf-8') as jf:
            info = json.load(jf)
        shape = info["shapes"][0]
        label = shape['label']
        points = shape['points']
        shape_type = shape['shape_type']
        mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0])
        cond_img = mask
        output = engine.inpaint_(crop_image=cv_img, 
                                cond_image=cond_img,
                                mask=mask,
                                label=label,
                                gd_w=0.0)
        output_lp = engine.inpaint_(crop_image=cv_img, 
                                cond_image=cond_img,
                                mask=mask,
                                label='lp',
                                gd_w=0.0)
        cv_img_msk = cv_img * (1 - mask.transpose(1, 2, 0))
        cv_img_msk = cv_img_msk.astype('uint8')
        diff = np.abs(cv_img - output)
        # im_res = cv_img
        im_res = np.concatenate([cv_img, cv_img_msk, output, output_lp, diff], 1)
        cv2.imshow('res_inpaint', im_res)
        cv2.waitKey(0)

    ## invert
    invert_steps = 30
    imp = './test/693_16903_cemian_0224-0008-07.jpg'
    jsp = imp.replace('.jpg', '.json')
    cv_img = cv2.imread(imp)
    with open(jsp, 'r', encoding='utf-8') as jf:
        info = json.load(jf)
    shape = info["shapes"][0]
    label = shape['label']
    points = shape['points']
    shape_type = shape['shape_type']
    mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0], mask_type='poly')
    engine.set_input(cv_img=cv_img, label=label, mask=mask)
    latent_codes = engine.image2latent(gt_image=engine.gt_image, 
                                       label=engine.label,
                                       inference_step=invert_steps, 
                                       return_invert_sample=False)
    output = engine.latent2image(latent=latent_codes, label=engine.label, 
                                 inference_step=invert_steps)
    
    
    ## edit 
    points = shape['points']
    shape_type = shape['shape_type']
    mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0], mask_type='poly')
    mask_interp = torch.from_numpy(mask).float().to(latent_codes.device).unsqueeze(0)
    mask_interp = F.interpolate(mask_interp, (32, 32), mode="bilinear")
    
    ### xy-shift
    opt_latent_codes = deepcopy(latent_codes)
    for p in torch.argwhere(mask_interp > 0):
        opt_latent_codes[:, :, p[2]-5, p[3]-6] = latent_codes[:, :, p[2], p[3]]
    output_xy_shift = engine.latent2image(latent=opt_latent_codes, label=engine.label, inference_step=invert_steps)

    ### rotate
    tfs = transforms.RandomRotation(degrees=(10, 15))
    opt_latent_codes = tfs(latent_codes)
    output_rotate = engine.latent2image(latent=opt_latent_codes, label=engine.label, inference_step=invert_steps)

    ### resize
    opt_latent_codes = F.interpolate(latent_codes, (48, 48))
    opt_latent_codes = opt_latent_codes[:, :, 8:40, 8:40]
    output_resize = engine.latent2image(latent=opt_latent_codes, label=engine.label, inference_step=invert_steps)

    ### norm
    opt_latent_codes = deepcopy(latent_codes)
    opt_latent_codes = (opt_latent_codes - latent_codes.mean()) / latent_codes.std()
    # pdb.set_trace()
    output_norm = engine.latent2image(latent=opt_latent_codes, label=engine.label, inference_step=invert_steps)

    # diff = np.abs(cv_img - output)
    im_res = np.concatenate([cv_img, output, output_xy_shift, output_rotate, output_resize, output_norm], 1)

    
    cv2.imshow('res_invert', im_res)
    cv2.waitKey(0)



        