import warnings
import torch
import json
from collections import OrderedDict
from core.praser import dict_to_nonedict

import core.util as Util
from models.model import MaLiang
from PIL import Image, ImageDraw
# from labelme.utils.shape import shape_to_mask, masks_to_bboxes
from torchvision import transforms
import numpy as np
import cv2
from copy import deepcopy
import math
import os

import pdb

def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "polygon" and len(xy) == 2:
        shape_type = "linestrip"
    if shape_type == "polygon" and len(xy) == 1:
        shape_type = "point"   
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1, width=line_width//2)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1, width=line_width//2)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1, width=line_width//2)
    mask = np.array(mask, dtype=bool)
    
    return mask

def parse(config):
    # pdb.set_trace()
    json_str = ''
    with open(config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    # pdb.set_trace()
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    ''' replace the config context using args '''
    opt['phase'] = "test"
    opt['datasets'] = {'test': {'dataloader': {'args': {'batch_size': 1}}}}

    ''' set cuda environment '''
    assert len(opt['gpu_ids']) == 1, 'Do not support distributed mode in infer processing.'
    opt['distributed'] = False
    # if len(opt['gpu_ids']) > 1:
    #     opt['distributed'] = True
    # else:
    #     opt['distributed'] = False

    return dict_to_nonedict(opt)

class Engine():
    def __init__(self, config_path, img_sz=256):
        opt = parse(config_path)
        if 'local_rank' not in opt:
            opt['local_rank'] = opt['global_rank'] = 0

        '''set seed and and cuDNN environment '''
        torch.backends.cudnn.enabled = False
        if torch.backends.cudnn.enabled:
            warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
        Util.set_seed(opt['seed'])

        '''network'''
        net_opt = opt['model']['which_networks'][0].get('args', {})
        name = opt['model']['which_networks'][0].get('name', {})[1]
        if name == 'DDPM':
            from models.ddpm import DDPM
            net = DDPM(**net_opt)
        elif name == 'Network':
            from models.network import Network
            net = Network(**net_opt)
        else:
            assert name == 'DDPM' or name == 'Network', "name must be DDPM or Nerwork!"
        net.__name__  = net.__class__.__name__

        self.model = MaLiang(networks=[net],
                             losses=["mse_loss"],
                             sample_num=8,
                             task='inpainting',
                             optimizers=[{"lr": 5e-5, "weight_decay": 0}],
                             ema_scheduler=None,
                             opt=opt,
                             phase_loader=None, 
                             val_loader=None,
                             metrics=None,
                             logger=None,
                             writer=None
                            )
        self.__name__  = self.__class__.__name__

        self.tfs = transforms.Compose([
                transforms.Resize((img_sz, img_sz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.im_crop_sz = img_sz
        # self.class_dict = {'huashang':1, 'pengshang':2, 'yise':3, 'aokeng':4, 'heidian':5}
        self.class_dict = {'huashang':1, 
                            'pengshang':2, 
                            'yise':3, 
                            'aokeng':4, 
                            'heidian':5,
                            'shahenyin':6, 
                            'bianxing':7, 
                            'tabian':8, 
                            'molie':9,
                            'gubao':10, 
                            'yiwu':11, 
                            'guashang':12, 
                            'caizhixian':13, 
                            'liewen':14,
                            'daowen':15, 
                            'zhanya':16, 
                            'aotuhen':17,
                            'cashang':18,
                            'yashang':19, 
                            'madian':20, 
                            'youmo':21,
                            'zangwu':22,
                            'baidian':23,
                            'maoxu':24,
                            'keli':25,
                            'quepeng':26,
                            'maoci':27,
                            'queliao':28,
                            'quepenghua':29,
                            'wuluowen':30,
                            'zhanliao':31,
                            'liuwen':32,
                            'aotu':33,
                            'juchi':34,
                            'qipao':35,
                            'zanghua':36,
                            'kailie':37,
                            'xianweimao':38,
                            'nzgs':39,
                            'jiaobuliang':40,
                            'aotudian':41}
        
    def inpaint(self, imp, jsp, 
                defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                gid_need_gen=[1000, 2000],
                SHOW=False,
                SAVE=False,
                SAVE_DIR=None,
                USE_SYN=False,
                ratio=1.0,
                gd_w=0.0):
        img = cv2.imread(imp, 1)
        with open(jsp, 'r', encoding='utf-8') as jf:
            info = json.load(jf)
        IMG_CHANGED, img, new_info = self.inpaint_(img, info, 
                                                   defect_need_gen=defect_need_gen,
                                                   gid_need_gen=gid_need_gen,
                                                   SHOW=SHOW,
                                                   SAVE=SAVE,
                                                   SAVE_DIR=SAVE_DIR,
                                                   USE_SYN=USE_SYN,
                                                   ratio=ratio,
                                                   gd_w=gd_w)
        
        return IMG_CHANGED, img, new_info

    def inpaint_(self, img, info, 
                defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                gid_need_gen=[1000, 2000],
                SHOW=False,
                SAVE=False,
                SAVE_DIR=None,
                USE_SYN=False,
                ratio=1.0,
                gd_w=0.0):
        IMG_CHANGED = False
        # img = cv2.imread(imp, 1)
        # with open(jsp, 'r', encoding='utf-8') as jf:
        #     info = json.load(jf)
        new_info = deepcopy(info)
        new_info["imageData"] = None
        new_info["shapes"] = list()

        for _shape in info['shapes']:
            label = _shape.get("label")
            # pdb.set_trace()
            if label not in self.class_dict:                
                new_info["shapes"].append(_shape)
                continue
            if label not in defect_need_gen:
                if _shape.get("group_id") in gid_need_gen:
                    continue
                new_info["shapes"].append(_shape)
                continue

            group_id = _shape.get("group_id")
            if len(gid_need_gen) > 0 and group_id not in gid_need_gen:
                new_info["shapes"].append(_shape)
                continue

            shape_type = _shape.get("shape_type")
            points = _shape.get("points")
            # pdb.set_trace()
            ret, crop_img, new_points, real_bbox = self.crop_image(img, points, self.im_crop_sz)
            if ret == 0:
                if len(gid_need_gen) <= 0:
                    new_info["shapes"].append(_shape)
                continue
      
            if SHOW or SAVE:
                img_show = crop_img.copy()
            
            pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            gt_image = self.tfs(pil_crop_img)
            mask = self.points2mask(new_points, shape_type, label=label, mask_type='rect')

            # if the area of mask is too large, we will skip 
            if mask.sum() / (self.im_crop_sz * self.im_crop_sz) > 0.5:
                if len(gid_need_gen) <= 0:
                    new_info["shapes"].append(_shape)
                continue
            
            cond_image = gt_image * (1. - mask) + mask * torch.randn_like(gt_image)
            if USE_SYN:
                cond_image = 0.0065 * gt_image + torch.randn_like(gt_image)
                cond_image = gt_image * (1. - mask) + mask * cond_image
            index = self.class_dict[label]

            ## 
            gt_image = self.model.set_device(gt_image.unsqueeze(0))
            cond_image = self.model.set_device(cond_image.unsqueeze(0))
            mask = self.model.set_device(mask.unsqueeze(0))
            index = self.model.set_device(torch.tensor([index]))

            output = self.model.generater(gt_image, cond_image, mask, index, ratio=ratio, gd_w=gd_w)
            output = Util.tensor2img(output.cpu())
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # check the diff
            mask_check = mask.squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
            diff = np.abs(output / 255.0 - crop_img / 255.0) * mask_check
            diff_sq = diff * diff

            ori = crop_img / 255.0 * mask_check
            ori_sq = ori * ori

            diff_ratio = diff_sq.sum() / (ori_sq.sum() + 1e-20)

            if diff_ratio < 0.001:
                continue

            boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox 
            _shape["shape_type"] = 'rectangle'
            label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
            pos = np.argwhere(label_mask > 0)
            (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
            _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
            # pdb.set_trace()

            new_info["shapes"].append(_shape)
            
            img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = output[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()

            IMG_CHANGED = True

            if SHOW:
                img_res = output.copy()
                img_diff = np.abs(img_res - img_show)
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                img_res = np.concatenate([img_show, img_res, img_diff], 1)
                
                cv2.imshow("res", img_res)
                cv2.waitKey(0)
            if SAVE:
                assert SAVE_DIR is not None, 'save_dir must have a path.'
                os.makedirs(SAVE_DIR, exist_ok=True)
                img_res = output.copy()
                cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                img_res = np.concatenate([img_show, img_res], 1)
                index = 0
                save_name = os.path.join(SAVE_DIR, label + '_' + str(index) + '.jpg')
                while os.path.exists(save_name):
                    index += 1
                    save_name = os.path.join(SAVE_DIR, label + '_' + str(index) + '.jpg')
                cv2.imwrite(save_name, img_res)


        return IMG_CHANGED, img, new_info

    def crop_image(self, img, points, im_crop_sz):
        x_list = [k[0] for k in points]
        y_list = [k[1] for k in points]

        x1 = min(x_list)
        y1 = min(y_list)
        x2 = max(x_list)
        y2 = max(y_list)

        if (x2 - x1 + 1) > im_crop_sz or (y2 - y1 + 1) > im_crop_sz:
            return 0, None, None, None

        cx = (x2 - x1 + 1) / 2.0 + x1
        cy = (y2 - y1 + 1) / 2.0 + y1 

        boxx_x1 = max(0, int(cx - im_crop_sz / 2.0))
        boxx_y1 = max(0, int(cy - im_crop_sz / 2.0))
        boxx_x2 = min(img.shape[1]-1, boxx_x1 + im_crop_sz - 1)
        boxx_y2 = min(img.shape[0]-1, boxx_y1 + im_crop_sz - 1)
        real_bbox =(boxx_x1, boxx_y1, boxx_x2, boxx_y2)

        crop_img = img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :].copy()
        if crop_img.shape[0] < im_crop_sz:
            crop_img = cv2.copyMakeBorder(crop_img, 0, im_crop_sz - crop_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if crop_img.shape[1] < im_crop_sz:
            crop_img = cv2.copyMakeBorder(crop_img, 0, 0, 0, im_crop_sz - crop_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

        new_pionts = list()
        for i in range(len(x_list)):
            x = min(im_crop_sz - 1, max(0, x_list[i] - boxx_x1))
            y = min(im_crop_sz - 1, max(0, y_list[i] - boxx_y1))
            new_pionts.append([x,y])

        return 1, crop_img, new_pionts, real_bbox

    def points2mask(self, new_points, shape_type, label='huashang', mask_type='rect'):
        image_shape = [self.im_crop_sz, self.im_crop_sz, 3]
        mask = np.zeros((self.im_crop_sz, self.im_crop_sz, 1))
        MIN_PIX = 16

        if shape_type == 'rectangle':
            label_mask = 255 * np.zeros((self.im_crop_sz, self.im_crop_sz)).astype('uint8')
            # label_mask = cv2.dilate(label_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
            (x1, y1), (x2, y2) = np.array(new_points).min(0), np.array(new_points).max(0)
            label_mask[int(y1):int(y2), int(x1):int(x2)] = 255
        else:
            try:
                label_mask = shape_to_mask(image_shape, new_points, shape_type=shape_type, line_width=16, point_size=8)
            except:
                mask = torch.from_numpy(mask).float().permute(2, 0, 1)
                return mask
            label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

            # if label == 'huashang' and shape_type == 'polygon':
            #     label_mask = cv2.dilate(label_mask, np.ones((8, 8), dtype=np.uint8), iterations=1)


            # label_mask = cv2.dilate(label_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
            # label_mask = cv2.erode(label_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        pos = np.argwhere(label_mask > 20)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1

        h = y2 - y1
        w = x2 - x1
        min_h_w = min(h, w)
        if min_h_w < MIN_PIX:
            delta = MIN_PIX - min_h_w + 1
            label_mask = cv2.dilate(label_mask, np.ones((delta, delta), dtype=np.uint8), iterations=1)

        if mask_type == 'poly' or label == 'huashang':
            mask = cv2.GaussianBlur(mask, (5, 5), 0, 0)
            mask = label_mask / 255.0
            mask = np.expand_dims(mask, -1)
            mask = torch.from_numpy(mask).float().permute(2, 0, 1)

            return mask
        
        where = np.argwhere(label_mask > 20)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        mask[y1:y2, x1:x2] = 255
        mask = cv2.GaussianBlur(mask, (5, 5), 0, 0)
        mask = np.expand_dims(mask, -1)

        mask = mask / 255.0
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)

        return mask
    
    def synthesis_(self, img, info, 
                defect_need_gen=['huashang', 'pengshang', 'yise', 'aokeng', 'heidian'],
                gid_need_gen=[1000, 2000],
                SHOW=False,
                SAVE=False,
                SAVE_DIR=None,
                USE_SYN=False,
                ratio=1.0,
                gd_w=0.0,
                prob=10.0,
                defect_aug_prob=None,
                defect_aug_ratio=None,
                special_label=None,
                special_gid=None,
                return_visuals=False):
        IMG_CHANGED = False
        new_info = deepcopy(info)
        new_info["shapes"] = list()

        for _shape in info['shapes']:
            label = _shape.get("label")      
            if defect_aug_ratio is not None:
                if label in defect_aug_ratio:
                    ratio = defect_aug_ratio[label]      
            if special_label is not None and label in special_label:
                ratio = 1.0
            if label not in defect_need_gen:
                new_info["shapes"].append(_shape)
                continue
            # if label not in self.class_dict:                
            #     new_info["shapes"].append(_shape)
            #     continue            

            group_id = _shape.get("group_id")
            if special_gid is not None and group_id in special_gid:
                ratio = 1.0
            if len(gid_need_gen) > 0 and group_id not in gid_need_gen:
                new_info["shapes"].append(_shape)
                continue

            shape_type = _shape.get("shape_type")
            points = _shape.get("points")
            # pdb.set_trace()
            ret, crop_img, new_points, real_bbox = self.crop_image(img, points, self.im_crop_sz)
            if ret == 0:
                if len(gid_need_gen) <= 0:
                    new_info["shapes"].append(_shape)
                continue
      
            if SHOW or SAVE or return_visuals:
                img_show = crop_img.copy()
            
            pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            gt_image = self.tfs(pil_crop_img)

            mask = self.points2mask(new_points, shape_type, label=label, mask_type='rect')
            # pdb.set_trace()

            # if the area of mask is too large, we will skip 
            if mask.sum() <= 0 or mask.sum() / (self.im_crop_sz * self.im_crop_sz) > 0.5:
                new_info["shapes"].append(_shape)
                continue

            # prob. select to aug
            if defect_aug_prob is not None:
                if label in defect_aug_prob:
                    prob = defect_aug_prob[label]
            if np.random.rand() > prob:
                new_info["shapes"].append(_shape)
                continue
            
            cond_image = gt_image * (1. - mask) + mask * torch.randn_like(gt_image)
            if USE_SYN:
                cond_image = 0.0065 * gt_image + torch.randn_like(gt_image)
                cond_image = gt_image * (1. - mask) + mask * cond_image

            index = 0
            if label in self.class_dict:
                index = self.class_dict[label] 

            ## 
            gt_image = self.model.set_device(gt_image.unsqueeze(0))
            cond_image = self.model.set_device(cond_image.unsqueeze(0))
            mask = self.model.set_device(mask.unsqueeze(0))
            index = self.model.set_device(torch.tensor([index]))
            
            if return_visuals:
                output, visuals = self.model.generater(gt_image, cond_image, mask, index, ratio=ratio, gd_w=gd_w, return_visuals=True)
            else:
                output = self.model.generater(gt_image, cond_image, mask, index, ratio=ratio, gd_w=gd_w)
            output = Util.tensor2img(output.cpu())
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            boxx_x1, boxx_y1, boxx_x2, boxx_y2 = real_bbox 
            _shape["shape_type"] = 'rectangle'
            
            label_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
            # pdb.set_trace()
            pos = np.argwhere(label_mask > 0)
            (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
            _shape["points"] = [[float(x1) + boxx_x1, float(y1) + boxx_y1], [float(x2) + boxx_x1, float(y2) + boxx_y1]]
            # pdb.set_trace()

            if special_gid is None:
                _shape["group_id"] = 1000

            new_info["shapes"].append(_shape)
            
            img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :] = output[:boxx_y2-boxx_y1+1, :boxx_x2-boxx_x1+1, :].copy()

            IMG_CHANGED = True

            if SHOW:
                img_res = output.copy()
                img_diff = np.abs(img_res - img_show)
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                img_res = np.concatenate([img_show, img_res, img_diff], 1)
                
                cv2.imshow("res", img_res)
                cv2.waitKey(0)
            if SAVE:
                assert SAVE_DIR is not None, 'save_dir must have a path.'
                img_res = output.copy()
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                img_res = np.concatenate([img_show, img_res], 1)
                index = 0
                save_name = os.path.join(SAVE_DIR, label + '_' + str(index) + '.jpg')
                while os.path.exists(save_name):
                    index += 1
                    save_name = os.path.join(SAVE_DIR, label + '_' + str(index) + '.jpg')
                cv2.imwrite(save_name, img_res)

                # img_res = output.copy()
                # cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                # cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                # index = 0
                # save_name = os.path.join(SAVE_DIR, 'aug_' + str(index) + '.jpg')
                # while os.path.exists(save_name):
                #     index += 1
                #     save_name = os.path.join(SAVE_DIR, 'aug_' + str(index) + '.jpg')
                # cv2.imwrite(save_name, img_show)
                # save_name = save_name.replace('aug_', 'syn_')
                # cv2.imwrite(save_name, img_res)
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
