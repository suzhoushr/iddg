import sys

sys.path.append("")
from tools.engine import Engine
import json
import cv2
import pdb
import os
import sys
import argparse
from core.praser import update_config
from utils.img_proc.image_process import random_crop, points2mask
import numpy as np
import random
import copy
import glob
from tqdm import tqdm
import torch.nn.functional as F

if __name__ == '__main__':
    config = './config/infer/infer_inpainting_ldm_ch192.json'
    engine = Engine(opt=config, img_sz=256)
    
    dir_defect = '/home/data0/project-datasets/micro_o_628/datasets/train_val_data_crop4lora_sz256/train/train_O/'
    dir_lp = '/home/data0/project-datasets/micro_o_628/datasets/lp/206/original_pictures/'
    defect_imgs = glob.glob(dir_defect + '*.jpg')
    lp_imgs = glob.glob(dir_lp + '*.jpg')
    random.shuffle(defect_imgs)

    invert_steps = 100
    for imp in tqdm(defect_imgs):        

        jsp = imp.replace('.jpg', '.json')
        if not os.path.exists(jsp):
            continue
        img_defect = cv2.imread(imp)
        with open(jsp, 'r', encoding='utf-8') as jf:
            info_defect = json.load(jf)
        shape = info_defect["shapes"][0]
        label = shape['label']
        points = shape['points']
        shape_type = shape['shape_type']
        mask = points2mask(points, shape_type, label, im_crop_sz=img_defect.shape[0], mask_type='rect')
        engine.set_input(cv_img=img_defect, label=label, mask=mask, cond_img=mask)
        defect_latent_codes, img_defect_ivt = engine.image2latent(gt_image=engine.gt_image, 
                                                  mask_image=engine.mask_image,
                                                  cond_image=engine.cond_image,
                                                  label=engine.label,                                           
                                                  inference_step=invert_steps, 
                                                  return_invert_sample=True)
        
        imp_lp = np.random.choice(lp_imgs)   
        img_lp_crop, _ = random_crop(imp=imp_lp, crop_shape=(256, 256))
        engine.set_input(cv_img=img_lp_crop, label=label, mask=mask, cond_img=mask)        
        lp_latent_codes, img_lp_crop_ivt = engine.image2latent(gt_image=engine.gt_image, 
                                              mask_image=engine.mask_image,
                                              cond_image=engine.cond_image,
                                              label=engine.label,
                                              inference_step=invert_steps, 
                                              return_invert_sample=True)
        label_mask = engine.mask_image.squeeze(0).squeeze(0).cpu().numpy()
        pos = np.argwhere(label_mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
        cv2.rectangle(img_defect, (x1, y1), (x2, y2), (0, 0, 255), 1, 4)
        cv2.rectangle(img_lp_crop_ivt, (x1, y1), (x2, y2), (0, 0, 255), 1, 4)
        img_res = np.concatenate([img_defect, img_defect_ivt, img_lp_crop, img_lp_crop_ivt], 1)
        cv2.imshow('com', img_res)
        cv2.waitKey(0)
        
        # defect_latent_codes = (defect_latent_codes - defect_latent_codes.mean()) / (defect_latent_codes.std() + 1e-20)
        # defect_latent_codes = defect_latent_codes * lp_latent_codes.std() + lp_latent_codes.mean()
        mask = F.interpolate(engine.mask_image, defect_latent_codes.shape[2:])
        syn_latent_codes = lp_latent_codes * (1 - mask) + defect_latent_codes * mask
        output = engine.latent2image(latent=syn_latent_codes, 
                                     mask_image=engine.mask_image,
                                     cond_image=engine.cond_image,
                                     label=engine.label, 
                                     inference_step=invert_steps)

        img_res = output.copy()
        label_mask = engine.mask_image.squeeze(0).squeeze(0).cpu().numpy()
        img_res = img_lp_crop * (1 - label_mask[:,:,None]) + output * label_mask[:,:,None]
        img_res = img_res.astype('uint8')
        pos = np.argwhere(label_mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
        cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
        cv2.rectangle(img_defect, (x1, y1), (x2, y2), (0, 0, 255), 1, 4)
        cv2.putText(img_res, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
        cv2.putText(img_defect, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
        img_res = np.concatenate([img_defect, img_lp_crop, img_res], 1)
        cv2.imshow('res', img_res)
        cv2.waitKey(0)


