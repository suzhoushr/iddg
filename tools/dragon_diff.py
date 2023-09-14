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
from core.praser import dict_to_nonedict
from models.model import MaLiang
from tools.engine import Engine
from utils.img_proc.image_process import shape_to_mask, crop_image, points2mask
import os
import glob
import pdb

class DragonDiff():
    def __init__(self, opt, img_sz=256):        
        self.Gud = Engine(config=opt, img_sz=img_sz)
        self.Gen = Engine(config=opt, img_sz=img_sz)
    
    @torch.no_grad()
    def object_move(self, 
                    cv_image,
                    gud_mask,
                    gen_mask,
                    label=None,
                    text=None,
                    ):
        return
    
    @torch.no_grad()
    def object_resize()
    
if __name__ == "__main__":
    config = './config/infer/infer_inpainting_ldm_ch256.json'    
    engine = Engine(config) 

    ## random sample
    # for seed in range(10):
    #     output1 = engine.rnd_sample_(label='lp', seed=seed)
    #     output2 = engine.rnd_sample_(label='ng', seed=seed)
    #     output = np.concatenate([output1, output2], 1)
    #     cv2.imshow('rand sample', output)
    #     cv2.waitKey(0)

    ## inpaint
    # imp = './test/aokeng_0_damian_0372-0008-01.jpg'
    # for imp in glob.glob('./test/' + '*.jpg'):
    #     jsp = imp.replace('.jpg', '.json')
    #     cv_img = cv2.imread(imp)
    #     with open(jsp, 'r', encoding='utf-8') as jf:
    #         info = json.load(jf)
    #     shape = info["shapes"][0]
    #     label = shape['label']
    #     points = shape['points']
    #     shape_type = shape['shape_type']
    #     mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0])
    #     cond_img = mask
    #     output = engine.inpaint_(crop_image=cv_img, 
    #                             cond_image=cond_img,
    #                             mask=mask,
    #                             label=label)
    #     output_lp = engine.inpaint_(crop_image=cv_img, 
    #                             cond_image=cond_img,
    #                             mask=mask,
    #                             label='lp')
    #     cv_img_msk = cv_img * (1 - mask.transpose(1, 2, 0))
    #     cv_img_msk = cv_img_msk.astype('uint8')
    #     diff = np.abs(cv_img - output)
    #     # im_res = cv_img
    #     im_res = np.concatenate([cv_img, cv_img_msk, output, output_lp, diff], 1)
    #     cv2.imshow('res_inpaint', im_res)
    #     cv2.waitKey(0)

    ## invert
    imp = './test/693_16903_cemian_0224-0008-07.jpg'
    jsp = imp.replace('.jpg', '.json')
    cv_img = cv2.imread(imp)
    with open(jsp, 'r', encoding='utf-8') as jf:
        info = json.load(jf)
    shape = info["shapes"][0]
    label = shape['label']
    engine.set_input(cv_img=cv_img, label=label)
    latent_codes = engine.image2latent(gt_image=engine.gt_image, 
                                       label=engine.label,
                                       inference_step=50, 
                                       return_invert_sample=False)
    ### 
    points = shape['points']
    shape_type = shape['shape_type']
    mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0], mask_type='poly')
    mask_interp = torch.from_numpy(mask).float().to(latent_codes.device).unsqueeze(0)
    mask_interp = F.interpolate(mask_interp, (32, 32), mode="bilinear")
    opt_latent_codes = deepcopy(latent_codes)
    for p in torch.argwhere(mask_interp > 0):
        opt_latent_codes[:, :, p[2], p[3]-10] = latent_codes[:, :, p[2], p[3]]
    latent_codes = opt_latent_codes

    output = engine.latent2image(latent=latent_codes, label=engine.label, inference_step=50)
    diff = np.abs(cv_img - output)
    im_res = np.concatenate([cv_img, output, diff], 1)
    cv2.imshow('res_invert', im_res)
    cv2.waitKey(0)



        