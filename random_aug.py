import torch
import core.util as Util
from PIL import Image, ImageEnhance
import os
import torch
import numpy as np
import json
import cv2
from utils.img_proc.image_process import shape_to_mask, random_crop
from imagecorruptions import corrupt, get_corruption_names

import pdb


if __name__ == "__main__":
    ## config
    image_size = 256
    index = 0

    train_file = 'datasets/history_dataset/train_ldm_his_all.flist'
    for line in open(train_file, 'r'):
        imp = line.replace('\n', '').strip()

        while True:
            im_crop, info_crop = random_crop(imp=imp, crop_shape=(image_size, image_size))

            ## im_crop is defect image
            if 'shapes' in info_crop and len(info_crop["shapes"]) > 0:
                break

        ## random aug
        img_pil = Image.fromarray(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))  
        min_th, max_th = 5, 18
        prob_aug = 1.0
        if torch.rand((1,)).item() < prob_aug:
            brightEnhancer = ImageEnhance.Brightness(img_pil)
            bright = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = brightEnhancer.enhance(bright)

        if torch.rand((1,)).item() < prob_aug:
            contrastEnhancer = ImageEnhance.Contrast(img_pil)
            contrast = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = contrastEnhancer.enhance(contrast)

        if torch.rand((1,)).item() < prob_aug:
            colorEnhancer = ImageEnhance.Color(img_pil)
            color = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = colorEnhancer.enhance(color)

        if torch.rand((1,)).item() < prob_aug:
            SharpnessEnhancer = ImageEnhance.Sharpness(img_pil)
            sharpness = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = SharpnessEnhancer.enhance(sharpness)

        im_aug = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)

        for shape in info_crop['shapes']:
            image_shape = [image_size, image_size, 3]
            points = shape["points"]
            shape_type = shape["shape_type"]
            label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
            label_inst = np.where(label_inst == True, 255, 0).astype('uint8')
            label_inst = cv2.dilate(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
            label_inst = cv2.erode(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
            label_inst = np.expand_dims(label_inst, -1) / 255.0
            
            im_cond = im_crop * (1 - label_inst) + im_aug * label_inst
            im_cond = im_cond.astype('uint8')
            
            # im_res = np.concatenate([im_crop, im_aug, im_cond], 1)
            # cv2.imshow('res', im_res)
            # cv2.waitKey(0)

            cv2.imwrite('test/random_aug/'+str(index)+'_gt.jpg', im_crop)
            cv2.imwrite('test/random_aug/'+str(index)+'_aug.jpg', im_aug)
            cv2.imwrite('test/random_aug/'+str(index)+'_cond.jpg', im_cond)
            index += 1
            
    