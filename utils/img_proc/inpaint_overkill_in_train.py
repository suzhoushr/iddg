import sys
sys.path.append('./')
import cv2
import os
import numpy as np
from tqdm import tqdm
import json
import shutil
import pdb

if __name__ == "__main__":
    src_dir = '/home/data0/project-datasets/dehongcheng/exp_SYB/train_val_data/train/'
    pred_dir = '/home/data0/project-datasets/dehongcheng/exp_SYB/test/train_infer_ana_0001/'
    save_dir = '/home/data0/project-datasets/dehongcheng/exp_SYB/train_val_data/train_inpaint_kill/'
  
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    imgs_list = list()
    for filepath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if '.jpg' in filename:
                imp = os.path.join(filepath, filename)
                imgs_list.append(imp)

    for imp in tqdm(imgs_list):
        jsp = imp.replace('.jpg', '.json')
        pred_jsp = jsp.replace(src_dir, pred_dir)

        img = cv2.imread(imp)
        with open(pred_jsp, "r", encoding="utf-8") as fj:
            info = json.load(fj)

        for shape in info['shapes']:
            if 'PRED_kill' == shape['group_id']:
                points = shape['points']
                x_list = [k[0] for k in points]
                y_list = [k[1] for k in points]

                x1 = int(min(x_list))
                y1 = int(min(y_list))
                x2 = int(max(x_list))
                y2 = int(max(y_list))

                mask = np.zeros_like(img)[:,:,0]
                # pdb.set_trace()
                mask[y1:y2, x1:x2] = 255

                # x1_crop = max(0, x1-20)
                # y1_crop = max(0, y1-20)
                # x2_crop = min(img.shape[1], x2+20)
                # y2_crop = min(img.shape[0], y2+20)
                # src_img_crop = img[y1_crop:y2_crop, x1_crop:x2_crop, :].copy()

                img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

                # dst_img_crop = img[y1_crop:y2_crop, x1_crop:x2_crop, :].copy()
                # res = np.concatenate([src_img_crop, dst_img_crop], 1)
                # cv2.imshow('res', res)
                # cv2.waitKey(0)
        save_sub_dir = '/'.join(imp.replace(src_dir, save_dir).split('/')[:-1])
        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)
        dst_imp = imp.replace(src_dir, save_dir)
        dst_jsp = dst_imp.replace('.jpg', '.json')
        cv2.imwrite(dst_imp, img)
        shutil.copy(jsp, dst_jsp)
        
                                
