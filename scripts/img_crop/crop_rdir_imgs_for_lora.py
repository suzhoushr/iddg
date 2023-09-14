import json
import cv2
from tqdm import tqdm
import os
import copy
import numpy as np
import pdb

def corp_box_by_json(imp, jsp, im_crop_sz=256):
    img = cv2.imread(imp, 1)

    img_list =list()
    json_list = list()

    with open(jsp,'r',encoding ='utf-8') as jf:
        info = json.load(jf)

    index = 0
    for _shape in info['shapes']:
        label = _shape.get("label")
        points = _shape.get("points")

        x_list = [k[0] for k in points]
        y_list = [k[1] for k in points]

        x1 = min(x_list)
        y1 = min(y_list)
        x2 = max(x_list)
        y2 = max(y_list)

        if (x2 - x1 + 1) > im_crop_sz and (y2 - y1 + 1) > im_crop_sz:
            continue
        if label not in ['huashang', 'liewen', 'guashang']:
            if (x2 - x1 + 1) > 200 and (y2 - y1 + 1) > 200:
                continue

        # if 1.0 * (x2 - x1 + 1) * (y2 - y1 + 1) / im_crop_sz / im_crop_sz > 0.8:
        #     continue

        cx = (x2 - x1 + 1) / 2.0 + x1
        cy = (y2 - y1 + 1) / 2.0 + y1 
        boxx_x1 = max(0, int(cx - im_crop_sz / 2.0))
        boxx_y1 = max(0, int(cy - im_crop_sz / 2.0))
        boxx_x2 = min(img.shape[1]-1, boxx_x1 + im_crop_sz - 1)
        boxx_y2 = min(img.shape[0]-1, boxx_y1 + im_crop_sz - 1)

        try:
            crop_img = img[boxx_y1:boxx_y2+1, boxx_x1:boxx_x2+1, :]
            if crop_img.shape[0] < im_crop_sz:
                crop_img = cv2.copyMakeBorder(crop_img, 0, im_crop_sz - crop_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if crop_img.shape[1] < im_crop_sz:
                crop_img = cv2.copyMakeBorder(crop_img, 0, 0, 0, im_crop_sz - crop_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        except:
            continue

        new_pionts = list()
        for i in range(len(x_list)):
            x = min(im_crop_sz - 1, max(0, x_list[i] - boxx_x1))
            y = min(im_crop_sz - 1, max(0, y_list[i] - boxx_y1))
            new_pionts.append([x,y])
        _shape["points"] = new_pionts
        
        crop_json = copy.deepcopy(info)
        crop_json["shapes"] = [_shape]
        crop_json["imagePath"] = label + '_' + str(index) + '_' + imp.split('/')[-1].replace(' ', '')
        crop_json["imageData"] = None
        crop_json["imageHeight"] = im_crop_sz
        crop_json["imageWidth"] = im_crop_sz
        crop_json["imageDepth"] = 3
        crop_json["srcImagePath"] = imp
        crop_json["srcX1"] = boxx_x1
        crop_json["srcX2"] = boxx_x2
        crop_json["srcY1"] = boxx_y1
        crop_json["srcY2"] = boxx_y2

        index += 1
        img_list.append(crop_img)
        json_list.append(crop_json)

    return img_list, json_list

if __name__ == "__main__":
    im_crop_sz = 256
    
    data_root = '/home/data0/project-datasets/dehongcheng/exp_SYB/train_val_data/'
    fi_in_name = 'train'
    fo_in_name = fi_in_name + '_crop4lora_sz' + str(im_crop_sz)

    in_dir = os.path.join(data_root, fi_in_name)
    out_dir = os.path.join(data_root, fo_in_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgs_list = list()
    for filepath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if '.jpg' in filename:
                imp = os.path.join(filepath, filename)
                jsp = imp.replace('.jpg', '.json')
                if not os.path.exists(jsp):
                    continue
                imgs_list.append(imp)
    
    for imp in tqdm(imgs_list):
        jsp = imp.replace('.jpg', '.json')
        imgs, jsons = corp_box_by_json(imp, jsp, im_crop_sz=im_crop_sz)

        save_dir = "/".join(imp.replace('/'+fi_in_name+'/', '/'+fo_in_name+'/').split('/')[:-1])
        prefix_im_name = imp.split('/')[-1][:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(imgs)):
            img_crop = imgs[i]
            json_crop = jsons[i]

            nimp = os.path.join(save_dir, json_crop["imagePath"])
            njsonp = nimp.replace('.jpg', '.json')

            if os.path.exists(nimp) and os.path.exists(njsonp):
                continue

            cv2.imwrite(nimp, img_crop)
            with open(njsonp, 'w', encoding='utf-8') as new_jf:   
                json.dump(json_crop, new_jf, ensure_ascii=False, indent=4)
        # pdb.set_trace()


            







    
