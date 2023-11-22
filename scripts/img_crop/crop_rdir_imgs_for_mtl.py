import json
import cv2
import glob
from tqdm import tqdm
import os
import copy
import numpy as np
from PIL import Image, ImageDraw
import math
from copy import deepcopy
import random
import pdb

def mask2polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    results = [item.squeeze() for item in contours]
    return results

def shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]

    ## check
    if shape_type in ["circle", "rectangle", "line"]:
        if len(xy) == 1:
            shape_type = "point"
        if len(xy) > 2:
            shape_type = "polygon"
    elif shape_type == "point":
        if len(xy) == 2:
            shape_type = "line"
        if len(xy) > 2:
            shape_type = "polygon"
    elif shape_type == "polygon":
        if len(xy) == 1:
            shape_type = "point"
        if len(xy) == 2:
            shape_type = "line"
    else:
        if len(xy) == 1:
            shape_type = "point"

    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1, width=line_width)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        (x1, y1), (x2, y2) = xy
        x11, y11, x22, y22 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        xy = (x11, y11), (x22, y22)
        draw.rectangle(xy, outline=1, fill=1, width=line_width)
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
        draw.polygon(xy=xy, outline=1, fill=1, width=line_width)
    mask = np.array(mask, dtype=bool)

    return mask

def calc_split_area(len_src, win, stride):
    area = list()
    if win >= len_src:
        p_b = 0
        p_e = len_src - 1

        area.append((p_b, p_e+1))
        return area
    
    p_b = 0 
    p_e = p_b + win - 1   
    area.append((p_b, p_e+1)) 
    while True:        
        p_b += stride
        p_e = p_b + win - 1

        if p_e <= len_src - 1:
            area.append((p_b, p_e+1))
        else:
            p_b = len_src - 1
            if p_b - p_e + 1 >= stride:
                area.append((p_b, p_e+1))
                break
            else:
                p_e = len_src - 1
                p_b = p_e - win + 1
                area.append((p_b, p_e+1))
                break

    return area

def corp_box_by_json(imp, jsp, 
                     im_crop_sz=2048, 
                     im_overlap=224,
                     d_defects=None):
    ## read image
    img = cv2.imread(imp, 1)
    img_w, img_h = img.shape[1], img.shape[0]

    ## prepare mask
    with open(jsp,'r',encoding ='utf-8') as jf:
        info = json.load(jf)
    masks_l = list()
    shape_l = list()
    image_shape = [img_h, img_w, 3]
    for _shape in info['shapes']:
        points = _shape.get("points")
        shape_type = _shape["shape_type"]       

        label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=8, point_size=8)
        label_mask = np.where(label_mask == True, 1, 0).astype('uint8')

        if "flags" in _shape:
            del _shape['flags']

        masks_l.append(label_mask)
        shape_l.append(_shape)


    ## crop img
    defect_img_list =list()
    defect_json_list = list()
    lp_img_list =list()
    lp_json_list = list()

    ### calc len_w, len_h
    areas_w = calc_split_area(len_src=img_w, win=im_crop_sz, stride=im_overlap)
    areas_h = calc_split_area(len_src=img_h, win=im_crop_sz, stride=im_overlap)
    
    index = 0
    for area_w in areas_w:
        for area_h in areas_h:
            x1, y1, x2, y2 = area_w[0], area_h[0], area_w[1], area_h[1]
            im_crop = img[y1:y2, x1:x2, :]

            mask = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
            mask[y1:y2, x1:x2] = 1

            new_json = copy.deepcopy(info)
            new_json["shapes"] = []
            new_json["imagePath"] = imp.split('/')[-1][:-4].replace(' ', '') + '_' + str(index) + '.jpg'
            new_json["imageData"] = None
            new_json["imageHeight"] = im_crop.shape[0]
            new_json["imageWidth"] = im_crop.shape[1]
            new_json["imageDepth"] = 3

            have_unk = False

            for i in range(len(masks_l)):
                th = mask * masks_l[i]
                iou_ration = 1.0 * (th.sum()+1e-20) / (masks_l[i].sum()+1e-20)
                if iou_ration < 0.33333:
                    continue

                shape = copy.deepcopy(shape_l[i])
                points = shape_l[i]['points']
                shape['points'] = list()
                if iou_ration == 1.0:     
                    label = shape["label"]
                    if d_defects is not None and label not in d_defects:
                        have_unk = True
                        break
                    shape["label"] = d_defects[label]

                    x_list = [k[0] for k in points]
                    y_list = [k[1] for k in points]
                    for k in range(len(x_list)):
                        x = x_list[k] - x1
                        y = y_list[k] - y1
                        shape['points'].append([x, y])
                    new_json["shapes"].append(shape)

                if iou_ration < 1.0 and iou_ration >= 0.333333:
                    label = shape["label"]
                    if d_defects is not None and label not in d_defects:
                        have_unk = True
                        break
                    shape["label"] = d_defects[label]

                    th = 255 * th
                    polygons = mask2polygon(th)
                    if len(polygons) <= 0:
                        continue
                    if len(polygons[0]) <= 2:
                        continue
                    polygons = [[float(v[0]), float(v[1])] for v in polygons[0]]
                    shape["shape_type"] = "polygon"

                    x_list = [k[0] for k in polygons]
                    y_list = [k[1] for k in polygons]
                    for k in range(len(x_list)):
                        x = min(im_crop.shape[1], max(0, x_list[k] - x1))
                        y = min(im_crop.shape[0], max(0, y_list[k] - y1))
                        shape['points'].append([x, y])
                    new_json["shapes"].append(shape)

            if have_unk:
                continue

            if len(new_json["shapes"]) <= 0:
                new_json["imageScale"] = 1.0
                lp_img_list.append(im_crop)
                lp_json_list.append(new_json)
            else:
                defect_img_list.append(im_crop)
                defect_json_list.append(new_json)
            index += 1
    
    return defect_img_list, defect_json_list, lp_img_list, lp_json_list

if __name__ == "__main__":
    d_defects = {"aohen":"aotuhen", "aohen-kuai":"aotuhen",
                "aokeng":"aokeng", "aotu":"aotuhen", "aotudian":"aotudian",
                "aotuhen":"aotuhen", "aotuhen1":"aotuhen", "aotuhen2":"aotuhen",
                "baidian":"baidian", "baisezaodian":"baidian", "baomo":"baomo",
                "bianxing":"bianxing", "biyin":"biyin", "canjiao":"canjiao",
                "cashang":"cashang", "damohen":"damohen", "damoyin":"damohen",
                "daowen":"daowen", "daowen-dantiao":"daowen", "daowen-jiao":"daowen",
                "daowen-mian":"daowen", "daowenxian":"daowen", "diaoqi":"diaoqi",
                "duoliao":"duoliao", "fabai":"fabai", "fushi":"fushi",
                "guashang":"guashang", "guashang-baiduan":"guashang", "guashang-baizhang":"guashang",
                "guashang-heibai":"guashang", "guashang-heiduan":"guashang", "guashang-heizhang":"guashang",
                "gubao":"gubao", "gubao3":"gubao", "guoqie":"guoqie",
                "guoxi":"guoxi", "heidian":"heidian", "huanxingdaowen":"daowen",
                "huashang":"huashang", "juchi":"juchi", "keli":"keli",
                "liangdian":"liangdian", "liangyin":"liangyin", "liaowen":"liaowen",
                "liewen":"liewen", "loutong":"loutong", "lvjixian":"lvjixian",
                "madian":"madian", "maobian":"maoci", "maoci":"maoci",
                "maoxu":"maoxu", "molie":"molie", "neiqiaopian":"neiqiaopian",
                "pengshang":"pengshang", "pengshang-bian":"pengshang", "pengshang-jiao":"pengshang",
                "pengshang-mian":"pengshang", "pengshang-zhang":"pengshang", "penshabujun":"penshabujun",
                "penshabujun-bai":"penshabujun", "penshabujun-hei":"penshabujun", "queliao":"queliao",
                "shahenyin":"shahenyin", "shangxian":"shangxian", "shayan":"shayan",
                "shuahen":"shuahen", "shuiyin":"shuiyin", "tabian":"tabian",
                "tabian-an":"tabian", "tabian-liang":"tabian",
                "tubao":"tubao", "tubao-kuai":"tubao", "waiqiaopian":"waiqiaopian",
                "xianhen":"xianhen", "xianweimao":"maoxu", "yahen":"aotuhen",
                "yashang":"yashang", "yinglihen":"yinglihen", "yise":"yise",
                "yise1":"yise", "yise2":"yise", "yise-bai":"yise",
                "yise-baitiao":"yise", "yise-hei":"yise", "yise-heitiao":"yise",
                "yise-hui":"yise", "yise-liang":"yise", "yisedian":"yise",
                "yiwu":"yiwu", "youmo":"youmo", "zangwu":"zangwu",
                "zhendaowen":"daowen", "zhipo":"zhipo"}

    im_crop_sz = 768
    im_overlap = 512

    data_root = '/home/data0/project-datasets/new_database/'
    fi_in_name = 'Stator'
    fo_in_name = fi_in_name + '_crop' + str(im_crop_sz) + '_overlap' + str(im_overlap) 

    SAVE_LP = True

    in_dir = os.path.join(data_root, fi_in_name)
    out_dir = os.path.join(data_root, fo_in_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_list = list()
    for filepath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if '.jpg' in filename:
                imp = os.path.join(filepath, filename)
                img_list.append(imp)
    # pdb.set_trace()
    
    index = 0
    for imp in tqdm(img_list):
        jsp = imp.replace('.jpg', '.json')
        if not os.path.exists(jsp):
            continue

        defect_save_dir = "/".join(imp.replace('/'+fi_in_name+'/', '/'+fo_in_name+'/').split('/')[:-1])
        defect_save_list = imp.replace('/'+fi_in_name+'/', '/'+fo_in_name+'/').split('/')[:-1]
        defect_save_list[-1] = 'LP_' + defect_save_list[-1]        
        lp_save_dir = "/".join(defect_save_list)
        defect_save_dir = lp_save_dir.replace('LP_', 'NG_')
        prefix_im_name = imp.split('/')[-1][:-4]
        if not os.path.exists(defect_save_dir):
            os.makedirs(defect_save_dir)
        if SAVE_LP and not os.path.exists(lp_save_dir):
            os.makedirs(lp_save_dir)

        defect_imgs, defect_jsons, lp_imgs, lp_jsons = corp_box_by_json(imp, jsp, im_crop_sz=im_crop_sz, im_overlap=im_overlap, d_defects=d_defects)
        # pdb.set_trace()

        for i in range(len(defect_imgs)):
            img_crop = defect_imgs[i]
            json_crop = defect_jsons[i]

            nimp = os.path.join(defect_save_dir, json_crop["imagePath"])
            njsonp = nimp.replace('.jpg', '.json')
            if os.path.exists(nimp) and os.path.exists(njsonp):
                continue

            cv2.imwrite(nimp, img_crop)
            with open(njsonp, 'w', encoding='utf-8') as new_jf:   
                json.dump(json_crop, new_jf, ensure_ascii=False, indent=4)

        if SAVE_LP:
            for i in range(len(lp_imgs)):
                img_crop = lp_imgs[i]
                json_crop = lp_jsons[i]

                nimp = os.path.join(lp_save_dir, json_crop["imagePath"])
                njsonp = nimp.replace('.jpg', '.json')
                if os.path.exists(nimp) and os.path.exists(njsonp):
                    continue

                cv2.imwrite(nimp, img_crop)
                with open(njsonp, 'w', encoding='utf-8') as new_jf:   
                    json.dump(json_crop, new_jf, ensure_ascii=False, indent=4)



            







    