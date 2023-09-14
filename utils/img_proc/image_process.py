import json
import os
import random
import copy
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
from copy import deepcopy
import math
import pdb

def paste_defect_on_lp(im_lp, im_defect, mask_defect, p0, method='0_1_alpha'):
    '''
    mehtod: 0_1_alpha | bright_ada | poisson | thin_color
    '''
    im_res = im_lp.copy()
    crop_im_defect = im_defect.copy()
    paste_x0, paste_y0 = p0[0], p0[1]
    paste_x1, paste_y1 = paste_x0 + crop_im_defect.shape[1], paste_y0 + crop_im_defect.shape[0]
    paste_img = im_res[paste_y0:paste_y1, paste_x0:paste_x1, :].copy()   

    if method == '0_1_alpha':                      
        m_alpha = mask_defect.copy()
        m_alpha = cv2.GaussianBlur(m_alpha, (5, 5), cv2.BORDER_DEFAULT)
        m_alpha = m_alpha[:, :, None]
        paste_img = (1 - m_alpha) * paste_img + m_alpha * crop_im_defect
        im_res[paste_y0:paste_y1, paste_x0:paste_x1, :] = paste_img
    
    if method == 'bright_ada':
        m_alpha = mask_defect.copy()
        m_alpha = cv2.GaussianBlur(m_alpha, (5, 5), cv2.BORDER_DEFAULT)
        m_alpha = m_alpha[:, :, None]
        defect_img = uniform_brightness(paste_img, crop_im_defect, mask=mask_defect, sharpness=1.0, contrast=1.0)
        paste_img = (1 - m_alpha) * paste_img + m_alpha * defect_img
        im_res[paste_y0:paste_y1, paste_x0:paste_x1, :] = paste_img

    if method == 'color_ada':
        m_alpha = mask_defect.copy()
        m_alpha = cv2.GaussianBlur(m_alpha, (5, 5), cv2.BORDER_DEFAULT)
        m_alpha = m_alpha[:, :, None]
        defect_img = color_transfer(paste_img, crop_im_defect)
        paste_img = (1 - m_alpha) * paste_img + m_alpha * defect_img
        im_res[paste_y0:paste_y1, paste_x0:paste_x1, :] = paste_img

    if method == 'poisson':
        mask = (mask_defect * 255).astype('uint8')
        mask = cv2.dilate(mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
        (y, x) = np.where(mask > 0)
        center = (int(y.mean()), int(x.mean()))
        # pdb.set_trace()
        try:
            im_blend = cv2.seamlessClone(crop_im_defect, paste_img, mask, center, cv2.NORMAL_CLONE)  # MONOCHROME_TRANSFER | NORMAL_CLONE     
        except:
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
            pdb.set_trace()
        m_alpha = mask_defect.copy()  
        m_alpha = cv2.GaussianBlur(m_alpha, (5, 5), cv2.BORDER_DEFAULT)           
        m_alpha = m_alpha[:, :, None]
        dst = (1 - m_alpha) * paste_img + m_alpha * im_blend
        im_res[paste_y0:paste_y1, paste_x0:paste_x1, :] = dst

    return im_res

def color_transfer(source, target):
    # 将源图像和目标图像从BGR颜色空间转到Lab颜色通道
    # 确保使用OpenCV图像为32位浮点类型数据
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # 计算源图像和目标图像的颜色统计信息(每个通道的均值和标准差)
    # L通道均值、标准差，a通道均值、标准差，b通道均值、标准差
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    # 从目标图像中减去均值
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    # 按标准差缩放(scale_rate = 目标图像标准差/源图像标准差)
    l = max(0.5, min(1.5, (lStdTar / (lStdSrc + 1e-20)))) * l
    a = max(0.5, min(1.5, (aStdTar / (aStdSrc + 1e-20)))) * a
    b = max(0.5, min(1.5, (bStdTar / (bStdSrc + 1e-20)))) * b
    # 加入源图像对应通道的均值
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # 如果像素强度超出范围，则将像素强度剪裁为[0, 255]范围
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    # 将通道合并在一起并转换回BGR颜色空间，确保确保使用 8 位无符号整数数据类型
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # 返回颜色迁移后的图像
    return transfer

def image_stats(image):
    # 计算每个通道的均值和标准差
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # 返回颜色统计信息
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def uniform_brightness(lp_img, defect_img, mask=None, sharpness=None, contrast=None):
    if lp_img.ndim == 3:  # 如果是三通道图片
        gray_lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2GRAY)
    else:  # 如果是单通道图片
        gray_lp_img = lp_img

    if defect_img.ndim == 3:  # 如果是三通道图片
        gray_defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    else:  # 如果是单通道图片
        gray_defect_img = defect_img

    mean_lp_img = cv2.mean(gray_lp_img)[0]
    mean_defect_img= 0 #cv2.mean(gray_defect_img)[0]
    if mask is not None and mask.sum() > 0.0:
        count = 0
        for [col, row] in np.argwhere(mask > 0):
            mean_defect_img += 0.0 + gray_defect_img[col, row]
            count += 1
        mean_defect_img = 1.0 / count * mean_defect_img
    else:
        mean_defect_img= cv2.mean(gray_defect_img)[0]

    alpha = mean_lp_img / (mean_defect_img + 1e-5)
    # print("alpha is ======> " + str(alpha))
    alpha = min(1.2, alpha)
    alpha = max(0.8, alpha)
    # mean_defect_img = cv2.convertScaleAbs(gray_defect_img, alpha=alpha, beta=0)
    # mean_defect_img = cv2.cvtColor(mean_defect_img, cv2.COLOR_GRAY2BGR)

    defect_img_pil = Image.fromarray(cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB))
    bright_enhancer = ImageEnhance.Brightness(defect_img_pil)
    defect_img_pil_bright = bright_enhancer.enhance(alpha)
    # bright_img = cv2.cvtColor(np.asarray(defect_img_pil_bright), cv2.COLOR_RGB2BGR)

    # contrast_enhancer = ImageEnhance.Contrast(defect_img_pil_bright)
    # defect_img_pil_bright = contrast_enhancer.enhance(alpha)

    if sharpness is not None and sharpness != 1.0:
        sharp_enhancer = ImageEnhance.Sharpness(defect_img_pil_bright)
        defect_img_pil_bright = sharp_enhancer.enhance(sharpness)

    if contrast is not None and contrast != 1.0: 
        contrast_enhancer = ImageEnhance.Contrast(defect_img_pil_bright)
        defect_img_pil_bright = contrast_enhancer.enhance(contrast)

    defect_img_pil_bright = cv2.cvtColor(np.asarray(defect_img_pil_bright), cv2.COLOR_RGB2BGR)

    # img_res = np.concatenate([lp_img, defect_img, bright_img, defect_img_pil_bright], 1)
    # cv2.imshow('res', img_res)
    # cv2.waitKey(0)

    return defect_img_pil_bright

def mask2polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    results = [item.squeeze() for item in contours]
    return results

def random_crop(imp, crop_shape=(2048, 2048)):
    '''
    crop_shape: (w, h)
    '''
    image = cv2.imread(imp)
    img_h = image.shape[0]
    img_w = image.shape[1]
    
    x0, y0 = 0, 0
    delta_w, delta_h = crop_shape[0], crop_shape[1]

    if img_w <= crop_shape[0]:
        delta_w = img_w
    else:
        x0 = random.randint(0, img_w - crop_shape[0])

    if img_h <= crop_shape[1]:
        delta_h = img_h
    else:
        y0 = random.randint(0, img_h - crop_shape[1])

    img_crop = image[y0:y0+delta_h, x0:x0+delta_w, :]
    mask = np.zeros((image.shape[0], image.shape[1])).astype('uint8')
    mask[y0:y0+delta_h, x0:x0+delta_w] = 1

    info_src = dict()
    info_src["imageData"] = None
    info_src["imageHeight"] = img_crop.shape[0]
    info_src["imageWidth"] = img_crop.shape[1]
    info_src["imagePath"] = ''
    info_src["shapes"] = list()
    info_src["version"] = "1.0"   

    jsp = imp.replace('.jpg', '.json')
    if os.path.exists(jsp):
        with open(jsp, 'r', encoding='utf-8') as jf:
            info = json.load(jf)
        
        image_shape = [image.shape[0], image.shape[1], 3]
        for _shape in info['shapes']:
            label = _shape.get("label")
            points = _shape.get("points")
            shape_type = _shape["shape_type"]
            
            if _shape["shape_type"] == 'polygon':
                if len(points) == 2:
                    shape_type = 'linestrip'
                if len(points) == 1:
                    shape_type = 'point'  
            if _shape["shape_type"] == 'rectangle':   
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])
                points = [[x1, y1], [x2, y2]]

            label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
            label_mask = np.where(label_mask == True, 1, 0).astype('uint8')

            th = mask * label_mask
            iou_ration = 1.0 * (th.sum()+1e-20) / (label_mask.sum()+1e-20)

            if iou_ration >= 1.0:
                shape = deepcopy(_shape)
                shape["shape_type"] = shape_type
                shape['label'] = label
                shape['points'] = list()                    
                x_list = [k[0] for k in points]
                y_list = [k[1] for k in points]
                for k in range(len(x_list)):
                    x = min(img_crop.shape[1] - 1, max(0, x_list[k] - x0))
                    y = min(img_crop.shape[0] - 1, max(0, y_list[k] - y0))
                    shape['points'].append([x, y])
                info_src["shapes"].append(shape)

            if iou_ration < 1.0 and iou_ration > 0.333:
                th = 255 * th
                polygons = mask2polygon(th)
                if len(polygons) <= 0:
                    continue
                if len(polygons[0]) <= 2:
                    continue
                polygons = [[float(v[0]), float(v[1])] for v in polygons[0]]

                shape = deepcopy(_shape)
                shape["shape_type"] = 'polygon'
                shape['label'] = label
                shape['points'] = list()
                x_list = [k[0] for k in polygons]
                y_list = [k[1] for k in polygons]
                for k in range(len(x_list)):
                    x = min(img_crop.shape[1] - 1, max(0, x_list[k] - x0))
                    y = min(img_crop.shape[0] - 1, max(0, y_list[k] - y0))
                    shape['points'].append([x, y])
                info_src["shapes"].append(shape)

    return img_crop, info_src

def shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    if shape_type == "rectangle":
        x1 = min(points[0][0], points[1][0])
        y1 = min(points[0][1], points[1][1])
        x2 = max(points[0][0], points[1][0])
        y2 = max(points[0][1], points[1][1])
        points = [[x1, y1], [x2, y2]]
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
        x1 = min(points[0][0], points[1][0])
        y1 = min(points[0][1], points[1][1])
        x2 = max(points[0][0], points[1][0])
        y2 = max(points[0][1], points[1][1])
        points = [[x1, y1], [x2, y2]]
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

def rnd_rotate_aug(img, info, return_mask=True):
    im_src = img.copy()
    info_src = copy.deepcopy(info)

    im_template_h, im_template_w = im_src.shape[0], im_src.shape[1]
    # up-down    
    if np.random.rand() > 0.5:
        im_src = cv2.flip(im_src, 0)
        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[p[0], im_template_h - 1 - p[1]] for p in points]
            info_src['shapes'][i]['points'] = points

    # left-right
    if np.random.rand() > 0.5:
        im_src = cv2.flip(im_src, 1)
        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[im_template_w - 1 - p[0], p[1]] for p in points]
            info_src['shapes'][i]['points'] = points

    # left-right and up-down
    if np.random.rand() > 0.5:
        im_src = cv2.flip(im_src, -1)
        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[im_template_w - 1 - p[0], im_template_h - 1 - p[1]] for p in points]
            info_src['shapes'][i]['points'] = points

    # clockwise 90
    if np.random.rand() > 0.5:
        im_src = cv2.rotate(im_src, cv2.ROTATE_90_CLOCKWISE)        

        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[im_template_h - 1 - p[1], p[0]] for p in points]
            info_src['shapes'][i]['points'] = points  

        im_template_h, im_template_w = im_src.shape[0], im_src.shape[1]          

    # clockwise 180
    if np.random.rand() > 0.5:
        im_src = cv2.rotate(im_src, cv2.ROTATE_180)

        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[im_template_w - 1 - p[0], im_template_h - 1 - p[1]] for p in points]
            info_src['shapes'][i]['points'] = points

    # clockwise 270
    if np.random.rand() > 0.5:
        im_src = cv2.rotate(im_src, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for i in range(len(info_src['shapes'])):
            points = info_src['shapes'][i]['points']
            points = [[p[1], im_template_w - 1 - p[0]] for p in points]
            info_src['shapes'][i]['points'] = points

    if not return_mask:
        return im_src, info_src
    
    mask_src = np.zeros((im_src.shape[0], im_src.shape[1])).astype('uint8')
    image_shape = [im_src.shape[0], im_src.shape[1], 3]
    for shape in info_src['shapes']:
        points = shape['points']
        shape_type = shape["shape_type"]

        label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=8, point_size=4)
        label_mask = np.where(label_mask == True, 1, 0).astype('uint8')
        mask_src += label_mask

    return im_src, info_src, mask_src

def crop_image(img, points, im_crop_sz):
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

def points2mask(points, shape_type, label='huashang', mask_type='rect', MIN_PIX=32, im_crop_sz=256):
    '''
    return output mask has the shape: 1 x H x W 
    '''
    image_shape = [im_crop_sz, im_crop_sz, 3]    
    MIN_PIX = 32
    label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=16, point_size=16)
    label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

    pos = np.argwhere(label_mask > 0)
    (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1

    h = y2 - y1
    w = x2 - x1
    if h < MIN_PIX or w < MIN_PIX:
        delta_x, delta_y = 1, 1
        if h < MIN_PIX:
            delta_y = MIN_PIX - h
        if w < MIN_PIX:
            delta_x = MIN_PIX - w
        label_mask = cv2.dilate(label_mask, np.ones((delta_y, delta_x), dtype=np.uint8), iterations=1)

    if label_mask.sum() <= 0:
        return np.zeros((1, im_crop_sz, im_crop_sz))
    
    mask = np.zeros((im_crop_sz, im_crop_sz)).astype('uint8')
    if label not in ['huashang', 'liewen', 'guashang'] and mask_type != 'poly':
        pos = np.argwhere(label_mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1            
        mask[y1:y2, x1:x2] = 255
    else:
        mask = label_mask 
    mask = cv2.GaussianBlur(mask, (5, 5), 0, 0)
    mask = np.expand_dims(mask, 0) / 255.0

    return mask
    

