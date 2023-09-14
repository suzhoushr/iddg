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
from utils.img_proc.image_process import shape_to_mask, random_crop, rnd_rotate_aug, paste_defect_on_lp
import numpy as np
import random
import copy

def calc_valid_area(im, defect_mask=None):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, binary_darker = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV) 
    _, binary_light = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) 
    binary = binary_darker
    for [col, row] in np.argwhere(binary_light > 0):
        binary[col, row] = 255
    binary = cv2.erode(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
    binary = cv2.dilate(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
    # pdb.set_trace()
    
    mask = binary / 255.0
    if defect_mask is not None:
        mask += defect_mask
    # pdb.set_trace()
    valid_area = list()
    if mask.sum() <= 0:
        x1, y1, x2, y2 = 0, 0, mask.shape[1], mask.shape[0]
        valid_area.append([x1, y1, x2, y2])
        return valid_area
    
    x_sum = mask.sum(0)
    x_valid = list()
    start = None
    end = None
    for x_pos in range(x_sum.shape[0]):
        if start is None:
            if x_sum[x_pos] == 0.0:
                start = x_pos
            continue
        if end is None:
            if x_sum[x_pos] > 0.0 or x_pos == x_sum.shape[0] - 1:
                end = x_pos - 1

                x_valid.append([start, end])
                start = None
                end = None
                continue

            if x_pos == x_sum.shape[0] - 1:
                end = x_pos

                x_valid.append([start, end])
                start = None
                end = None

    y_sum = mask.sum(1)
    y_valid = list()
    start = None
    end = None
    for y_pos in range(y_sum.shape[0]):
        if start is None:
            if y_sum[y_pos] == 0.0:
                start = y_pos
            continue
        if end is None:
            if y_sum[y_pos] > 0.0 or y_pos == y_sum.shape[0] - 1:
                end = y_pos - 1

                y_valid.append([start, end])
                start = None
                end = None
                continue

            if y_pos == y_sum.shape[0] - 1:
                end = y_pos

                y_valid.append([start, end])
                start = None
                end = None

    xyxy_valid = list()
    for x in x_valid:
        xyxy_valid.append([x[0], 0, x[1], mask.shape[0]-1])

    for y in y_valid:
        xyxy_valid.append([0, y[0], mask.shape[1]-1, y[1]])
    # pdb.set_trace()

    random.shuffle(xyxy_valid)
    return xyxy_valid

def rnd_sel_tgt(imgs_list, imp_src, try_iters=10):
    im_src_crop, info_src_crop = random_crop(imp_src)
    im_src_crop, info_src_crop, defect_mask = rnd_rotate_aug(im_src_crop, info_src_crop, return_mask=True)
    xyxy_valid = calc_valid_area(im_src_crop, defect_mask)
    if len(xyxy_valid) <= 0:
        return False, None, None      

    for _ in range(try_iters):
        imp_tgt = np.random.choice(imgs_list)
        jsp_tgt = imp_tgt.replace('.jpg', '.json')

        im_tgt = cv2.imread(imp_tgt)
        with open(jsp_tgt,'r',encoding ='utf-8') as jf:
            info_tgt = json.load(jf)

        ## rotate aug
        im_tgt, info_tgt = rnd_rotate_aug(im_tgt, info_tgt, return_mask=False)

        candi_masks = list()
        candi_shapes = list()
        candi_imgs = list()
        image_shape = [im_tgt.shape[0], im_tgt.shape[1], 3]
        for shape in info_tgt['shapes']:
            points = shape.get("points")
            shape_type = shape["shape_type"]
            try:
                label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
                label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

                label_mask_dilate = cv2.dilate(label_mask, np.ones((30, 30), dtype=np.uint8), iterations=1)
                pos = np.argwhere(label_mask_dilate > 0)  
                (crop_y1, crop_x1), (crop_y2, crop_x2) = pos.min(0), pos.max(0) + 1                
            except:
                continue
            label_mask = label_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            k_sz = np.random.randint(0, 5)
            if k_sz > 0:
                if label_mask.shape[0] > 16 and label_mask.shape[1] > 16 and shape["label"] not in ['huashang', 'guashang', 'liewen']:
                    label_mask = cv2.erode(label_mask, np.ones((k_sz, k_sz), dtype=np.uint8), iterations=1)
                    label_mask = cv2.dilate(label_mask, np.ones((k_sz, k_sz), dtype=np.uint8), iterations=1)
                else:
                    label_mask = cv2.dilate(label_mask, np.ones((k_sz, k_sz), dtype=np.uint8), iterations=1)
                    label_mask = cv2.erode(label_mask, np.ones((k_sz, k_sz), dtype=np.uint8), iterations=1)

            crop_mask = label_mask / 255.0
            if crop_mask.sum() < 16:
                continue
            try:
                where = np.argwhere(crop_mask > 0)
                (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
                if x2 - x1 > im_src_crop.shape[1] or y2 -y1 > im_src_crop.shape[0]:
                    continue
                if x2 - x1 > 200 or y2 -y1 > 200:
                    continue     
            except:
                continue       

            crop_img = im_tgt[crop_y1:crop_y2, crop_x1:crop_x2, :]
            crop_shape = copy.deepcopy(shape)
            crop_shape['points'] = [[float(p[0]-crop_x1), float(p[1]-crop_y1)] for p in points]

            candi_masks.append(crop_mask)
            candi_shapes.append(crop_shape)
            candi_imgs.append(crop_img)

        if len(candi_masks) <= 0:
            continue
        candi_indexs = list(range(len(candi_masks)))  
        random.shuffle(candi_indexs)
        for idx in candi_indexs:
            tgt_mask = candi_masks[idx]
            x1_t, y1_t, x2_t, y2_t = 0, 0, tgt_mask.shape[1], tgt_mask.shape[0]
            for xyxy in xyxy_valid:
                x1_s, y1_s, x2_s, y2_s = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                if x2_t - x1_t > x2_s - x1_s + 1 or y2_t -y1_t > y2_s - y1_s + 1:
                    continue
                gap_x = x2_s - x1_s + 1 - (x2_t - x1_t)
                gap_y = y2_s -y1_s + 1 - (y2_t - y1_t)
                
                try:
                    if gap_x == 0:
                        paste_x0 = x1_s
                    else:
                        paste_x0 = x1_s + np.random.randint(0, gap_x)
                    if gap_y == 0:
                        paste_y0 = y1_s
                    else:
                        paste_y0 = y1_s + np.random.randint(0, gap_y)

                    im_src_crop = paste_defect_on_lp(im_lp=im_src_crop,
                                                     im_defect=candi_imgs[idx],
                                                     mask_defect=tgt_mask,
                                                     p0=(paste_x0, paste_y0),
                                                     method='color_ada')   # 0_1_alpha | bright_ada | color_ada

                    shape = candi_shapes[idx]
                    shape['points'] = [[p[0]+paste_x0, p[1]+paste_y0] for p in shape['points']]
                    shape['group_id'] = 1000
                    info_src_crop['shapes'].append(shape)

                    return True, im_src_crop, info_src_crop

                except:
                    continue
        ## end for idx in candi_indexs
    ## end for _ in range(try_iters)

    return False, None, None  

def main(args, is_show=False):
    engine = Engine(args, img_sz=256)

    lp_dir = args['lp_data_dir']
    defect_dir = args['defect_data_dir']
    ng_syn_out_dir = args['output_dir']
    if not os.path.exists(ng_syn_out_dir):
        os.makedirs(ng_syn_out_dir)

    ng_imgs_list = list()
    for filepath, _, filenames in os.walk(defect_dir):
        for filename in filenames:
            if '.json' in filename:
                continue
            imp = os.path.join(filepath, filename)
            jsp = imp.replace('.jpg', '.json')
            if not os.path.exists(jsp):
                continue            
            ng_imgs_list.append(imp)

    src_imgs_list = list()
    for filepath, _, filenames in os.walk(lp_dir):
        for filename in filenames:
            if '.jpg' in filename:
                imp = os.path.join(filepath, filename)      
                src_imgs_list.append(imp)  

    ## ng syn
    num_success = 0
    while num_success < args['nums_need_syn']:
        src_imp = np.random.choice(src_imgs_list)   
        
        ## paste defect to lp area
        ret, im_paste, info_paste = rnd_sel_tgt(ng_imgs_list, src_imp, try_iters=10)  

        if not ret:
            continue

        ret, img, new_info = engine.synthesis_(im_paste, info_paste, 
                                                defect_need_gen=args['defect_need_gen'], 
                                                gid_need_gen=args['gid_need_gen'], 
                                                ratio=args['encoding_ratio'], 
                                                gd_w=args['gd_w'],
                                                SHOW=is_show)
        if not ret:
            continue
        
        name = src_imp.split('/')[-1][:-4] + '_' + str(num_success)            
        dst_imp = os.path.join(ng_syn_out_dir, name + '.jpg')
        dst_jsp = dst_imp.replace('.jpg', '.json')
        new_info["imagePath"] = name + '.jpg'
        cv2.imwrite(dst_imp, img)
        with open(dst_jsp, 'w', encoding='utf-8') as fi:   
            json.dump(new_info, fi, ensure_ascii=False, indent=4)
        num_success += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical characteristic size")
    ## model configuration
    parser.add_argument("--config_file", type=str, default='./config/infer/infer_inpainting_ldm_ch192.json', help="")
    parser.add_argument("--resume_path", type=str, default='', help="")
    parser.add_argument("--sample_type", type=str, default='ddim', choices=["ddpm", "ddim", "dpmsolver", "dpmsolver++"], help="")
    parser.add_argument("--sample_timesteps", type=int, default=100, help="")
    parser.add_argument("--gd_w", type=float, default=0.0, help="")
    parser.add_argument("--encoding_ratio", type=float, default=0.5, help="")
    parser.add_argument("--gid_need_gen", type=int, nargs='*', default=[1000], help="")
    parser.add_argument("--defect_need_gen", type=str, nargs='*',
                                                       default=['aotuhen', 
                                                                'daowen', 
                                                                'guashang', 
                                                                'heidian', 
                                                                'pengshang', 
                                                                'shahenyin', 
                                                                'tabian', 
                                                                'yashang', 
                                                                'yinglihen',
                                                                'yise',
                                                                'huashang',
                                                                'cashang'], 
                                                       help="")
    ## debug configuration
    parser.add_argument("--is_show", action='store_true', default=False, help="")
    ## dir configuration
    parser.add_argument("--lp_data_dir", type=str, default='/your/liangpin/data/dir/path/', help="")
    parser.add_argument("--defect_data_dir", type=str, default='/your/defect/data/dir/path/', help="")
    parser.add_argument("--output_dir", type=str, default='/your/output/data/dir/path/', help="")
    parser.add_argument("--nums_need_syn", type=int, default=500, help="")
    
    args = parser.parse_args()
    cfg = update_config(args)

    cfg['lp_data_dir'] = args.lp_data_dir
    cfg['defect_data_dir'] = args.defect_data_dir
    cfg['output_dir'] = args.output_dir
    cfg['nums_need_syn'] = args.nums_need_syn

    main(cfg, is_show=args.is_show)
