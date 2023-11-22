import sys
sys.path.append("./")
from tools.mengine import MEngine
from utils.img_proc.dhash import dHashBase
import json
import cv2
import pdb
import os
import sys
import argparse
from core.praser import update_config
from utils.img_proc.image_process import shape_to_mask
import numpy as np
import random
import copy

def paste_defect_to_lp(imp_q, jsp_q, imp_r, defect_need_gen=None, im_crop_sz=256):
    img_q = cv2.imread(imp_q)
    img_r = cv2.imread(imp_r)

    with open(jsp_q,'r',encoding ='utf-8') as jf:
        info_q = json.load(jf) 

    new_info = copy.deepcopy(info_q)
    new_info["shapes"] = list()

    image_shape = [img_q.shape[0], img_q.shape[1], 3]
    for shape in info_q['shapes']:
        label = shape.get("label")
        if defect_need_gen is not None and label not in defect_need_gen:
            continue
        points = shape.get("points")
        shape_type = shape["shape_type"]

        x_list = [k[0] for k in points]
        y_list = [k[1] for k in points]

        x1 = min(x_list)
        y1 = min(y_list)
        x2 = max(x_list)
        y2 = max(y_list)

        if (x2 - x1 + 1) > im_crop_sz or (y2 - y1 + 1) > im_crop_sz:
            continue

        try:
            label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
            label_inst = np.where(label_inst == True, 255, 0).astype('uint8')
            label_inst = cv2.dilate(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
            label_inst = cv2.erode(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
            label_inst = np.expand_dims(label_inst, -1)
            inst = label_inst / 255.0

        except:
            continue

        img_r = img_r * (1 - inst) + img_q * inst
        new_info["shapes"].append(shape)

    img_r = img_r.astype('uint8')

    return img_r, new_info

def main(args, is_show=False):
    engine = MEngine(args, img_sz=256)

    lp_dir = args['lp_data_dir']
    defect_dir = args['defect_data_dir']
    ng_syn_out_dir = args['output_dir']
    if not os.path.exists(ng_syn_out_dir):
        os.makedirs(ng_syn_out_dir)

    db_lp_dhash = dHashBase(data_dir=lp_dir)
    db_lp_dhash.buildDB()

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

    ## ng syn
    num_success = 0
    while num_success < args['nums_need_syn']:
        imp_q = np.random.choice(ng_imgs_list)  
        jsp_q = imp_q.replace('.jpg', '.json')
        if not os.path.exists(jsp_q):
            continue

        candi_imps = db_lp_dhash.search(imp=imp_q, top_k=1, thrs=12)
        if len(candi_imps) <= 0:
            print("Opps, there is not similarity image in LP images!")
            continue

        imp_r = candi_imps[0]
        
        ## paste defect to lp area
        im_paste, info_paste = paste_defect_to_lp(imp_q, jsp_q, imp_r, defect_need_gen=args['defect_need_gen'], im_crop_sz=200)  

        if len(info_paste["shapes"]) <= 0:
            continue

        img, new_info = engine.defect_selfaug_(im_paste, info_paste,
                                                 aug_type_prob={'re-generate':1.0,'defect2lp':0.0,'xy-shift':0.0,'flip':0.0},
                                                 ratio=1.0,
                                                 gd_w=0.0, 
                                                 SHOW=is_show,
                                                 mask_type='rect',
                                                 task='synthesis',
                                                 use_rectangle_labelme=False)
        
        name = imp_r.split('/')[-1][:-4] + '_' + str(num_success)            
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
                                                       default=['aotuhen', 'daowen', 'guashang', 
                                                                'heidian', 'pengshang', 'shahenyin', 'tabian', 
                                                                'yashang', 'yinglihen', 'yise', 'huashang',
                                                                'cashang', 'liangdian', 'zangwu', 'wudaojiao',
                                                                'liewen', 'qikong', 'zazhi', 'jiagongbuliang',
                                                                'duoliao', 'qipi', 'lvxie', 'queliao', 'lengge'], 
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
