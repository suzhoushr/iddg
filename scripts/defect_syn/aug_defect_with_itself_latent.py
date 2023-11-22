import sys
sys.path.append("./")
from tools.engine import Engine
import json
import cv2
import os
import sys
from tqdm import tqdm
import argparse
from core.praser import update_config
from utils.img_proc.image_process import shape_to_mask
import numpy as np
import pdb

def crop_imgs_by_info(imp, 
                      jsp,
                      crop_size=256):
    img = cv2.imread(imp)
    with open(jsp,'r',encoding ='utf-8') as jf:
        info = json.load(jf)

    shape_masks = list()
    image_shape = [img.shape[0], img.shape[1], 3]
    for shape in info['shapes']:
        points = shape['points']
        shape_type = shape["shape_type"]

        label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=8, point_size=4)
        label_mask = np.where(label_mask == True, 1, 0).astype('uint8')
        shape_masks.append(label_mask)
    

def main(args, is_show=False):
    engine = Engine(args, img_sz=256)

    src_dir = args['src_data_dir']    

    src_imgs_list = list()
    for filepath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if '.json' in filename:
                continue
            imp = os.path.join(filepath, filename)
            jsp = imp.replace('.jpg', '.json')               
            if not os.path.exists(jsp):
                continue 
            src_imgs_list.append(imp)
    src_imgs_list_ex = list()
    for _ in range(args['iters']):
        src_imgs_list_ex.extend(src_imgs_list)

    out_dir = args['output_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## ng syn
    index = 0
    for imp in tqdm(src_imgs_list_ex):    
        jsp = imp.replace('.jpg', '.json')  
        random_ratio = args['encoding_ratio']  
        if args['jit_ratio'] and args['encoding_ratio'] > 0.15:
            random_ratio = 0.001 * np.random.randint(150, int(args['encoding_ratio']*1000)+1)
        ret, img, new_info = engine.defect_selfaug(imp, jsp, 
                                                   defect_need_gen=args['defect_need_gen'], 
                                                   aug_ops=['flip', 'xy-shift'],
                                                   ratio=random_ratio, 
                                                   gd_w=args['gd_w'],
                                                   prob_syn=args['prob_syn'],
                                                   prob_defect2lp=0.5,
                                                   mask_type='rect', # rect | poly
                                                   SHOW=is_show)
        
        if not ret:
            continue
        name = imp.split('/')[-1][:-4]
        name = name + '_' + str(index)            
        dst_imp = os.path.join(out_dir, name + '.jpg')
        dst_jsp = dst_imp.replace('.jpg', '.json')
        new_info["imagePath"] = name + '.jpg'
        cv2.imwrite(dst_imp, img)
        with open(dst_jsp, 'w', encoding='utf-8') as fi:   
            json.dump(new_info, fi, ensure_ascii=False, indent=4)

        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical characteristic size")
    ## model configuration
    parser.add_argument("--config_file", type=str, default='./config/infer/infer_v1.2.0.json', help="")
    parser.add_argument("--resume_path", type=str, default='', help="")
    parser.add_argument("--sample_type", type=str, default='ddim', choices=["ddpm", "ddim", "dpmsolver", "dpmsolver++"], help="")
    parser.add_argument("--sample_timesteps", type=int, default=100, help="")
    parser.add_argument("--gd_w", type=float, default=0.0, help="")
    parser.add_argument("--encoding_ratio", type=float, default=0.5, help="")
    parser.add_argument("--defect_need_gen", type=str, nargs='*',
                                                       default=['aotuhen', 'daowen', 'guashang', 
                                                                'heidian', 'pengshang', 'shahenyin', 'tabian', 
                                                                'yashang', 'yinglihen', 'yise', 'huashang',
                                                                'cashang', 'liangdian', 'zangwu', 'wudaojiao',
                                                                'liewen', 'qikong', 'zazhi', 'jiagongbuliang',
                                                                'duoliao', 'qipi', 'lvxie', 'queliao', 'lengge'], 
                                                       help="")
    parser.add_argument("--jit_ratio", action='store_true', default=False, help="")
    ## debug configuration
    parser.add_argument("--is_show", action='store_true', default=False, help="")    
    parser.add_argument("--prob_syn", type=float, default=1.0, help="")
    ## dir configuration
    parser.add_argument("--src_data_dir", type=str, default='/your/liangpin/data/dir/path/', help="")
    parser.add_argument("--output_dir", type=str, default='/your/output/data/dir/path/', help="")
    parser.add_argument("--iters", type=int, default=1, help="")
    
    args = parser.parse_args()
    cfg = update_config(args)

    cfg['src_data_dir'] = args.src_data_dir
    cfg['output_dir'] = args.output_dir
    cfg['iters'] = args.iters
    cfg['jit_ratio'] = args.jit_ratio
    cfg['prob_syn'] = args.prob_syn

    main(cfg, is_show=args.is_show)
