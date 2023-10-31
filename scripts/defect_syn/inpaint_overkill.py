import sys
sys.path.append("./")
from tools.engine import Engine
import shutil
import cv2
import os
import sys
from tqdm import tqdm
import argparse
from core.praser import update_config
import pdb

def main(args, is_show=False):
    engine = Engine(args, img_sz=256)

    src_dir = args['src_data_dir'] 
    pred_dir = args['pred_data_dir']     
    if src_dir[-1] != '/':
        src_dir += '/'
    if pred_dir != '/':
        pred_dir += '/' 

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

    out_dir = args['output_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## ng to liangpin
    for imp in tqdm(src_imgs_list):    
        sub_dir = imp.split('/')[-2]
        save_dir = os.path.join(out_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        jsp = imp.replace('.jpg', '.json')  
        jsp_pred = jsp.replace(src_dir, pred_dir)

        ret, img, new_info = engine.synthesis(imp, jsp_pred, 
                                              defect_need_gen=args['defect_need_gen'], 
                                              gid_need_gen=args['gid_need_gen'], 
                                              ratio=1.0, 
                                              gd_w=args['gd_w'],
                                              prob_syn=1.0,
                                              mask_type='rect', # rect | poly
                                              SHOW=is_show,
                                              task="inpaint")  # aug | inpaint

        name = imp.split('/')[-1][:-4]         
        dst_imp = os.path.join(save_dir, name + '.jpg')
        dst_jsp = os.path.join(save_dir, name + '.json')
        cv2.imwrite(dst_imp, img)
        shutil.copy(jsp, dst_jsp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical characteristic size")
    ## model configuration
    parser.add_argument("--config_file", type=str, default='./config/infer/infer_v1.2.0.json', help="")
    parser.add_argument("--resume_path", type=str, default='', help="")
    parser.add_argument("--sample_type", type=str, default='ddim', choices=["ddpm", "ddim", "dpmsolver", "dpmsolver++"], help="")
    parser.add_argument("--sample_timesteps", type=int, default=100, help="")
    parser.add_argument("--gd_w", type=float, default=0.0, help="")
    parser.add_argument("--gid_need_gen", type=int, nargs='*', default=['PRED_kill'], help="")
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
    parser.add_argument("--src_data_dir", type=str, default='/your/liangpin/data/dir/path/', help="")
    parser.add_argument("--pred_data_dir", type=str, default='/your/liangpin/data/dir/path/', help="")
    parser.add_argument("--output_dir", type=str, default='/your/output/data/dir/path/', help="")
    
    args = parser.parse_args()
    cfg = update_config(args)

    cfg['src_data_dir'] = args.src_data_dir
    cfg['pred_data_dir'] = args.pred_data_dir
    cfg['output_dir'] = args.output_dir

    main(cfg, is_show=args.is_show)
