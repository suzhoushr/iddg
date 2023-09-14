import torch
import sys
sys.path.append('./models/sd_v15_modules')

from models.sd_v15_modules.autoencoderkl import AutoencoderKL
from torchvision import transforms

from PIL import Image
import cv2
import numpy as np
import core.util as Util
import json
from PIL import Image, ImageDraw
import math
import os
import glob
from tqdm import tqdm
import pdb

def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1, width=line_width//2)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1, width=line_width//2)
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
        draw.polygon(xy=xy, outline=1, fill=1, width=line_width//2)
    mask = np.array(mask, dtype=bool)
    
    return mask

def get_bbox(imp):
    jsp = imp.replace('.jpg', '.json')
    with open(jsp, 'r', encoding='utf-8') as jf:
        info = json.load(jf)
    image_shape = [info["imageHeight"], info["imageWidth"], 3]
    mask = np.zeros((info["imageHeight"], info["imageWidth"]))

    MIN_PIX = 16
    shape = info['shapes'][0]
    # pdb.set_trace()
    points = shape["points"]
    shape_type = shape["shape_type"]
    label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=16, point_size=8)
    label_mask = np.where(label_mask == True, 255, 0).astype('uint8')
    if label_mask.shape[0] == 340:
        mask = label_mask[42:298, 42:298]
    elif label_mask.shape[0] == 256:
        mask = label_mask

    if mask.sum() <= 0:
        return 0, 0, 0, 255, 255, shape['label']
    else:
        pos = np.argwhere(mask > 0)
        (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1

        h = y2 - y1
        w = x2 - x1
        min_h_w = min(h, w)
        if min_h_w < MIN_PIX:
            delta = MIN_PIX - min_h_w + 1
            mask = cv2.dilate(mask, np.ones((delta, delta), dtype=np.uint8), iterations=1)
            pos = np.argwhere(mask > 0)
            (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1

        return 1, x1, y1, x2, y2, shape['label']

if __name__ == "__main__":

    #####################  step1: build model and loss #######################
    with_cuda = True
    cuda_condition = torch.cuda.is_available() and with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    cuda_devices = [0] if cuda_condition else []

    ddconfig = {"double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0}
    embed_dim = 4

    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=embed_dim)
    model = model.to(device)

    #####################  step2: test #######################
    pre_model = "/home/shr/shr_workspace/palette_class/experiments/ldm_aotoencoder_0512105420/ckpts/model_epoch_33.pth"
    pre_model = "/home/shr/shr_workspace/palette_class/experiments/ldm_aotoencoder_0506135518/ckpts/model_epoch_9.pth"
    model.load_state_dict(torch.load(pre_model, map_location="cpu"), strict=True)
    tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.CenterCrop((512, 512))
    ])

    USE_EXP = False
    if not USE_EXP:        
        test_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/test_autoencoder_his.flist'
        # test_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/test_autoencoder_cvpr.flist'
        test_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/test_autoencoder_taida.flist'
        concat_dir = os.path.join('/home/shr/shr_workspace/palette_class/test/test_autoencoder_diff_sz/', 'concat512')

        os.makedirs(concat_dir, exist_ok=True)
        
        Print_Info = False
        for line in open(test_file, 'r').readlines():
            imp = line.replace('\n', '').strip()
            img = Image.open(imp).convert('RGB')
            if Print_Info:
                ret, x1, y1, x2, y2, defect_name = get_bbox(imp)
                if ret <= 0:
                    continue
            img = tfs(img)
            img = img.to(device).unsqueeze(0)
            with torch.no_grad(): 
                z = model.encode(img).mode()
                pred_img = model.decode(z)      

            img = Util.tensor2img(img.cpu())
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
            pred_img = Util.tensor2img(pred_img.cpu())
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)   

            if Print_Info:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
                
            # img_con = np.concatenate((img, pred_img), 1)
            gap = 10
            tex = 20
            sz = img.shape[0]
            img_con = 255 * np.ones((sz + 20, sz*2 + 10, 3)).astype('uint8')
            img_con[tex:, :sz, :] = img
            img_con[tex:, sz+gap:, :] = pred_img
            cv2.putText(img_con, "ori(512x512)", (sz//2-20, 12), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
            cv2.putText(img_con, "v0.1.2", (sz//2+sz-20, 12), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)


            if Print_Info:
                cv2.putText(img_con, defect_name, (225, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("test", img_con)
            cv2.waitKey(0) 

            name = imp.split('/')[-1]
            cv2.imwrite(os.path.join(concat_dir, name), img_con)

    else:
        data_root = '/home/shr/shr_workspace/palette_class/test/test_autoencoder_his/'
        epoch = '23'
        imgs_dir = os.path.join(data_root, 'ori')
        src_dir = os.path.join(data_root, 'src_' + epoch)
        dst_dir = os.path.join(data_root, 'dst_' + epoch)
        concat_dir = os.path.join(data_root, 'concat_' + epoch)

        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)

        imgs_list = glob.glob(imgs_dir + '/*.jpg')

        for imp in tqdm(imgs_list):
            img = Image.open(imp).convert('RGB')
            img = tfs(img)
            img = img.to(device).unsqueeze(0)
            with torch.no_grad(): 
                z = model.encode(img).mode()
                pred_img = model.decode(z)      

            img = Util.tensor2img(img.cpu())
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
            pred_img = Util.tensor2img(pred_img.cpu())
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

            name = imp.split('/')[-1]
            cv2.imwrite(os.path.join(src_dir, name), img)
            cv2.imwrite(os.path.join(dst_dir, name), pred_img)

            gap = 10
            tex = 20
            sz = img.shape[0]
            img_con = 255 * np.ones((sz + 20, sz*2 + 10, 3)).astype('uint8')
            img_con[tex:, :sz, :] = img
            img_con[tex:, sz+gap:, :] = pred_img
            cv2.putText(img_con, "ori(512x512)", (sz//2-20, 12), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
            cv2.putText(img_con, "v0.1.2", (sz//2+sz-20, 12), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
            cv2.imwrite(os.path.join(concat_dir, name), img_con)

            if 0:
                cv2.imshow("test", img_con)
                cv2.waitKey(0) 
            
    