import torch
import sys
sys.path.append('./models/sd_v15_modules')

from data.dataset import MultiTaskDataset, BaseDataset
from torch.utils.data import DataLoader
import core.util as Util

import numpy as np
import cv2
import pdb


if __name__ == "__main__":
    train_file = 'datasets/history_dataset/train_ldm_his_all.flist'

    ## MultiTaskDataset
    train_dataset = MultiTaskDataset(data_root=train_file, phase='train', ratio_pos_neg=1.0, prob_gen_task=0.5)
    batch = 2
    num_worker = 0
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_worker, shuffle=True)  #sampler=sampler

    for i, ret in enumerate(train_data_loader):
        gt_image = Util.tensor2img(ret['gt_image'])
        cond_image = Util.tensor2img(ret['cond_image'])
        mask = Util.tensor2img(ret['mask']*2-1)
        diff_image = np.abs(gt_image - cond_image)
        
        img_res = np.concatenate([gt_image, cond_image, mask, diff_image], 1)
        img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        
        text = ret['text']
        print(text)
        cv2.imshow('res', img_res)
        cv2.waitKey(0)

    # BaseDataset
    # train_dataset = BaseDataset(data_root=train_file, phase='train')
    # batch = 2
    # num_worker = 0
    # train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_worker, shuffle=True)

    # for i, ret in enumerate(train_data_loader):
    #     gt_image = Util.tensor2img(ret['gt_image'])
    #     img_res = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)        
    #     text = ret['text']

    #     print(text)
    #     cv2.imshow('res', img_res)
    #     cv2.waitKey(0)
    