from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
import json

def dHash(imp):
    image = Image.open(imp)
    image = np.array(image.resize((9, 8), Image.Resampling.LANCZOS).convert('L'), 'f')

    hash = []
    for i in range(8):
        for j in range(8):
            if image[i,j] > image[i,j+1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def Hamming_distance(hash1, hash2): 
    num = 0
    for index in range(len(hash1)): 
        if hash1[index] != hash2[index]: 
            num += 1
    return num

class dHashBase():
    def __init__(self, data_dir):
        self.imgs_list = list()
        for filepath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if '.jpg' not in filename:
                    continue
                imp = os.path.join(filepath, filename)            
                self.imgs_list.append(imp)

    def buildDB(self,):
        self.hashDB = dict()
        for imp in tqdm(self.imgs_list):
            dhash = dHash(imp)
            self.hashDB[imp] = dhash

    def search(self, imp, top_k=1, thrs=3):
        ref_dhash = dHash(imp)
        res_imgs = self.search_(ref_dhash, top_k=top_k, thrs=thrs)
        return res_imgs

    def search_(self, ref_dhash, top_k=1, thrs=3):
        d_sim = dict()
        for i in range(thrs+1):
            d_sim[i] = list()

        for imp in self.hashDB.keys():
            sea_dhash = self.hashDB[imp]
            dist = Hamming_distance(ref_dhash, sea_dhash)
            if dist > thrs:
                continue

            d_sim[dist].append(imp)

        res = list()
        for i in range(thrs+1):
            if len(res) >= top_k:
                break
            res.extend(d_sim[i])
        
        np.random.shuffle(res)

        return res[:top_k]

if __name__ == "__main__":
    defect_data_dir = '/home/data0/project-datasets/dehongcheng/SYB/train_val_data/train/train_origin_data_crop'
    lp_data_dir = '/home/data0/project-datasets/dehongcheng/SYB/train_val_data/train/train_origin_data_crop_lp'

    db_lp_dhash = dHashBase(data_dir=lp_data_dir)
    db_lp_dhash.buildDB()

    defect_imgs_list = list()
    for filepath, _, filenames in os.walk(defect_data_dir):
        for filename in filenames:
            if '.jpg' not in filename:
                continue
            imp = os.path.join(filepath, filename)            
            defect_imgs_list.append(imp)

    for imp in defect_imgs_list:
        candi_imps = db_lp_dhash.search(imp=imp, top_k=1, thrs=32)
        if len(candi_imps) <= 0:
            print("Opps, there is not similarity image in LP images!")
            continue

        defect_img = cv2.imread(imp)
        lp_img = cv2.imread(candi_imps[0])

        jsp = imp.replace('.jpg', '.json')
        with open(jsp,'r',encoding ='utf-8') as jf:
            info = json.load(jf)
        for _shape in info['shapes']:
            points = _shape["points"]
            label = _shape["label"]
            x_list = [k[0] for k in points]
            y_list = [k[1] for k in points]

            x1 = int(min(x_list))
            y1 = int(min(y_list))
            x2 = int(max(x_list))
            y2 = int(max(y_list))

            cv2.rectangle(defect_img, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)
            cv2.putText(defect_img, label, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)

            cv2.rectangle(lp_img, (x1, y1), (x2, y2), (255, 0, 0), 1, 4)

        img_res = np.concatenate([defect_img, lp_img], 1)
        cv2.imshow('res', img_res)
        cv2.waitKey(0)
