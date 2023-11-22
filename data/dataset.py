import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageEnhance
import os
import torch
import numpy as np
import json
import cv2
from utils.img_proc.image_process import shape_to_mask, random_crop
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def random_bbox(img_shape=256, max_bbox_shape=256, min_bbox_shape=32):
    '''
    shape: (w, h)
    '''
    random_sz_x = torch.randint(min_bbox_shape, max_bbox_shape+1, (1,)).item() 
    random_sz_y = torch.randint(min_bbox_shape, max_bbox_shape+1, (1,)).item() 
    delta_x = img_shape - random_sz_x
    delta_y = img_shape - random_sz_y
    x1 = torch.randint(0, delta_x+1, (1,)).item()
    y1 = torch.randint(0, delta_y+1, (1,)).item()
    x2 = x1 + random_sz_x
    y2 = y1 + random_sz_y

    return (x1, y1, x2, y2)

def pil_loader(path):
    return Image.open(path).convert('RGB')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = list()
    if os.path.isfile(dir):
        fi = open(dir, 'r')
        for line in fi.readlines():
            line = line.replace('\n', '').strip()
            images.append(line)
        fi.close()
    else:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, 
                 image_size=[256, 256], 
                 loader=pil_loader, 
                 validation_split=40,
                 phase="train",
                 ratio_pos_neg=1.0,
                 prob_inpaint_task=0.5,
                 class_name=None):
        self.ratio_pos_neg = ratio_pos_neg
        self.prob_inpaint_task = prob_inpaint_task
        self.cnt_pos = 0
        self.cnt_neg = 0
        ## defect name and its index, 0 is left to UNK/noneless label
        self.class_dict = {'noneless':0, 'lp':1, 'ng':2, 'huashang':3, 'pengshang':4, 'yise':5, 
                           'aokeng':6, 'heidian':7, 'shahenyin':8, 'bianxing':9, 
                           'tabian':10, 'molie':11, 'gubao':12, 'yiwu':13, 'guashang':14, 
                           'caizhixian':15, 'liewen':16, 'daowen':17, 'zhanya':18, 'aotuhen':19,
                           'cashang':20, 'yashang':21, 'madian':22, 'youmo':23,
                           'zangwu':24, 'baidian':25, 'maoxu':26, 'keli':27,
                           'quepeng':28, 'maoci':29, 'queliao':30, 'quepenghua':31,
                           'wuluowen':32, 'zhanliao':33, 'liuwen':34, 'aotu':35,
                           'juchi':36, 'qipao':37, 'zanghua':38, 'kailie':39, 'xianweimao':40,
                           'nzgs':41, 'jiaobuliang':42, 'aotudian':43}
        if class_name is not None:
            id_codec = 44
            for name in class_name:
                if name in self.class_dict:
                    continue
                self.class_dict[name] = id_codec
                id_codec += 1
        # make dataset
        imgs = make_dataset(data_root)

        if not isinstance(validation_split, int):
            validation_split = int(validation_split * len(imgs))

        if phase == "train":
            self.database = imgs[validation_split:]
        elif phase == "val":
            self.database = imgs[:validation_split]
        elif phase == "test":
            self.database = imgs

        self.phase = phase
        self.loader = loader
        self.image_size = image_size  # (w, h)
        self.len_ = len(self.database)
 
        # transformers
        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.trans_geo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
        self.trans_rotate = transforms.Compose([
            transforms.RandomRotation(45)
        ])
        self.trans_color = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        self.trans_crop = transforms.Compose([
            transforms.RandomCrop((image_size[0], image_size[1]))
        ])

    def __getitem__(self, index):
        ret = {}
        imp = self.database[index]

        if self.cnt_neg > 1e8 or self.cnt_pos > 1e8:
            self.cnt_neg = 0
            self.cnt_pos = 0
        num_try = 0
        while True:
            im_crop, info_crop = random_crop(imp=imp, crop_shape=(self.image_size[0], self.image_size[1]))
            num_try += 1
            if num_try >= 500:
                break

            if im_crop.shape[0] == self.image_size[1] and im_crop.shape[1] == self.image_size[0]:
                ## im_crop is defect image
                if 'shapes' in info_crop and len(info_crop["shapes"]) > 0:
                    self.cnt_pos += 1
                    break

                if 1.0 * self.cnt_pos > self.ratio_pos_neg * self.cnt_neg:
                    self.cnt_neg += 1
                    break
                        
        im_crop = Image.fromarray(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))  
        img = self.tfs(im_crop)

        mask, defect, _ = self.get_mask_and_label(info=info_crop, return_inst=False)
        label = self.defect_to_label(defect=defect)

        if self.phase == "train":
            img_cat = torch.cat([img, mask], dim=0)
            img_cat = self.trans_geo(img_cat)

            img = img_cat[:3, :, :]
            mask = img_cat[3:4, :, :]

        ret['gt_image'] = img
        ret['cond_image'] = mask 
        ret['label'] = label
        ret['mask'] = mask
        ret['path'] = imp.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return self.len_

    def defect_to_label(self, defect):
        if self.phase == 'train' and torch.rand((1,)).item() > 0.9:
            return 0

        label = 1 # 2 means ng, 1 means lp, 0 is left to UNK
        if defect == "lp":
            return label

        label = 2
        if self.phase == 'train' and torch.rand((1,)).item() > 0.8:
            return label
        
        if defect in ['huashang', 'guashang']:
            if torch.rand((1,)).item() > 0.5:
                defect = 'huashang'
            else:
                defect = 'guashang'

        if defect in ['aotuhen', 'aotu']:
            if torch.rand((1,)).item() > 0.5:
                defect = 'aotuhen'
            else:
                defect = 'aotu'

        if defect in self.class_dict:
            label = self.class_dict[defect]
            
        return label

    def get_mask_and_label(self, info, return_inst=True):              

        if 'shapes' not in info or len(info["shapes"]) <= 0:
            label = 'lp' 
            if self.phase == 'train' and torch.rand((1,)).item() > self.prob_inpaint_task:
                mask = torch.ones(1, info["imageHeight"], info["imageWidth"])
                if return_inst:
                    inst = torch.ones(1, info["imageHeight"], info["imageWidth"])
                    return mask, label, inst
                return mask, label, None

            (x1, y1, x2, y2) = random_bbox(img_shape=self.image_size[0])
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])    
            mask[:, y1:y2, x1:x2] = 1.0        
            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                inst[:, y1:y2, x1:x2] = 1.0   
                return mask, label, inst
            return mask, label, None
        
        try:
            label = 'ng' 
            if len(info['shapes']) == 1:
                label = info['shapes'][0]["label"]
            if self.phase == 'train' and torch.rand((1,)).item() > self.prob_inpaint_task:
                mask = torch.ones(1, info["imageHeight"], info["imageWidth"])
                if return_inst:
                    inst = torch.ones(1, info["imageHeight"], info["imageWidth"])
                    return mask, label, inst
                return mask, label, None
            
            MIN_PIX = 32
            idx = torch.randint(0, len(info['shapes']), (1,)).item()
            shape = info['shapes'][idx]
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            image_shape = [info["imageHeight"], info["imageWidth"], 3]
            label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=16, point_size=16)
            label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

            if return_inst:
                label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
                label_inst = np.where(label_inst == True, 1.0, 0).astype('uint8')
                label_inst = np.expand_dims(label_inst, 0)
                inst = torch.from_numpy(label_inst).float()

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
            if torch.rand((1,)).item() > 0.5:
                ksz = torch.randint(3, 5, (1,)).item()
                label_mask = cv2.dilate(label_mask, np.ones((ksz, ksz), dtype=np.uint8), iterations=1)

            mask = np.zeros((info["imageHeight"], info["imageWidth"], 1))
            if shape['label'] not in ['huashang', 'liewen', 'guashang']:
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1            
                mask[y1:y2, x1:x2, :] = 1.0
            else:
                mask = np.expand_dims(label_mask, -1) / 255.0

            mask = torch.from_numpy(mask).float().permute(2, 0, 1)
            if return_inst:
                return mask, label, inst
            return mask, label, None
        except:
            label = 'ng'
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])
            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                return mask, label, inst 
            return mask, label, None
                
class AutoEncoderDataset(data.Dataset):
    def __init__(self, flist_path,
                 image_size=[256, 256],
                 loader=pil_loader, 
                 phase="train"):
        self.phase = phase
        self.loader = loader
        self.image_size = image_size
        
        self.database = list()
        assert os.path.exists(flist_path), 'file is not exists!'
        fi = open(flist_path, 'r')
        for line in fi.readlines():
            line = line.replace('\n', '').strip()
            self.database.append(line)

        self.len_ = len(self.database)
 
        # transformers
        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.trans_geo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
        self.trans_rotate = transforms.Compose([
            transforms.RandomRotation(45)
        ])
        self.trans_color = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        self.trans_crop = transforms.Compose([
            transforms.RandomCrop((image_size[0], image_size[1]))
        ])

    def __getitem__(self, index):
        ret = {}
        path = self.database[index]
        img = self.tfs(self.loader(path))

        if img.shape[1] > self.image_size[0] and img.shape[2] > self.image_size[1]:
            img = self.trans_crop(img)

        if self.phase == "train":
            img = self.trans_geo(img)
            if torch.rand((1,)).item() < 0.2:
                img = self.trans_color(img)
        
        ret['img'] = img
        ret['path'] = path

        return ret

    def __len__(self):
        return self.len_
    
class Inpaint4LoRADataset(data.Dataset):
    def __init__(self, data_root, 
                 image_size=[256, 256], 
                 loader=pil_loader, 
                 validation_split=40,
                 phase="train",
                 class_name=None):
        ## defect name and its index, 0 is left to UNK label
        self.class_dict = {'lp':1, 'ng':2, 'huashang':3, 'pengshang':4, 'yise':5, 
                           'aokeng':6, 'heidian':7, 'shahenyin':8, 'bianxing':9, 
                           'tabian':10, 'molie':11, 'gubao':12, 'yiwu':13, 'guashang':14, 
                           'caizhixian':15, 'liewen':16, 'daowen':17, 'zhanya':18, 'aotuhen':19,
                           'cashang':20, 'yashang':21, 'madian':22, 'youmo':23,
                           'zangwu':24, 'baidian':25, 'maoxu':26, 'keli':27,
                           'quepeng':28, 'maoci':29, 'queliao':30, 'quepenghua':31,
                           'wuluowen':32, 'zhanliao':33, 'liuwen':34, 'aotu':35,
                           'juchi':36, 'qipao':37, 'zanghua':38, 'kailie':39, 'xianweimao':40,
                           'nzgs':41, 'jiaobuliang':42, 'aotudian':43}
        self.extra_class_dict = dict()
        if class_name is not None:
            id_codec = 44
            for name in class_name:
                if name in self.class_dict:
                    continue
                if name not in self.extra_class_dict:
                    self.extra_class_dict[name] = id_codec
                    id_codec += 1

        # self.extra_class_dict = {'wudaojiao':44, 'qikong':45, 'zazhi':46, 'lvxie':47,
        #                          'jiagongbuliang':48, 'lengge':49, 'duoliao':50, 'queliao':51, 'qipi':52}
        # make dataset
        imgs = make_dataset(data_root)

        if not isinstance(validation_split, int):
            validation_split = int(validation_split * len(imgs))

        if phase == "train":
            self.database = imgs[validation_split:]
        elif phase == "val":
            self.database = imgs[:validation_split]
        elif phase == "test":
            self.database = imgs

        self.phase = phase
        self.loader = loader
        self.image_size = image_size  # (w, h)
        self.len_ = len(self.database)
 
        # transformers
        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.trans_geo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
        self.trans_rotate = transforms.Compose([
            transforms.RandomRotation(45)
        ])
        self.trans_color = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        self.trans_crop = transforms.Compose([
            transforms.RandomCrop((image_size[0], image_size[1]))
        ])

    def __getitem__(self, index):
        ret = {}
        imp = self.database[index]
        while True:
            im_crop, info_crop = random_crop(imp=imp, crop_shape=(self.image_size[0], self.image_size[1]))
            if im_crop.shape[0] == self.image_size[1] and im_crop.shape[1] == self.image_size[0]:
                if 'shapes' in info_crop or len(info_crop["shapes"]) > 0:
                    break

                if ('shapes' not in info_crop or len(info_crop["shapes"]) <= 0) and torch.rand((1,)).item() > 0.5:
                    break
        im_crop = Image.fromarray(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))  
        img = self.tfs(im_crop)

        mask, defect, _ = self.get_mask_and_label(info=info_crop, return_inst=False)
        label = self.defect_to_label(defect=defect)

        if self.phase == "train":
            img_cat = torch.cat([img, mask], dim=0)
            img_cat = self.trans_geo(img_cat)

            img = img_cat[:3, :, :]
            mask = img_cat[3:4, :, :]

        ret['gt_image'] = img
        ret['cond_image'] = mask 
        ret['label'] = label
        ret['mask'] = mask
        ret['path'] = imp.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return self.len_

    def defect_to_label(self, defect):
        if self.phase == 'train' and torch.rand((1,)).item() > 0.9:
            return 0

        label = 1 # 2 means ng, 1 means lp, 0 is left to UNK
        if defect == "lp":
            return label

        label = 2
        if self.phase == 'train' and torch.rand((1,)).item() > 0.8:
            return label

        if defect in self.class_dict:
            label = self.class_dict[defect]
        if defect in self.extra_class_dict:
            label = self.extra_class_dict[defect]
            
        return label

    def get_mask_and_label(self, info, return_inst=True):              

        if 'shapes' not in info or len(info["shapes"]) <= 0:
            label = 'lp' 
            if self.phase == 'train' and torch.rand((1,)).item() > 0.5:
                mask = torch.ones(1, info["imageHeight"], info["imageWidth"])
                if return_inst:
                    inst = torch.ones(1, info["imageHeight"], info["imageWidth"])
                    return mask, label, inst
                return mask, label, None

            (x1, y1, x2, y2) = random_bbox(img_shape=self.image_size[0])
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])    
            mask[:, y1:y2, x1:x2] = 1.0        
            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                inst[:, y1:y2, x1:x2] = 1.0   
                return mask, label, inst
            return mask, label, None
        
        try:
            label = 'ng' 
            if len(info['shapes']) == 1:
                label = info['shapes'][0]["label"]
            if self.phase == 'train' and torch.rand((1,)).item() > 0.5:
                mask = torch.ones(1, info["imageHeight"], info["imageWidth"])
                if return_inst:
                    inst = torch.ones(1, info["imageHeight"], info["imageWidth"])
                    return mask, label, inst
                return mask, label, None
            
            MIN_PIX = 32
            idx = torch.randint(0, len(info['shapes']), (1,)).item()
            shape = info['shapes'][idx]
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            image_shape = [info["imageHeight"], info["imageWidth"], 3]
            label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=16, point_size=16)
            label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

            if return_inst:
                label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
                label_inst = np.where(label_inst == True, 1.0, 0).astype('uint8')
                label_inst = np.expand_dims(label_inst, 0)
                inst = torch.from_numpy(label_inst).float()

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
            if torch.rand((1,)).item() > 0.5:
                ksz = torch.randint(3, 5, (1,)).item()
                label_mask = cv2.dilate(label_mask, np.ones((ksz, ksz), dtype=np.uint8), iterations=1)

            mask = np.zeros((info["imageHeight"], info["imageWidth"], 1))
            if shape['label'] not in ['huashang', 'liewen', 'guashang']:
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1            
                mask[y1:y2, x1:x2, :] = 1.0
            else:
                mask = np.expand_dims(label_mask, -1) / 255.0

            mask = torch.from_numpy(mask).float().permute(2, 0, 1)
            if return_inst:
                return mask, label, inst
            return mask, label, None
        except:
            label = 'ng'
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])
            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                return mask, label, inst 
            return mask, label, None
        
class MultiTaskDataset(data.Dataset):
    def __init__(self, data_root, 
                 image_size=[256, 256], 
                 loader=pil_loader, 
                 validation_split=40,
                 phase="train",
                 ratio_pos_neg=2.0,
                 prob_gen_task=0.5,
                 extra_class_name=None):
        self.ratio_pos_neg = ratio_pos_neg
        self.prob_gen_task = prob_gen_task
        self.cnt_pos = 0
        self.cnt_neg = 0
        ## defect name and its index, 0 is left to UNK/noneless label
        self.defect2eng = {'noneless':'noneless', 'lp':1, 'ng':2, 
                            'huashang':'scratch', 'pengshang':'bruise', 'yise':'color variation',
                            'aokeng':'dent', 'heidian':'black spot', 'shahenyin':'sanding marks', 
                            'bianxing':'deformation', 'tabian':'collapse edge', 'molie':'film cracking', 
                            'gubao':'bulge', 'yiwu':'foreign object', 'guashang':'scrape', 
                            'caizhixian':'material line', 'liewen':'crack', 'daowen':'Knife mark', 
                            'zhanya':'adsorption crushing', 'aotuhen':'recessed and raised Marks', 'cashang':'abrasion', 
                            'yashang':'crushing', 'madian':'pitting spots', 'youmo':'ink stain', 'zangwu':'stain', 
                            'baidian':'white spot', 'maoxu':'lint', 'keli':'particles', 'quepeng':'edge bruise', 
                            'maoci':'burrs', 'queliao':'missing material', 'quepenghua':'bruise-scratch',                          
                            'wuluowen':'threadless', 'zhanliao':'residue', 'liuwen':'flow marks', 
                            'aotu':'concave-convex', 'juchi':'serrated edge', 'qipao':'air bubble', 
                            'zanghua':'blemis', 'kailie':'crack', 'xianweimao':'fiber hair', 'nzgs':'unknow', 
                            'jiaobuliang':'poor adhesion', 'aotudian':'concave and convex spot'}
        if extra_class_name is not None:
            for name in extra_class_name:
                if name in self.defect2eng:
                    continue
                self.defect2eng[name] = extra_class_name[name]
        # make dataset
        imgs = make_dataset(data_root)

        if not isinstance(validation_split, int):
            validation_split = int(validation_split * len(imgs))

        if phase == "train":
            self.database = imgs[validation_split:]
        elif phase == "val":
            self.database = imgs[:validation_split]
            if self.prob_gen_task > 0:
                self.prob_gen_task = 1.0
        elif phase == "test":
            self.database = imgs
            if self.prob_gen_task > 0:
                self.prob_gen_task = 1.0

        self.phase = phase
        self.loader = loader
        self.image_size = image_size  # (w, h)
        self.len_ = len(self.database)

    def __getitem__(self, index):
        ret = {}
        imp = self.database[index]

        if self.cnt_neg > 1e8 or self.cnt_pos > 1e8:
            self.cnt_neg = 0
            self.cnt_pos = 0
        num_try = 0
        while True:
            im_crop, info_crop = random_crop(imp=imp, crop_shape=(self.image_size[0], self.image_size[1]))
            if im_crop.shape[0] != self.image_size[1] or im_crop.shape[1] != self.image_size[0]:
                index = torch.randint(0, len(self.database), (1,)).item()
                imp = self.database[index]

            if self.phase != "train":
                break
            num_try += 1
            if num_try >= 100:
                break

            ## we just need ng image
            if self.ratio_pos_neg <= 0:
                if 'shapes' in info_crop and len(info_crop["shapes"]) > 0:
                    break
            ## we just need lp image
            elif self.ratio_pos_neg > 0 and self.ratio_pos_neg < 1.0:
                if 'shapes' not in info_crop and len(info_crop["shapes"]) <= 0:
                    break
            ## lp and ng prob. sample
            else:
                if 'shapes' in info_crop and len(info_crop["shapes"]) > 0:
                    self.cnt_pos += 1
                    break
                else:
                    if 1.0 * self.cnt_pos > self.ratio_pos_neg * self.cnt_neg:
                        self.cnt_neg += 1
                        break  
        # end of while         
                        
        img = torch.from_numpy(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        img = (img / 127.5) - 1 

        mask, label, inst = self.get_mask_and_label(info=info_crop, return_inst=True)
        im_cond, prompt = self.get_prompt(img=im_crop, 
                                          mask=mask.squeeze(0).numpy(),
                                          label=label,
                                          inst=inst.squeeze(0).numpy())

        ret['gt_image'] = img
        ret['cond_image'] = im_cond 
        ret['text'] = prompt
        ret['mask'] = mask
        ret['path'] = imp.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return self.len_

    def get_mask_and_label(self, info, return_inst=False):       
        if 'shapes' not in info or len(info["shapes"]) <= 0:
            label = 'lp' 
            (x1, y1, x2, y2) = random_bbox(img_shape=self.image_size[0])
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])    
            mask[:, y1:y2, x1:x2] = 1.0        
            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                inst[:, y1:y2, x1:x2] = 1.0   
                return mask, label, inst

            return mask, label
        
        try:           
            MIN_PIX = 32
            idx = torch.randint(0, len(info['shapes']), (1,)).item()
            shape = info['shapes'][idx]
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            image_shape = [info["imageHeight"], info["imageWidth"], 3]
            label_mask = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=16, point_size=16)
            label_mask = np.where(label_mask == True, 255, 0).astype('uint8')

            if return_inst:
                label_inst = shape_to_mask(image_shape, points, shape_type=shape_type, line_width=4, point_size=4)
                label_inst = np.where(label_inst == True, 255, 0).astype('uint8')
                label_inst = cv2.dilate(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
                label_inst = cv2.erode(label_inst, np.ones((3, 3), dtype=np.uint8), iterations=1)
                label_inst = np.expand_dims(label_inst, 0)
                inst = torch.from_numpy(label_inst).float() / 255.0

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
            if torch.rand((1,)).item() > 0.5:
                ksz = torch.randint(3, 5, (1,)).item()
                label_mask = cv2.dilate(label_mask, np.ones((ksz, ksz), dtype=np.uint8), iterations=1)

            mask = np.zeros((info["imageHeight"], info["imageWidth"], 1))
            if shape['label'] not in ['huashang', 'liewen', 'guashang']:
                pos = np.argwhere(label_mask > 0)
                (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1            
                mask[y1:y2, x1:x2, :] = 1.0
            else:
                mask = np.expand_dims(label_mask, -1) / 255.0

            mask = torch.from_numpy(mask).float().permute(2, 0, 1)

            if return_inst:
                return mask, label, inst

            return mask, label
        except:
            label = 'ng'
            mask = torch.zeros(1, info["imageHeight"], info["imageWidth"])

            if return_inst:
                inst = torch.zeros(1, info["imageHeight"], info["imageWidth"])
                return mask, label, inst 

            return mask, label
        
    def get_prompt(self, img, mask, label, inst): 
        try:     
            pos = np.argwhere(mask > 0)
            (y1, x1), (y2, x2) = pos.min(0), pos.max(0) + 1
        except:
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
        # area_prompt = str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2) + '.'
        area_prompt = '<{:d}><{:d}><{:d}><{:d}>'.format(x1, y1, x2, y2)
        
        if label == 'lp':
            # prompt = 'This is a inpaint task. The area needs to inpaint is: ' + area_prompt
            prompt = '[inpaint] please inpaint the area: ' + area_prompt
            mask = np.expand_dims(mask, -1)
            im_cond = img * (1 - mask)
            im_cond = torch.from_numpy(cv2.cvtColor(im_cond, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            im_cond = (im_cond / 127.5) - 1 

            return im_cond, prompt
        
        if label == 'ng':
            # prompt = 'This is a defect generation task. The area needs to generate is: ' + area_prompt
            prompt = '[generation] there is a defect in the area: ' + area_prompt
            mask = np.expand_dims(mask, -1)
            im_cond = img * (1 - mask)
            im_cond = torch.from_numpy(cv2.cvtColor(im_cond , cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            im_cond = (im_cond / 127.5) - 1 

            return im_cond, prompt
        
        if torch.rand((1,)).item() <= self.prob_gen_task:
            if label not in self.defect2eng:
                # prompt = 'This is a defect generation task. The area needs to generate is: ' + area_prompt
                prompt = '[generation] there is a defect in the area: ' + area_prompt
            else:
                # prompt = 'This is a {:s} defect generation task. The area needs to generate is: '.format(self.defect2eng[label]) + area_prompt
                prompt = '[generation] there is a {:s} defect in the area: '.format(self.defect2eng[label]) + area_prompt
            mask = np.expand_dims(mask, -1)
            im_cond = img * (1 - mask)
            im_cond = torch.from_numpy(cv2.cvtColor(im_cond, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            im_cond = (im_cond / 127.5) - 1 

            return im_cond, prompt
        
        ## aug
        if label not in self.defect2eng:
            # prompt = 'This is a defect sythesis task. The area needs to synthesize is: ' + area_prompt
            prompt = '[sythesis] please synthesize a defect in the area: ' + area_prompt
        else:
            # prompt = 'This is a {:s} defect sythesis task. The area needs to synthesize is: '.format(self.defect2eng[label]) + area_prompt
            prompt = '[sythesis] please synthesize a {:s} defect in the area: '.format(self.defect2eng[label]) + area_prompt
        mask = np.expand_dims(mask, -1)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        min_th, max_th = 5, 18
        prob_aug = 1.0
        if torch.rand((1,)).item() < prob_aug:
            brightEnhancer = ImageEnhance.Brightness(img_pil)
            bright = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = brightEnhancer.enhance(bright)

        if torch.rand((1,)).item() < prob_aug:
            contrastEnhancer = ImageEnhance.Contrast(img_pil)
            contrast = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = contrastEnhancer.enhance(contrast)

        if torch.rand((1,)).item() < prob_aug:
            colorEnhancer = ImageEnhance.Color(img_pil)
            color = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = colorEnhancer.enhance(color)

        if torch.rand((1,)).item() < prob_aug:
            SharpnessEnhancer = ImageEnhance.Sharpness(img_pil)
            sharpness = torch.randint(min_th, max_th, (1,)).item() / 10.0
            img_pil = SharpnessEnhancer.enhance(sharpness)

        img_aug = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
        inst = np.expand_dims(inst, -1)
        im_cond = img * (1 - inst) + img_aug * inst
        im_cond = torch.from_numpy(cv2.cvtColor(im_cond, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        im_cond = (im_cond / 127.5) - 1 

        return im_cond, prompt
    
class BaseDataset(data.Dataset):
    def __init__(self, data_root, 
                 image_size=[256, 256], 
                 loader=pil_loader, 
                 validation_split=40,
                 phase="train",
                 ratio_pos_neg=2.0,
                 extra_class_name=None):
        self.ratio_pos_neg = ratio_pos_neg
        self.cnt_pos = 0
        self.cnt_neg = 0
        ## defect name and its index, 0 is left to UNK/noneless label
        self.defect2eng = {'noneless':'noneless', 'lp':1, 'ng':2, 
                            'huashang':'scratch', 'pengshang':'bruise', 'yise':'color variation',
                            'aokeng':'dent', 'heidian':'black spot', 'shahenyin':'sanding marks', 
                            'bianxing':'deformation', 'tabian':'collapse edge', 'molie':'film cracking', 
                            'gubao':'bulge', 'yiwu':'foreign object', 'guashang':'scrape', 
                            'caizhixian':'material line', 'liewen':'crack', 'daowen':'Knife mark', 
                            'zhanya':'adsorption crushing', 'aotuhen':'recessed and raised Marks', 'cashang':'abrasion', 
                            'yashang':'crushing', 'madian':'pitting spots', 'youmo':'ink stain', 'zangwu':'stain', 
                            'baidian':'white spot', 'maoxu':'lint', 'keli':'particles', 'quepeng':'edge bruise', 
                            'maoci':'burrs', 'queliao':'missing material', 'quepenghua':'bruise-scratch',                          
                            'wuluowen':'threadless', 'zhanliao':'residue', 'liuwen':'flow marks', 
                            'aotu':'concave-convex', 'juchi':'serrated edge', 'qipao':'air bubble', 
                            'zanghua':'blemis', 'kailie':'crack', 'xianweimao':'fiber hair', 'nzgs':'unknow', 
                            'jiaobuliang':'poor adhesion', 'aotudian':'concave and convex spot'}
        if extra_class_name is not None:
            for name in extra_class_name:
                if name in self.defect2eng:
                    continue
                self.defect2eng[name] = extra_class_name[name]
        # make dataset
        imgs = make_dataset(data_root)

        if not isinstance(validation_split, int):
            validation_split = int(validation_split * len(imgs))

        if phase == "train":
            self.database = imgs[validation_split:]
        elif phase == "val":
            self.database = imgs[:validation_split]
            if self.prob_gen_task > 0:
                self.prob_gen_task = 1.0
        elif phase == "test":
            self.database = imgs
            if self.prob_gen_task > 0:
                self.prob_gen_task = 1.0

        self.phase = phase
        self.loader = loader
        self.image_size = image_size  # (w, h)
        self.len_ = len(self.database)

    def __getitem__(self, index):
        ret = {}
        imp = self.database[index]

        if self.cnt_neg > 1e8 or self.cnt_pos > 1e8:
            self.cnt_neg = 0
            self.cnt_pos = 0
        num_try = 0
        while True:
            im_crop, info_crop = random_crop(imp=imp, crop_shape=(self.image_size[0], self.image_size[1]))
            num_try += 1
            if num_try >= 500:
                break

            if im_crop.shape[0] == self.image_size[1] and im_crop.shape[1] == self.image_size[0]:
                ## im_crop is defect image
                if 'shapes' in info_crop and len(info_crop["shapes"]) > 0:
                    self.cnt_pos += 1
                    break

                if self.ratio_pos_neg <= 0:
                    continue 

                if 1.0 * self.cnt_pos > self.ratio_pos_neg * self.cnt_neg:
                    self.cnt_neg += 1
                    break
                        
        img = torch.from_numpy(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        img = (img / 127.5) - 1 

        prompt = self.get_prompt(info=info_crop)

        ret['gt_image'] = img
        ret['text'] = prompt
        ret['path'] = imp.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return self.len_
        
    def get_prompt(self, info): 
        # prompt = 'A partial iamge of a workpiece surface.'  
        prompt = ''
        if 'shapes' not in info or len(info["shapes"]) <= 0:
            prompt = 'no defects |' 
        else:
            for shape in info["shapes"]:
                label = shape["label"]
                points = shape["points"]
                x_list = [k[0] for k in points]
                y_list = [k[1] for k in points]

                x1, y1, x2, y2 = int(min(x_list)), int(min(y_list)), int(max(x_list)), int(max(y_list))
                area_prompt = '<{:d}><{:d}><{:d}><{:d}>'.format(x1, y1, x2, y2)
                
                if label in self.defect2eng:
                    prompt += 'a {:s} defect in the area '.format(self.defect2eng[label]) + area_prompt + ' | '
                else:
                    prompt += 'a unknown defect in the area ' + area_prompt + ' | '

        return prompt.strip()
