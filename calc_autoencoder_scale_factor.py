import torch
import sys
sys.path.append('./models/sd_v15_modules')

from models.sd_v15_modules.autoencoderkl import AutoencoderKL
from torchvision import transforms

from PIL import Image
import cv2
import core.util as Util
from PIL import Image
import pdb

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
    pre_model = '/data/RND/sunhuanrong/maliang/experiments/ldm_aotoencoder_0810095435/ckpts/model_epoch_119.pth'
    model.load_state_dict(torch.load(pre_model, map_location="cpu"), strict=True)
    tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.CenterCrop((256, 256))
    ])
     
    test_file = '/data/RND/sunhuanrong/maliang/datasets/history/train_autoencoder.flist'
    sample_len = 10000
    sample_z = list()
    for line in open(test_file, 'r').readlines():
        imp = line.replace('\n', '').strip()
        img = Image.open(imp).convert('RGB')
        img = tfs(img)
        img = img.to(device).unsqueeze(0)
        with torch.no_grad(): 
            z = model.encode(img).sample()
            sample_z.append(z)
        if len(sample_z) >= sample_len:
            break
    sample_z = torch.cat(sample_z, 0)
    print(sample_z.std().item())

    