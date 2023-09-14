## Brief
This project is a Industrial Defect Diffusion Generater which named as IDDG.

## Insatall (you need pytorch env.)
```bash
1. git clone https://gitlab.svfactory.com/Sunnyway/iddg
2. cd ./iddg
3. pip install -r requirements.txt
```

## Train LoRA
LoRA works as a plugins in IDDG. So, to train a LoRA model, you need download the Diffusion Model [NAS PATH: /homes/sunhuanrong/maliang/ldm/v2.0.0].
### prepare data (you need modify the data_dir in the file crop_rdir_imgs_for_lora.py)
``` bash
1. cd ./iddg
2. python scripts/img_crop/crop_rdir_imgs_for_lora.py
```
### modify the config
```text
config/train_lora/train_lora_inpainting_ldm_256.json
```
### train
```bash
1. python run.py -p train -c config/train_lora/train_lora_inpainting_ldm_256.json
```

## Applications
### sythesis data for object detection 
## Trian a IDDG model
