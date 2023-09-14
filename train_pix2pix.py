import time
import sys
import os
sys.path.append('./models/pix2pix_hp')
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from models.pix2pix_hp.options.train_options import TrainOptions
from models.pix2pix_hp.models.pix2pixHD import Pix2PixHDModel
import models.pix2pix_hp.util.util as util
from models.pix2pix_hp.util.visualizer import Visualizer
from data.dataset import Pix2PixHPDataset

from utils.logger.logger import InfoLogger
from utils.visual.visual_writer import VisualWriter
import pdb
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    t = time.localtime() 
    train_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/history_all_340_less200x200.flist'
    test_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/history_all_340_more200x200.flist'
    save_root = '/home/shr/shr_workspace/palette_class/experiments/pix2pix_{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}/'.format(t.tm_mon,
                                                                                                        t.tm_mday, 
                                                                                                        t.tm_hour,
                                                                                                        t.tm_min,
                                                                                                        t.tm_sec)
    os.makedirs(save_root, exist_ok=True)

    log_dir = os.path.join(save_root, 'tb_log')
    result_dir = os.path.join(save_root, 'results')
    ckpts_dir = os.path.join(save_root, 'ckpts')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)
    
    ## config
    opt = TrainOptions().parse()
    opt.checkpoints_dir = ckpts_dir
    opt.name = 'defect'
    opt.continue_train = True
    opt.load_pretrain = '/home/shr/shr_workspace/palette_class/experiments/pix2pix_5420450/ckpts/defect'
    opt.restore_D_path = ''
    opt.which_epoch = 'bak' #'latest'
    opt.input_nc = 4
    opt.output_nc = 4
    opt.ngf = 64
    opt.netG = 'p2p_simple'  # 'gd' | 'p2p_simple'
    opt.ndf = 64
    opt.num_D = 2
    opt.no_ganFeat_loss = False
    opt.n_layers_D = 3
    opt.batchSize = 8
    opt.print_freq = 10
    opt.beta1 = 0.5
    opt.niter = 100
    opt.niter_decay = 10
    opt.gpu_ids = [0]
    opt.display_winsize = 256
    opt.no_lsgan = False
    opt.lambda_feat = 1.0
    opt.rgba_mask_loss = True
    opt.lr = 0.0002
    opt.save_epoch_freq = 1

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    ## logger, visual
    visualizer = Visualizer(ckpts_dir=opt.checkpoints_dir, 
                            name=opt.name, 
                            display_winsize=opt.display_winsize)
    logger = InfoLogger(work_dir=save_root, phase='train', screen=True)
    writer = VisualWriter(log_dir=log_dir, result_dir=result_dir)

    ## Dataset
    logger.info("Loading Train Dataset")    
    train_dataset = Pix2PixHPDataset(flist_path=train_file, phase='train')
    logger.info("Finished Loading Train Dataset")

    logger.info("Creating Dataloader")
    batch = opt.batchSize
    num_worker = batch
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_worker, shuffle=True)  #sampler=sampler
    logger.info("Finished Dataloader")
    dataset_size = len(train_data_loader)

    ## Model
    model = Pix2PixHDModel()
    model.initialize(opt=opt)    
    with_cuda = True if len(opt.gpu_ids) > 0 else False
    cuda_condition = torch.cuda.is_available() and with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    model = model.to(device)
    if len(opt.gpu_ids) > 1:
        import torch.nn as nn
        logger.info("Using %d GPUS for faceAttributer" % len(opt.gpu_ids))
        model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    global_iter = 0
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        for i, data in enumerate(train_data_loader):
            im_gt = data['real'].to(device)
            im_cond = data['cond'].to(device)
            label = data['label'].to(device)
            mask = data['mask'].to(device)
            inst = data['inst'].to(device)
            # pdb.set_trace()

            ############## Forward Pass ######################
            save_fake = False
            if (global_iter + 1) % opt.print_freq == 0:
                save_fake = True
            loss_dict, generated = model(im_gt, im_cond, label, mask, inst, infer=save_fake) 
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + \
                     loss_dict.get('G_GAN_Feat', 0) + \
                     loss_dict.get('G_VGG', 0) + \
                     loss_dict.get('DICE', 0) * 0.01
            loss_dict['d_loss'] = loss_D
            loss_dict['g_loss'] = loss_G
            # pdb.set_trace()
            ############### Backward Pass ####################
            model.optimizer_G.zero_grad()
            loss_G.backward()          
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward() 
            model.optimizer_D.step()
                    
            ############## Display results and errors ##########
            ### display output images
            if save_fake:
                info = 'step: {:d}, lr: {:f}, '.format(global_iter+1, model.old_lr)
                writer.set_iter(epoch, global_iter, phase='train')
                for key, value in loss_dict.items():
                    info += key + ':' + str(value.item()) + ', '
                    writer.add_scalar(key, value.item())
                logger.info(info.strip().strip(','))   

                visuals = OrderedDict([('img_cond', util.tensor2im(data['cond'][0])),
                                    ('img_syn', util.tensor2im(generated.data[0])),
                                    ('img_gt', util.tensor2im(data['real'][0]))])
                visualizer.display_current_results(visuals, epoch) # 确认下中间结果是否保存下来了？

            ### save latest model
            if (global_iter + 1) % (dataset_size//4) == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, global_iter+1))
                model.save('bak')  

            global_iter += 1          
        # end of epoch 
        
        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, global_iter))        
            model.save('bak')
            model.save(str(epoch))
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()
