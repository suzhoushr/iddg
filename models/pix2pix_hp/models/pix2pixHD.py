import itertools
import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
from torch.nn.parallel import gather
import torch.nn.functional as F
import random
import pdb

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
        
    def initialize(self, opt):       
        BaseModel.initialize(self, opt)
        if not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True

        self.isTrain = opt.isTrain
        input_nc = opt.input_nc
        output_nc = opt.output_nc
        ngf = opt.ngf
        netG_name = opt.netG

        ##### define networks                        
        self.netG = networks.define_G(input_nc=input_nc, 
                                      output_nc=output_nc, 
                                      ngf=ngf, 
                                      netG_name=netG_name)
        # Discriminator network
        if self.isTrain: 
            input_nc = opt.input_nc - 1
            ndf = opt.ndf
            n_layers_D = opt.n_layers_D
            num_D = opt.num_D
            getIntermFeat = not opt.no_ganFeat_loss
            self.netD = networks.define_D(input_nc=input_nc, 
                                          ndf=ndf, 
                                          n_layers_D=n_layers_D,
                                          num_D=num_D,
                                          getIntermFeat=getIntermFeat)

        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # pdb.set_trace()
            if not self.isTrain:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)          
            if self.isTrain:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                if not os.path.isfile(opt.restore_D_path):
                    self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path) 

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss() 
            self.criterionVGG = networks.VGGLoss()

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, test_image):
        input = test_image.detach()
        return self.netD.forward(input)

    def forward(self, im_gt, im_cond, label, mask, inst, infer=False):
        # Fake Generation
        input_concat = torch.cat([im_cond, mask], 1)
        fake_image = self.netG.forward(input_concat, y=label)
        fake_image = torch.tanh(fake_image)
            
        if self.opt.rgba_mask_loss:
            fake_mask  = 0.5 * (1 + fake_image[:, 3, :, :].unsqueeze(1))
            fake_image = fake_image[:, :3, :, :]

            fake_mask = fake_mask * mask        
            fake_image = fake_image * mask + im_gt * (1.0 - mask)      

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(fake_image)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)  

        # Real Detection and Loss        
        pred_real = self.discriminate(im_gt)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(fake_image) 
        loss_G_GAN = self.criterionGAN(pred_fake, True)  
      
        # GAN feature matching loss
        loss_G_GAN_Feat = 0.0
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                   
        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(fake_image, im_gt)

        
        # RGBA的mask
        loss_dice = 0
        if self.opt.rgba_mask_loss:
            N = 2 * (fake_mask * inst).sum(dim=[1,2,3])
            U = inst.sum(dim=[1,2,3]) + fake_mask.sum(dim=[1,2,3]) + 1e-20
            loss_dice = 1 - N / U
            loss_dice = loss_dice.mean()

        loss_d = {"D_fake": loss_D_fake,
               "D_real": loss_D_real,
               "G_GAN": loss_G_GAN,
               "G_GAN_Feat": loss_G_GAN_Feat,
               "G_VGG": loss_G_VGG,
               "DICE": loss_dice}
        
        # if infer:
        #     fake_image = fake_image * fake_mask + im_gt * (1.0 - fake_mask)  
                
        return loss_d, None if not infer else fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


        
