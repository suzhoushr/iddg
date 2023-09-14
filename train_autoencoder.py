import torch
import torch.nn as nn
import sys
import os
sys.path.append('./models/sd_v15_modules')

from models.sd_v15_modules.autoencoderkl import AutoencoderKL
from data.dataset import AutoEncoderDataset
from torch.utils.data import DataLoader
from models.sd_v15_modules.ldm.modules.losses import LPIPSWithDiscriminator

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import core.util as Util
import time
import logging
import pdb

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, work_dir, phase='train', screen=False):
        self.phase = phase
        self.rank = 0

        self.setup_logger(None, work_dir, phase, level=logging.INFO, screen=screen)
        self.logger = logging.getLogger(phase)
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, log_dir, result_dir):
        self.log_dir = log_dir
        self.result_dir = result_dir

        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = Util.postprocess(results['result'])
            for i in range(len(names)): 
                Image.fromarray(outputs[i]).save(os.path.join(result_path, str(self.iter) + '_' + names[i]))
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr

if __name__ == "__main__":
    #####################  step0: file path #######################   
    t = time.localtime() 
    train_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/train_autoencoder.flist'
    test_file = '/home/shr/shr_workspace/palette_class/datasets/history_dataset/test_autoencoder.flist'
    save_root = '/home/shr/shr_workspace/palette_class/experiments/ldm_aotoencoder_{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}/'.format(t.tm_mon,
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

    logger = InfoLogger(work_dir=save_root, phase='train', screen=True)
    writer = VisualWriter(log_dir=log_dir, result_dir=result_dir)

    #####################  step1: prep train and val dataset #######################
    sys.stdout.flush()
    logger.info("Loading Train Dataset")    
    train_dataset = AutoEncoderDataset(flist_path=train_file, phase='train')
    logger.info("Finished Loading Train Dataset")

    logger.info("Loading Test Dataset")
    test_dataset = AutoEncoderDataset(flist_path=test_file, phase='val')
    logger.info("Finished Loading Train Dataset")

    logger.info("Creating Dataloader")
    batch = 2
    num_worker = batch
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_worker, shuffle=True)  #sampler=sampler
    test_data_loader = DataLoader(test_dataset, batch_size=batch, num_workers=num_worker, shuffle=True)
    logger.info("Finished Dataloader")

    #####################  step2: build model and loss #######################
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
    if len(cuda_devices) > 1:
        logger.info("Using %d GPUS for faceAttributer" % len(cuda_devices))
        model = nn.DataParallel(model, device_ids=cuda_devices)

    disc_start = 50001 * 12 // batch
    kl_weight = 0.000001
    disc_weight = 0.5
    loss_fn = LPIPSWithDiscriminator(disc_start=disc_start, kl_weight=kl_weight, disc_weight=disc_weight).to(device)

    #####################  step3: build optimizer #######################
    learning_rate = 4.5e-6
    curr_lr = learning_rate
    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                              list(model.decoder.parameters())+
                              list(model.quant_conv.parameters())+
                              list(model.post_quant_conv.parameters()),
                              lr=curr_lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                                lr=curr_lr, betas=(0.5, 0.9))

    #####################  step4: train #######################
    num_epochs = 500000
    total_step = len(train_data_loader)
    test_iter = iter(test_data_loader)
    itest = 0
    start_iter = 24
    global_step = 7885000
    step_print_log = 100
    step_val_log = total_step // 10
    epoch_save = 2
    num_val_imgs = min(1000, len(test_data_loader) // 4)
    step_decay_lr = 100
    load_pretrain = True
    if start_iter > 0 or load_pretrain:
        pre_model = "/home/shr/shr_workspace/palette_class/experiments/ldm_aotoencoder_0506135518/ckpts/model_epoch_9.pth"
        model.load_state_dict(torch.load(pre_model), strict=True)

        pre_loss_model = "/home/shr/shr_workspace/palette_class/experiments/ldm_aotoencoder_0506135518/ckpts/loss_model_epoch_9.pth"
        loss_fn.load_state_dict(torch.load(pre_loss_model), strict=True)
    logger.info("Training Start")
    optimizer_idx = 0
    loss_need_print = ['total_loss', 'kl_loss', 'nll_loss', 'g_loss', 'disc_loss', 'logits_real', 'logits_fake']
    for epoch in range(start_iter, num_epochs):
        logger.info('------------------------------Epoch {} Trainning Start------------------------------'.format(str(epoch)))
        count_tr = 0
        for i, ret in enumerate(train_data_loader):
            model.train()
            loss_fn.train()
            inputs = ret['img'].to(device)
            # pdb.set_trace()

            # Forward Pass
            reconstructions, posterior = model(inputs)
            # pdb.set_trace()
            if optimizer_idx == 0: #global_step < disc_start or optimizer_idx == 0:
                # train encoder+decoder+logvar
                loss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                                last_layer=model.get_last_layer(), split="train")
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()

            if optimizer_idx == 1: #global_step < disc_start or optimizer_idx == 1:
                # train the discriminator
                loss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                last_layer=model.get_last_layer(), split="train")
                opt_disc.zero_grad()
                loss.backward()
                opt_disc.step()

            if optimizer_idx == 0:
                optimizer_idx = 1
            else:
                optimizer_idx = 0
            # pdb.set_trace()
            # Backward Pass
            
            ## print the loss in train stage
            if (global_step + 1) % step_print_log == 0:
                writer.set_iter(epoch, global_step, phase='train')
                info = 'step: {:d}, lr: {:f}, '.format(global_step+1, curr_lr)
                for key, value in log_dict_ae.items():
                    if key.split('/')[-1] in loss_need_print:
                        info += key + ':' + str(value.item()) + ', '
                    writer.add_scalar(key, value.item())
                for key, value in log_dict_disc.items():
                    if key.split('/')[-1] in loss_need_print:
                        info += key + ':' + str(value.item()) + ', '
                    writer.add_scalar(key, value.item())
                logger.info(info.strip().strip(','))    
            # pdb.set_trace()        

            # val 
            if (global_step + 1) % step_val_log == 0:
                logger.info('------------------------------Epoch {} Validating Start------------------------------'.format(str(epoch)))
                # Test the model
                # pdb.set_trace()
                model.eval()
                loss_fn.eval()
                split = 'val'
                test_log_dct_ae = {"{}/total_loss".format(split): 0, 
                                   "{}/logvar".format(split): 0,
                                   "{}/kl_loss".format(split): 0, 
                                   "{}/nll_loss".format(split): 0,
                                   "{}/rec_loss".format(split): 0,
                                   "{}/d_weight".format(split): 0,
                                   "{}/disc_factor".format(split): 0,
                                   "{}/g_loss".format(split): 0,
                                   }
                test_log_dct_disc = {"{}/disc_loss".format(split): 0, 
                                   "{}/logits_real".format(split): 0,
                                   "{}/logits_fake".format(split): 0, 
                                   }
                with torch.no_grad():           
                    for k in range(num_val_imgs):
                        ret = next(test_iter)
                        inputs = ret['img'].to(device)
                        names = [name.split('/')[-1] for name in ret['path']]
                        reconstructions, posterior = model(inputs, sample_posterior=False)

                        aeloss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                                      last_layer=model.get_last_layer(), split="val")

                        discloss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                          last_layer=model.get_last_layer(), split="val")
                        
                        for key, value in log_dict_ae.items():
                            test_log_dct_ae[key] += value.item()
                        for key, value in log_dict_disc.items():
                            test_log_dct_disc[key] += value.item()

                        itest += 1
                        if itest == len(test_data_loader):
                            itest = 0
                            test_iter = iter(test_data_loader)
                    
                    info = ''
                    writer.set_iter(epoch, global_step, phase='val')
                    for key, value in test_log_dct_ae.items():
                        value = value / num_val_imgs
                        if key.split('/')[-1] in loss_need_print:
                            info += key + ':' + str(value) + ', '
                        writer.add_scalar(key, value)
                    for key, value in test_log_dct_disc.items():
                        value = value / num_val_imgs
                        if key.split('/')[-1] in loss_need_print:
                            info += key + ':' + str(value) + ', '
                        writer.add_scalar(key, value)
                    logger.info(info.strip().strip(','))

                    im_save = torch.cat([inputs, reconstructions], -1).cpu()
                    results = {'name': names, 'result': im_save}
                    writer.save_images(results)

            global_step += 1

        # Save Model
        if (epoch + 1) % epoch_save == 0:
            logger.info('save epoch {:d} model...'.format(epoch))
            model_name = os.path.join(ckpts_dir, 'model_epoch_{:d}.pth'.format(epoch))
            loss_model_name = os.path.join(ckpts_dir, 'loss_model_epoch_{:d}.pth'.format(epoch))
            torch.save(model.state_dict(), model_name)
            torch.save(loss_fn.state_dict(), loss_model_name)

        # Decay learning rate
        if (epoch + 1) % step_decay_lr == 0:
            curr_lr /= 10
            for param_group in opt_ae.param_groups:
                param_group['lr'] = curr_lr
            for param_group in opt_disc.param_groups:
                param_group['lr'] = curr_lr

    writer.close()
