import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
    def __init__(self, work_dir, phase='train', rank=0, screen=False):
        self.phase = phase
        self.rank = rank

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
    def __init__(self, log_dir, result_dir, rank=0, enabled=False):
        self.rank = rank
        self.enabled = enabled
        self.log_dir = log_dir
        self.result_dir = result_dir

        self.writer = None
        if self.enabled and self.rank == 0:
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
        
def main_worker(gpu, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True

    ## logger and writer
    logger = InfoLogger(work_dir=opt['save_root'], phase='train', rank=gpu, screen=True)
    writer = VisualWriter(log_dir=opt['log_dir'], result_dir=opt['result_dir'], rank=gpu)

    ## dataloader
    train_file = opt['train_file']
    train_dataset = AutoEncoderDataset(flist_path=train_file, phase='train')
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_data_loader = DataLoader(train_dataset, sampler=data_sampler, batch_size=opt['batch'], num_workers=opt['num_worker'], shuffle=False)
    else:
        train_data_loader = DataLoader(train_dataset, sampler=data_sampler, batch_size=opt['batch'], num_workers=opt['num_worker'], shuffle=True)

    if gpu == 0:
        test_file = opt['test_file']
        test_dataset = AutoEncoderDataset(flist_path=test_file, phase='val')
        test_data_loader = DataLoader(test_dataset, batch_size=opt['batch'], num_workers=opt['num_worker'], shuffle=True)
        test_iter = iter(test_data_loader)

    ## model
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
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=embed_dim).to(gpu)
    if opt['model_ckpt'] is not None:
        model.load_state_dict(torch.load(opt['model_ckpt']), strict=True)
    
    disc_start = 50001 * 12 // opt['batch']
    kl_weight = 0.000001
    disc_weight = 0.5
    loss_fn = LPIPSWithDiscriminator(disc_start=disc_start, kl_weight=kl_weight, disc_weight=disc_weight).to(gpu)
    if opt['dis_ckpt'] is not None:
        loss_fn.load_state_dict(torch.load(opt['dis_ckpt']), strict=True)

    ## optimizer
    curr_lr = opt['curr_lr']
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                              list(model.decoder.parameters())+
                              list(model.quant_conv.parameters())+
                              list(model.post_quant_conv.parameters()),
                              lr=curr_lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                                lr=curr_lr, betas=(0.5, 0.9))

    if opt['distributed']:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        loss_fn = DDP(loss_fn, device_ids=[gpu], find_unused_parameters=True)

    ## train
    logger.info("Training Start")
    optimizer_idx = 0
    loss_need_print = ['total_loss', 'kl_loss', 'nll_loss', 'g_loss', 'disc_loss', 'logits_real', 'logits_fake']
    global_step = opt['global_step']
    step_val_log = min(opt['step_val_log'], len(train_data_loader) // 5) 
    
    itest = 0
    for epoch in range(opt['start_iter'], opt['num_epochs']):
        if gpu == 0:
            logger.info('------------------------------Epoch {} Trainning Start------------------------------'.format(str(epoch)))
        if opt['distributed']:
            train_data_loader.sampler.set_epoch(epoch)
        for i, ret in enumerate(train_data_loader):
            model.train()
            loss_fn.train()
            inputs = ret['img'].to(gpu)

            # Forward Pass
            reconstructions, posterior = model(inputs)

            if optimizer_idx == 0:
                # train encoder+decoder+logvar
                if opt['distributed']:
                    loss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                                    last_layer=model.module.get_last_layer(), split="train")
                else:
                    loss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                                last_layer=model.get_last_layer(), split="train")
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()

            if optimizer_idx == 1:
                # train the discriminator
                if opt['distributed']:
                    loss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                    last_layer=model.module.get_last_layer(), split="train")
                else:
                    loss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                last_layer=model.get_last_layer(), split="train")
                opt_disc.zero_grad()
                loss.backward()
                opt_disc.step()

            if optimizer_idx == 0:
                optimizer_idx = 1
            else:
                optimizer_idx = 0
            
            ## print the loss in train stage
            if gpu == 0 and (global_step + 1) % opt['step_print_log'] == 0:
                writer.set_iter(epoch, global_step, phase='train')
                info = 'step: {:d}, gpu: {:d}, lr: {:.10f}, '.format(global_step+1, gpu, curr_lr)
                for key, value in log_dict_ae.items():
                    if key.split('/')[-1] in loss_need_print:
                        # info += key + ':' + str(value.item()) + ', '
                        info += key + ': {:6f}, '.format(value.item())
                    writer.add_scalar(key, value.item())
                for key, value in log_dict_disc.items():
                    if key.split('/')[-1] in loss_need_print:
                        # info += key + ':' + str(value.item()) + ', '
                        info += key + ': {:6f}, '.format(value.item())
                    writer.add_scalar(key, value.item())
                logger.info(info.strip().strip(','))    
            # pdb.set_trace()        

            # val 
            if gpu == 0 and (global_step + 1) % step_val_log == 0:
                logger.info('------------------------------Epoch {} Validating Start------------------------------'.format(str(epoch)))
                num_val_imgs = min(opt['num_val_imgs'], len(test_data_loader) // 4)
                # Test the model
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
                        inputs = ret['img'].to(gpu)
                        names = [name.split('/')[-1] for name in ret['path']]
                        reconstructions, posterior = model(inputs, sample_posterior=False)

                        if opt['distributed']:
                            aeloss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                                        last_layer=model.module.get_last_layer(), split="val")

                            discloss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                            last_layer=model.module.get_last_layer(), split="val")
                        else:
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
                    
                    info = 'gpu: {:d}, '.format(gpu)
                    writer.set_iter(epoch, global_step, phase='val')
                    for key, value in test_log_dct_ae.items():
                        value = value / num_val_imgs
                        if key.split('/')[-1] in loss_need_print:
                            # info += key + ':' + str(value) + ', '
                            info += key + ': {:6f}, '.format(value)
                        writer.add_scalar(key, value)
                    for key, value in test_log_dct_disc.items():
                        value = value / num_val_imgs
                        if key.split('/')[-1] in loss_need_print:
                            # info += key + ':' + str(value) + ', '
                            info += key + ': {:6f}, '.format(value)
                        writer.add_scalar(key, value)
                    logger.info(info.strip().strip(','))

                    im_save = torch.cat([inputs, reconstructions], -1).cpu()
                    results = {'name': names, 'result': im_save}
                    writer.save_images(results)

            global_step += 1

        # Save Model
        if gpu == 0 and (epoch + 1) % opt['epoch_save'] == 0:
            logger.info('save epoch {:d} model...'.format(epoch))
            model_name = os.path.join(opt['ckpts_dir'], 'model_epoch_{:d}.pth'.format(epoch))
            loss_model_name = os.path.join(opt['ckpts_dir'], 'loss_model_epoch_{:d}.pth'.format(epoch))
            if opt['distributed']:
                torch.save(model.module.state_dict(), model_name)
                torch.save(loss_fn.module.state_dict(), loss_model_name)
            else:
                torch.save(model.state_dict(), model_name)
                torch.save(loss_fn.state_dict(), loss_model_name)

            # state_dict = model.state_dict()
            # for key, param in state_dict.items():
            #     state_dict[key] = param.cpu()
            # torch.save(state_dict, model_name)

            # state_dict = loss_fn.state_dict()
            # for key, param in state_dict.items():
            #     state_dict[key] = param.cpu()
            # torch.save(state_dict, loss_model_name)

        # Decay learning rate
        if (epoch + 1) % opt['step_decay_lr'] == 0:
            curr_lr /= 10
            for param_group in opt_ae.param_groups:
                param_group['lr'] = curr_lr
            for param_group in opt_disc.param_groups:
                param_group['lr'] = curr_lr

    writer.close()

if __name__ == "__main__":
    ## configs
    t = time.localtime() 
    save_root = 'experiments/ldm_aotoencoder_{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}/'.format(t.tm_mon,
                t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    os.makedirs(save_root, exist_ok=True)
    log_dir = os.path.join(save_root, 'tb_log')
    result_dir = os.path.join(save_root, 'results')
    ckpts_dir = os.path.join(save_root, 'ckpts')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)
    train_file = 'datasets/history/train_autoencoder.flist' 
    test_file = 'datasets/history/test_autoencoder.flist' 
    model_ckpt = 'experiments/autoencoder/model_epoch_119.pth'
    dis_ckpt = 'experiments/autoencoder/loss_model_epoch_119.pth'

    batch = 12
    num_worker = 12
    lr = 5e-6
    gpu_ids = [2,3,4,5]
    port = '21012'    
    
    start_iter = 120
    num_epochs = 500000
    global_step = 10262200
    step_print_log = 100
    step_val_log = 10000
    epoch_save = 2
    num_val_imgs = 1000
    step_decay_lr = 100

    ## opt
    opt = dict()
    opt['save_root'] = save_root
    opt['log_dir'] = log_dir
    opt['result_dir'] = result_dir
    opt['ckpts_dir'] = ckpts_dir
    opt['train_file'] = train_file
    opt['test_file'] = test_file
    if model_ckpt == '':
        opt['model_ckpt'] = None
    else:
        opt['model_ckpt'] = model_ckpt
    if dis_ckpt == '':
        opt['dis_ckpt'] = None
    else:
        opt['dis_ckpt'] = dis_ckpt

    opt['start_iter'] = start_iter 
    opt['num_epochs'] = num_epochs
    opt['global_step'] = global_step
    opt['step_print_log'] = step_print_log
    opt['step_val_log'] = step_val_log
    opt['epoch_save'] = epoch_save
    opt['num_val_imgs'] = num_val_imgs
    opt['step_decay_lr'] = step_decay_lr

    opt['batch'] = batch
    opt['num_worker'] = num_worker
    opt['curr_lr'] = lr
    opt['port'] = port
    opt['gpu_ids'] = gpu_ids
    opt['distributed'] = False
    if len(opt['gpu_ids']) > 1:
        opt['distributed'] = True
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ opt['port']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt,))
    else:
        opt['world_size'] = 1 
        opt['device'] = 'cpu'
        if len(opt['gpu_ids']) >= 1:
            opt['device'] = 'cuda:' + str(opt['gpu_ids'][0])
        main_worker(0, opt)
