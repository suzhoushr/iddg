import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from models.lora_diffusion.LoRA import inject_trainable_lora
import pdb

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class MaLiang(BaseModel):
    def __init__(self, 
                 networks, 
                 losses, 
                 sample_num, 
                 task, 
                 optimizers, 
                 ema_scheduler=None,     
                 lora_config=None,             
                 **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(MaLiang, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()  

        if lora_config is None:
            '''
            NOTE: It means the lora in infer stage when goes here.
                  That means the 'flag' in lora_config must be infer. 
                  The config is put in model when in train stage
            '''
            if 'lora_config' in self.opt['model']['which_networks'][0]:
                lora_config = self.opt['model']['which_networks'][0]['lora_config']
                if lora_config['flag'] != 'infer':
                    lora_config = None

        self.lora_config = lora_config
        if lora_config is not None and lora_config["flag"] != "not_use":      
            self.netG.eval()      
            unet = None
            if self.opt['distributed']:
                unet = self.netG.module.denoise_fn 
                # unet.requires_grad_(False)
                # if self.netG.module.cond_fn is not None:
                #     self.netG.module.cond_fn.requires_grad_(False)
            else:
                unet = self.netG.denoise_fn
                # unet.requires_grad_(False)
                # if self.netG.cond_fn is not None:
                #     self.netG.cond_fn.requires_grad_(False)
            
            if lora_config["flag"] == "infer":
                assert lora_config["loras"] is not None, "In infer stage, we must have a lora model."
            unet_lora_params, _ = inject_trainable_lora(model=unet,
                                                        loras=lora_config["loras"],
                                                        r=lora_config["lora_rank"],
                                                        target_replace_module=lora_config["target_replace_module"],
                                                        dropout_p=lora_config["lora_dropout_p"],
                                                        scale=lora_config["lora_scale"],
                                                        inject_embedding=lora_config["inject_emb"])

        if lora_config is None or lora_config["flag"] != "ft":        
            if self.opt['distributed']:
                opt_params = list(self.netG.module.denoise_fn.parameters())
                if self.netG.module.cond_fn is not None:
                    opt_params += list(self.netG.module.cond_fn.parameters())
            else:
                opt_params = list(self.netG.denoise_fn.parameters())
                if self.netG.cond_fn is not None:
                    opt_params += list(self.netG.cond_fn.parameters())

            self.optG = torch.optim.AdamW(opt_params, **optimizers[0])
            self.optimizers.append(self.optG)
            self.resume_training() 
        else:
            # if self.opt['distributed']:
            #     opt_params = list(self.netG.module.denoise_fn.label_emb.parameters())
            # else:
            #     opt_params = list(self.netG.denoise_fn.label_emb.parameters())
            opt_params = list(unet_lora_params[0])
            for i in range(1, len(unet_lora_params)):
                opt_params += list(unet_lora_params[i])
            self.optG = torch.optim.AdamW(opt_params, lr=lora_config['lora_lr'])
            self.optimizers.append(self.optG)

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase, device=self.opt['device'])

        ''' can rewrite in inherited class for more informations logging '''
        if kwargs['logger'] is not None:
            self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
            self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
            self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

        if self.opt['phase'] == 'test':
            self.netG.eval()
            self.optG = None
            self.optimizers = list()
            self.loss_fn = None
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        ## we must have gt_image
        self.gt_image = self.set_device(data.get('gt_image'))
        self.path = data['path']
        self.batch_size = len(data['path'])

        self.cond_image, self.mask, self.inst, self.label, self.text = None, None, None, None, None
        if "cond_image" in data:
            self.cond_image = self.set_device(data['cond_image'])
        if "mask" in data:
            self.mask = self.set_device(data['mask'])
        if "inst" in data:
            self.inst = self.set_device(data['inst'])
        if "label" in data:
            self.label = self.set_device(data['label'])
        if "text" in data:
            self.text = self.set_device(data['text'])        
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2
        }
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        # pdb.set_trace()
        # self.netG.train()
        if self.lora_config is not None and self.lora_config["flag"] != "ft":
            if self.opt['distributed']:
                self.netG.module.denoise_fn.train()
                if self.netG.module.cond_fn is not None:
                    self.netG.module.cond_fn.train()
            else:
                self.netG.denoise_fn.train()
                if self.netG.cond_fn is not None:
                    self.netG.cond_fn.train()

        self.train_metrics.reset()
        accum_iter = 4
        for batch_idx, train_data in enumerate(tqdm.tqdm(self.phase_loader)):
            self.set_input(train_data)

            loss = self.netG(y_0=self.gt_image, 
                             y_cond=self.cond_image, 
                             mask=self.mask,
                             inst=self.inst,                 
                             label=self.label, 
                             text=self.text)
            loss = loss / accum_iter 
            loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(self.phase_loader)):
                self.optG.step()
                self.optG.zero_grad()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, accum_iter * loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting']:
                        self.output, self.visuals = self.netG.module.inpaint_restoration(y_0=self.gt_image, 
                                                                                         y_cond=self.cond_image,                             
                                                                                         mask=self.mask,
                                                                                         label=self.label,
                                                                                         text=self.text,
                                                                                         sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting']:
                        self.output, self.visuals = self.netG.inpaint_restoration(y_0=self.gt_image, 
                                                                                  y_cond=self.cond_image,                             
                                                                                  mask=self.mask,
                                                                                  label=self.label,
                                                                                  text=self.text,
                                                                                  sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting']:
                        self.output, self.visuals = self.netG.module.inpaint_restoration(y_0=self.gt_image, 
                                                                                         y_cond=self.cond_image,                             
                                                                                         mask=self.mask,
                                                                                         label=self.label,
                                                                                         text=self.text,
                                                                                         sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting']:
                        self.output, self.visuals = self.netG.inpaint_restoration(y_0=self.gt_image, 
                                                                                  y_cond=self.cond_image,                             
                                                                                  mask=self.mask,
                                                                                  label=self.label,
                                                                                  text=self.text,
                                                                                  sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))
    
    def inpaint_generater(self, 
                          gt_image, 
                          cond_image=None, 
                          mask=None,
                          label=None,
                          text=None, 
                          ratio=1.0, 
                          gd_w=0.0, 
                          return_visuals=False):
        with torch.no_grad():
            output, visuals = self.netG.inpaint_restoration(y_0=gt_image, 
                                                            y_cond=cond_image,                             
                                                            mask=mask,
                                                            label=label,
                                                            text=text,
                                                            sample_num=self.sample_num,
                                                            ratio=ratio, 
                                                            gd_w=gd_w)
        if return_visuals:
            return output, visuals
        return output

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)

        if self.lora_config is not None and self.lora_config["flag"] != 'ft':
            if self.ema_scheduler is not None:
                self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
            self.save_training_state()
