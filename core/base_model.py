import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn

import core.util as Util
from models.lora_diffusion.LoRA import save_lora_weight
import pdb
CustomResult = collections.namedtuple('CustomResult', 'name result')

class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.opt = opt
        self.phase = opt['phase']
        self.set_device = partial(Util.set_device, rank=opt['global_rank'])

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = 1
        if 'datasets' in self.opt:
            self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = 0
        self.iter = 0 

        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1
            if self.opt['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            # val_log = self.val_step()
            train_log = self.train_step()

            ''' save logged informations into log dict ''' 
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                # pdb.set_trace()
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    def inference(self, gt_image, cond_image, mask, label):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        lora_config = None
        if 'lora_config' in self.opt["model"]["which_model"]["args"]:
            lora_config = self.opt["model"]["which_model"]["args"]["lora_config"]
            if lora_config["flag"] == "ft":
                network_label = 'LoRA'
                
                save_filename = '{}_{}.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
                
                save_lora_weight(model=network.denoise_fn, 
                                 path=save_path,
                                 target_replace_module=lora_config["target_replace_module"],
                                 inject_embedding=lora_config["inject_emb"])

                return
            
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):        
        if self.opt['path']['resume_state'] is None:
            opt_network_args = self.opt['model']['which_networks'][0]['args'] 
            if "use_ldm" not in opt_network_args or not opt_network_args["use_ldm"]:
                return 
            if "use_ldm" in opt_network_args and opt_network_args["use_ldm"]:
                assert self.opt['path']['encoder_resume_state'] is not None, 'We must have a trained autoencoder model when train a ldm.'
            if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
                network = network.module            
            network.first_stage_fn.load_state_dict(torch.load(self.opt['path']['encoder_resume_state'], 
                                                              map_location = lambda storage, 
                                                              loc: self.set_device(storage)), 
                                                   strict=True)
            network.first_stage_fn.eval()
            
            return 
        if self.logger is not None:
            self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self.opt['path']['resume_state'], network_label)

        assert os.path.exists(model_path), 'Pretrained model in [{:s}] is not existed, Skip it'.format(model_path)
        
        if not os.path.exists(model_path):
            if self.logger is not None:
                self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return
        
        if self.logger is not None:
            self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, 
                                           map_location=lambda storage,
                                           loc: self.set_device(storage)), 
                                strict=strict)

        if self.opt['path']['encoder_resume_state'] is not None:
            network.first_stage_fn.load_state_dict(torch.load(self.opt['path']['encoder_resume_state'], 
                                                              map_location = lambda storage, 
                                                              loc: self.set_device(storage)), 
                                                   strict=True)
            network.first_stage_fn.eval()

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase !='train' or self. opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self. opt['path']['resume_state'])
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')