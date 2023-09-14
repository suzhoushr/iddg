import os
import torch
import sys
import pdb

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        # pdb.set_trace()
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)   
        print("load model %s." %(save_path))     
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path), strict=True)
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)
                if self.opt.verbose:
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                                
    def update_learning_rate():
        pass
