from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import core.util as Util
from core.praser import init_obj
import pdb

def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    if opt['phase'] == 'train':    
        dataset_opt['args']['validation_split'] = opt['datasets'][opt['phase']]['dataloader']['validation_split']    
        dataset_opt['args']['phase'] = 'train'        
        phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')

        dataset_opt['args']['phase'] = 'val'
        val_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    elif opt['phase'] == 'test':
        dataset_opt['args']['phase'] = 'test'
        phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
        val_dataset = None

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        # dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        dataloader_args = opt['datasets'][opt['phase']]['dataloader']['val_args']
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    # pdb.set_trace()
    return dataloader, val_dataloader
