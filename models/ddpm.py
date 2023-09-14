import math
import torch
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from core.base_network import BaseNetwork
from copy import deepcopy
import pdb

class DDPM(BaseNetwork):
    def __init__(self, unet, beta_schedule, 
                 module_name='sd_v15', 
                 sample_type='ddim', 
                 sample_timesteps=300,
                 use_cond=False,
                 use_ldm=False, 
                 ddconfig=None,                 
                 cond=None,
                 **kwargs):
        super(DDPM, self).__init__(**kwargs)
        self.cond_fn = None
        self.first_stage_fn = None
        self.denoise_fn = None
        self.module_name = module_name

        if module_name == 'sd_v15':
            import sys
            sys.path.append('./models/sd_v15_modules')
            from models.sd_v15_modules.unet import UNet

            if use_cond:
                from models.sd_v15_modules.ldm.modules.encoders.modules import ClassEmbedder
                self.cond_fn = ClassEmbedder(embed_dim=cond['context_dim'] ,n_classes=cond['num_classes'])

            if use_ldm:
                from models.sd_v15_modules.autoencoderkl import AutoencoderKL
                self.embed_dim = 4
                self.first_stage_fn = AutoencoderKL(ddconfig=ddconfig, embed_dim=self.embed_dim)
        else:
            raise ValueError("The model is not supported now.")
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

        self.sample_type = sample_type
        if sample_type == 'ddim':
            self.ddim_timesteps = sample_timesteps
        else:
            raise ValueError('Only ddim sampler are supported in current Maliang')

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        # to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas = torch.from_numpy(gammas).float().to(device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', gammas)
 
        if self.sample_type == 'ddim':
            assert self.ddim_timesteps < self.num_timesteps, 'num_timesteps must greater than ddim_timesteps'
            from utils.sampler.ddim_solver import DDIM_Solver
            self.sampler = DDIM_Solver(model_fn=self.denoise_fn, 
                                       alphas_cumprod=self.gammas,
                                       schedule='uniform', 
                                       num_timesteps=self.num_timesteps,
                                       module_name=self.module_name)

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        y_t = sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise
        return y_t   

    def forward(self, 
                y_0, 
                y_cond=None, 
                mask=None,
                inst=None,                 
                label=None, 
                text=None,
                noise=None,
                ldm_scale_factor=0.31723):
        
        ## use_ldm
        if self.first_stage_fn is not None:
            self.first_stage_fn.eval()
            ## use batch=1 infer to save GPU-MEM
            with torch.no_grad():
                # y_0 = self.first_stage_fn.encode(y_0).mode()
                y_0_hat = list()
                for i in range(y_0.shape[0]):
                    y_0_i = y_0[i, :, :, :].unsqueeze(0)
                    y_0_i = self.first_stage_fn.encode(y_0_i).sample()
                    y_0_hat.append(y_0_i)
                y_0 = torch.cat(y_0_hat, 0)
                y_0 = ldm_scale_factor * y_0
                y_0 = y_0.detach()

            if mask is not None:
                scale_factor = 1.0 * y_0.shape[2] / mask.shape[2]
                mask = F.interpolate(mask, scale_factor=scale_factor, mode="bilinear")
                mask = mask.detach()

            if y_cond is not None:
                scale_factor = 1.0 * y_0.shape[2] / y_cond.shape[2]
                y_cond = F.interpolate(y_cond, scale_factor=scale_factor, mode="bilinear")
                y_cond = y_cond.detach()

        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        sample_gammas = extract(self.gammas, t, x_shape=y_0.shape)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(y_0=y_0, 
                                sample_gammas=sample_gammas.view(-1, 1, 1, 1),
                                noise=noise)
    
        context = None
        if self.cond_fn is not None:
            assert text is not None, 'We must have text while in condition guided mode.'
            context = self.cond_fn(text)

        if mask is not None:
            y_noisy = y_0 * (1.0 - mask) + y_noisy * mask

        if y_cond is not None:
            y_noisy = torch.cat([y_noisy, y_cond], dim=1)
        
        if self.module_name == 'sd_v15':
            noise_hat = self.denoise_fn(y_noisy, t, y=label, context=context)

        if mask is not None:
            loss = 1.0 * self.loss_fn(mask * noise, mask * noise_hat)
        else:
            loss = self.loss_fn(noise, noise_hat)
        return loss
    
    @torch.no_grad()
    def inpaint_restoration(self, 
                            y_0, 
                            y_cond,                             
                            mask,
                            label=None,
                            text=None,
                            sample_num=8, 
                            eta=0.0, 
                            ratio=1.0, 
                            gd_w=0.0,
                            ldm_scale_factor=0.31723):
        
        ## use_ldm
        if self.first_stage_fn is not None:
            y_gt = deepcopy(y_0)
            mask_gt = deepcopy(mask)
            self.first_stage_fn.eval()
            with torch.no_grad():
                y_0 = self.first_stage_fn.encode(y_0).mode()
                y_0 = ldm_scale_factor * y_0
                y_0 = y_0.detach()

            scale_factor = 1.0 * y_0.shape[2] / mask.shape[2]
            mask = F.interpolate(mask, scale_factor=scale_factor, mode="bilinear")
            mask = mask.detach()

            y_cond = F.interpolate(y_cond, scale_factor=scale_factor, mode="bilinear")
            y_cond = y_cond.detach()

        context = None
        if self.cond_fn is not None:            
            context = self.cond_fn(text)

        noise = torch.randn_like(y_0)
        if ratio < 1.0:
            t = int(self.num_timesteps * ratio)
            t = (torch.ones(y_0.shape[0]) * t).to(y_0.device).long()
            at = extract(self.gammas, t, x_shape=y_0.shape)            
            y_t = self.q_sample(y_0=y_0, sample_gammas=at, noise=noise)            
        else:
            y_t = noise   

        if self.sample_type == 'ddim': 
            y_t, ret_arr = self.sampler.sample(y_t=y_t,
                                               y_cond=y_cond, 
                                               y_0=y_0, 
                                               mask=mask, 
                                               label=label, 
                                               context=context,
                                               sample_num=sample_num, 
                                               ddim_num_steps=self.ddim_timesteps, 
                                               eta=eta, 
                                               ratio=ratio, 
                                               gd_w=gd_w)

        if self.first_stage_fn is not None:
            with torch.no_grad():
                y_t = 1.0 / ldm_scale_factor * y_t
                y_t = self.first_stage_fn.decode(y_t)
            if mask is not None:
                y_t = y_gt * (1 - mask_gt) + y_t * mask_gt

            ret_arr_l = list()
            delta = ret_arr.shape[0] // y_t.shape[0]
            for i in range(delta):  
                z_t = ret_arr[i*y_t.shape[0]:i*y_t.shape[0]+y_t.shape[0], :, :, :]      
                with torch.no_grad():
                    z_t = 1.0 / ldm_scale_factor * z_t
                    img = self.first_stage_fn.decode(z_t)
                if mask is not None:
                    img = y_gt * (1 - mask_gt) + img * mask_gt
                ret_arr_l.append(img)
            ret_arr = torch.cat(ret_arr_l, 0)
    
        return y_t, ret_arr
    
    @torch.no_grad()
    def generate_restoration(self, 
                            label=None,
                            text=None,
                            sample_num=8,
                            eta=0.0,
                            gd_w=0.0,
                            im_sz=256,
                            seed=None,
                            ldm_scale_factor=0.31723,
                            device='cuda:0'):
        ratio = 1.0
            
        y_0 = torch.randn(1, 3, im_sz, im_sz).to(device)
        mask = torch.ones(1, 1, im_sz, im_sz).to(device)
        y_cond = torch.ones(1, 1, im_sz, im_sz).to(device)
        
        ## use_ldm
        if self.first_stage_fn is not None:
             y_0 = torch.randn(1, 4, im_sz//8, im_sz//8).to(device)
             mask = torch.ones(1, 1, im_sz//8, im_sz//8).to(device)
             y_cond = torch.ones(1, 1, im_sz//8, im_sz//8).to(device)

        context = None
        if self.cond_fn is not None:       
            assert text is not None, 'oops, you forgot to set text'     
            context = self.cond_fn(text)
            
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn_like(y_0)
        y_t = noise   

        if self.sample_type == 'ddim': 
            y_t, ret_arr = self.sampler.sample(y_t=y_t,
                                               y_cond=y_cond, 
                                               y_0=y_0, 
                                               mask=mask, 
                                               label=label, 
                                               context=context,
                                               sample_num=sample_num, 
                                               ddim_num_steps=self.ddim_timesteps, 
                                               eta=eta, 
                                               ratio=ratio, 
                                               gd_w=gd_w)

        if self.first_stage_fn is not None:
            with torch.no_grad():
                y_t = 1.0 / ldm_scale_factor * y_t
                y_t = self.first_stage_fn.decode(y_t)

            ret_arr_l = list()
            delta = ret_arr.shape[0] // y_t.shape[0]
            for i in range(delta):  
                z_t = ret_arr[i*y_t.shape[0]:i*y_t.shape[0]+y_t.shape[0], :, :, :]      
                with torch.no_grad():
                    z_t = 1.0 / ldm_scale_factor * z_t
                    img = self.first_stage_fn.decode(z_t)
                ret_arr_l.append(img)
            ret_arr = torch.cat(ret_arr_l, 0)
        
        return y_t, ret_arr
    
# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return deepcopy(val)
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


