import numpy as np
import torch
from tqdm import tqdm
from models.ddpm import default, extract
import math
import copy
import pdb

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=False):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    if verbose:
        print(f'Selected timesteps for ddim sampler: {ddim_timesteps}')
    return ddim_timesteps

class DDIM_Solver:
    def __init__(
        self,
        model_fn,
        alphas_cumprod,
        schedule="uniform",
        num_timesteps=1000,
        module_name='sd_v15'
    ):
        self.denoise_fn = model_fn
        self.schedule = schedule
        self.ddpm_num_timesteps = num_timesteps
        assert len(alphas_cumprod) == num_timesteps, "the alphas' lenght is mismatch to num_timesteps"
        self.alphas_cumprod = alphas_cumprod
        self.module_name = module_name
        
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ratio=1.0, verbose=False):
        assert ddim_num_steps < self.ddpm_num_timesteps, 'num_timesteps must greater than ddim_num_steps'
        total_num_timesteps = self.ddpm_num_timesteps
        if ratio < 1.0:
            total_num_timesteps = int(self.ddpm_num_timesteps * ratio)
        assert ddim_num_steps < total_num_timesteps, 'num_timesteps must greater than ddim_num_steps'

        ddim_timesteps_seq = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=total_num_timesteps, verbose=verbose)
        self.time_seq = ddim_timesteps_seq[1:]
        self.time_seq_prev = ddim_timesteps_seq[0:-1]

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def sample(self, 
               y_t,
               y_hint=None,
               y_cond=None, 
               y_0=None, 
               mask=None, 
               label=None, 
               context=None,
               sample_num=8, 
               ddim_num_steps=300, 
               eta=0.0, 
               ratio=1.0, 
               gd_w=0.0):
        '''
        eta controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
        ddim_timesteps controls how many timesteps used in the process. Its value must less than self.num_timesteps
        '''
        self.make_schedule(ddim_num_steps=ddim_num_steps, ddim_discretize=self.schedule, ratio=ratio)
        if sample_num > 0:
            sample_inter = (len(self.time_seq) // sample_num)
        else: 
            sample_inter = len(self.time_seq) + 1

        n = y_0.size(0)

        if y_hint is None:
            if mask is not None and y_0 is not None:
                y_t = y_t * mask + y_0 * (1 - mask)
        ret_arr = y_t
        
        if gd_w > 0.0:
            if label is not None:
                label_l = list()
                for i in range(label.shape[0]):
                    label_l.append(0)
                for i in range(label.shape[0]):
                    label_l.append(label[i])
                label = torch.tensor(label_l).to(label.device)

            ## TODO, text
            if context is not None:
                non_context = torch.zeros_like(context)
                context = torch.cat([context, non_context], 0)

        if gd_w > 0.0:
            y_t = torch.cat([y_t, y_t], 0)
            if y_cond is not None:
                y_cond = torch.cat([y_cond, y_cond], 0)
            if y_hint is not None:
                y_hint = torch.cat([y_hint, y_hint], 0)
            n = 2 * n

        icount = 0
        for i, j in zip(reversed(self.time_seq), reversed(self.time_seq_prev)):
            t = (torch.ones(n) * i).to(y_t.device).long()
            t_prev = (torch.ones(n) * j).to(y_t.device).long()
            at = extract(self.alphas_cumprod, t, x_shape=(1, 1))
            at_prev = extract(self.alphas_cumprod, t_prev, x_shape=(1, 1))

            if y_hint is None:
                if y_cond is not None:
                    y_t_con = torch.cat([y_t, y_cond], dim=1)
                
                if self.module_name == 'sd_v15':
                    et = self.denoise_fn(y_t_con, t, y=label, context=context)
            else:
                if self.module_name == 'sd_v15':
                    et = self.denoise_fn(y_t, y_hint, t, y=label, context=context)

            if gd_w > 0.0:
                et_uc = et[0:n//2, :, :, :]
                et_c = et[n//2:, :, :, :]
                et = et_c + gd_w * (et_c - et_uc)
                y_t = y_t[n//2:, :, :, :]
                at = at[n//2:]
                at_prev = at_prev[n//2:]
            
            at = at.unsqueeze(-1).unsqueeze(-1)
            at_prev = at_prev.unsqueeze(-1).unsqueeze(-1)
            y0_hat = (y_t - et * (1 - at).sqrt()) / at.sqrt()
            if eta > 0:
                c1 = (
                    eta * ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_prev) - c1 ** 2).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c1 * torch.randn_like(y_t) + c2 * et
            else:
                c2 = (1 - at_prev).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c2 * et

            if mask is not None and y_hint is None and i != self.time_seq[0]:
                y_t = y_t * mask + y_0 * (1. - mask)   

            icount += 1
            if icount % sample_inter == 0 and (ret_arr.shape[0] // y_t.shape[0] - 1 <= sample_num):
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            if i == self.time_seq[0]:
                ret_arr[-y_t.shape[0]:, :, :, :] =  y_t

            if gd_w > 0.0 and i > self.time_seq[0]:
                y_t = torch.cat([y_t, y_t], 0)

        return y_t, ret_arr
    
    @torch.no_grad()
    def sample_v2(self, 
               y_t,
               y_cond=None, 
               y_0=None, 
               mask=None, 
               label=None, 
               context=None,
               sample_num=8, 
               ddim_num_steps=300, 
               eta=0.0, 
               ratio=1.0, 
               gd_w=0.0):
        '''
        eta controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
        ddim_timesteps controls how many timesteps used in the process. Its value must less than self.num_timesteps
        '''
        self.make_schedule(ddim_num_steps=ddim_num_steps, ddim_discretize=self.schedule, ratio=ratio)
        if sample_num > 0:
            sample_inter = (len(self.time_seq) // sample_num)
        else: 
            sample_inter = len(self.time_seq) + 1

        n = y_0.size(0)

        if mask is not None and y_0 is not None:
            y_t = y_t * mask + y_0 * (1 - mask)
        ret_arr = y_t
        
        if gd_w > 0.0:
            if label is not None:
                label_l = list()
                for i in range(label.shape[0]):
                    label_l.append(0)
                for i in range(label.shape[0]):
                    label_l.append(label[i])
                label = torch.tensor(label_l).to(label.device)

            ## TODO, text

        if gd_w > 0.0:
            y_t = torch.cat([y_t, y_t], 0)
            if y_cond is not None:
                y_cond = torch.cat([y_cond, y_cond], 0)
            n = 2 * n

        icount = 0
        for i, j in zip(reversed(self.time_seq), reversed(self.time_seq_prev)):
            t = (torch.ones(n) * i).to(y_t.device).long()
            t_prev = (torch.ones(n) * j).to(y_t.device).long()
            at = extract(self.alphas_cumprod, t, x_shape=(1, 1))
            at_prev = extract(self.alphas_cumprod, t_prev, x_shape=(1, 1))

            if y_cond is not None:
                y_t_con = torch.cat([y_t, y_cond], dim=1)
                
            if self.module_name == 'sd_v15':
                et = self.denoise_fn(y_t_con, t, y=label, context=context)

            if gd_w > 0.0:
                et_uc = et[0:n//2, :, :, :]
                et_c = et[n//2:, :, :, :]
                et = et_c + gd_w * (et_c - et_uc)
                y_t = y_t[n//2:, :, :, :]
                at = at[n//2:]
                at_prev = at_prev[n//2:]
            
            at = at.unsqueeze(-1).unsqueeze(-1)
            at_prev = at_prev.unsqueeze(-1).unsqueeze(-1)
            y0_hat = (y_t - et * (1 - at).sqrt()) / at.sqrt()
            if eta > 0:
                c1 = (
                    eta * ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_prev) - c1 ** 2).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c1 * torch.randn_like(y_t) + c2 * et
            else:
                c2 = (1 - at_prev).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c2 * et

            if mask is not None:
                y_t = y_t * mask + y_0 * (1. - mask)   

            icount += 1
            if icount % sample_inter == 0 and (ret_arr.shape[0] // y_t.shape[0] - 1 <= sample_num):
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            if i == self.time_seq[0]:
                ret_arr[-y_t.shape[0]:, :, :, :] =  y_t

            if gd_w > 0.0 and i > self.time_seq[0]:
                y_t = torch.cat([y_t, y_t], 0)

        return y_t, ret_arr
    
    @torch.no_grad()
    def sample_pro(self, 
               y_t,
               y_cond=None, 
               y_0=None, 
               mask=None, 
               label=None, 
               context=None,
               sample_num=8, 
               ddim_num_steps=300, 
               inference_step=100,
               eta=0.0, 
               ratio=1.0, 
               gd_w=0.0,
               flag='invert'):
        '''
        eta controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
        ddim_timesteps controls how many timesteps used in the process. Its value must less than self.num_timesteps
        flag: invert | sample
        '''
        self.make_schedule(ddim_num_steps=ddim_num_steps, ddim_discretize=self.schedule, ratio=ratio)
        if sample_num > 0:
            sample_inter = (len(self.time_seq) // sample_num)
        else: 
            sample_inter = len(self.time_seq) + 1

        n = y_0.size(0)

        if mask is not None and y_0 is not None:
            y_t = y_t * mask + y_0 * (1 - mask)
        ret_arr = y_t
        
        if gd_w > 0.0:
            if label is not None:
                label_l = list()
                for i in range(label.shape[0]):
                    label_l.append(0)
                for i in range(label.shape[0]):
                    label_l.append(label[i])
                label = torch.tensor(label_l).to(label.device)

            ## TODO, text

        if gd_w > 0.0:
            y_t = torch.cat([y_t, y_t], 0)
            if y_cond is not None:
                y_cond = torch.cat([y_cond, y_cond], 0)
            n = 2 * n

        icount = 0
        if flag == "sample":
            time_seq, time_seq_prev = reversed(self.time_seq[:inference_step]), reversed(self.time_seq_prev[:inference_step])
        elif flag == 'invert':
            time_seq, time_seq_prev = self.time_seq_prev[:inference_step], self.time_seq[:inference_step]
        for i, j in zip(time_seq, time_seq_prev):
            t = (torch.ones(n) * i).to(y_t.device).long()
            t_prev = (torch.ones(n) * j).to(y_t.device).long()
            at = extract(self.alphas_cumprod, t, x_shape=(1, 1))
            at_prev = extract(self.alphas_cumprod, t_prev, x_shape=(1, 1))

            if y_cond is not None:
                y_t_con = torch.cat([y_t, y_cond], dim=1)
                
            if self.module_name == 'sd_v15':
                et = self.denoise_fn(y_t_con, t, y=label, context=context)

            if gd_w > 0.0:
                et_uc = et[0:n//2, :, :, :]
                et_c = et[n//2:, :, :, :]
                et = et_c + gd_w * (et_c - et_uc)
                y_t = y_t[n//2:, :, :, :]
                at = at[n//2:]
                at_prev = at_prev[n//2:]
            
            at = at.unsqueeze(-1).unsqueeze(-1)
            at_prev = at_prev.unsqueeze(-1).unsqueeze(-1)
            y0_hat = (y_t - et * (1 - at).sqrt()) / at.sqrt()
            if eta > 0:
                c1 = (
                    eta * ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_prev) - c1 ** 2).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c1 * torch.randn_like(y_t) + c2 * et
            else:
                c2 = (1 - at_prev).sqrt()
                y_t = at_prev.sqrt() * y0_hat + c2 * et

            if mask is not None:
                y_t = y_t * mask + y_0 * (1. - mask)   

            icount += 1
            if icount % sample_inter == 0 and (ret_arr.shape[0] // y_t.shape[0] - 1 <= sample_num):
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            if i == self.time_seq[0]:
                ret_arr[-y_t.shape[0]:, :, :, :] =  y_t

            if gd_w > 0.0 and i > self.time_seq[0]:
                y_t = torch.cat([y_t, y_t], 0)

        return y_t, ret_arr
    
    def step(self, y_t, time, et, ddim_num_steps=300, eta=0.0):
        '''
        predict the sample of the next step in the denoise process
        '''
        assert time < ddim_num_steps, "current time must less than ddim steps."
        assert time > 0, "current time must larger than 0."
        self.make_schedule(ddim_num_steps=ddim_num_steps, ddim_discretize=self.schedule, ratio=1.0)        

        n = y_t.shape[0]
        t = (torch.ones(n) * self.time_seq[time-1]).to(y_t.device).long()
        t_prev = (torch.ones(n) * self.time_seq_prev[time-1]).to(y_t.device).long()
        at = extract(self.alphas_cumprod, t, x_shape=(1, 1))
        at_prev = extract(self.alphas_cumprod, t_prev, x_shape=(1, 1))

        y0_hat = (y_t - et * (1 - at).sqrt()) / at.sqrt()
        if eta > 0:
            c1 = (
                eta * ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_prev) - c1 ** 2).sqrt()
            y_t = at_prev.sqrt() * y0_hat + c1 * torch.randn_like(y_t) + c2 * et
        else:
            c2 = (1 - at_prev).sqrt()
            y_t = at_prev.sqrt() * y0_hat + c2 * et

        return y_t
    
    def get_time(self, time, ddim_num_steps=300):
        assert time < ddim_num_steps, "current time must less than ddim steps."
        assert time > 0, "current time must larger than 0."
        self.make_schedule(ddim_num_steps=ddim_num_steps, ddim_discretize=self.schedule, ratio=1.0)   
        return self.time_seq[time-1]     
    