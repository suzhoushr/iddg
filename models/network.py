import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from copy import deepcopy
import pdb
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, 
                 module_name='sr3',
                 sample_type='ddim', 
                 sample_timesteps=300, 
                 **kwargs):
        super(Network, self).__init__(**kwargs)
        self.module_name = module_name
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

        self.sample_type = sample_type
        if sample_type == 'ddim':
            assert sample_timesteps > 0, 'ddim sample steps must large than 0.'
            self.ddim_timesteps = sample_timesteps
        elif sample_type == 'dpmsolver++' or sample_type == 'dpmsolver':
            self.dpm_timesteps = sample_timesteps

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])
        # pdb.set_trace()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

        # if self.sample_type == 'ddim':
        #     assert self.ddim_timesteps <= self.num_timesteps, 'num_timesteps must greater than ddim_timesteps'

        #     c = self.num_timesteps // self.ddim_timesteps
        #     ddim_timesteps_seq = np.asarray(list(range(0, self.num_timesteps, c)))

        #     self.time_seq = ddim_timesteps_seq[1:]
        #     self.time_seq_next = ddim_timesteps_seq[0:-1]
        if self.sample_type == 'ddim':
            assert self.ddim_timesteps < self.num_timesteps, 'num_timesteps must greater than ddim_timesteps'

            c = self.num_timesteps // self.ddim_timesteps
            ddim_timesteps_seq = np.asarray(list(range(0, self.num_timesteps, c)))

            self.time_seq = ddim_timesteps_seq[1:]
            self.time_seq_next = ddim_timesteps_seq[0:-1]

            from utils.sampler.ddim_solver import DDIM_Solver
            self.sampler = DDIM_Solver(model_fn=self.denoise_fn, 
                                       alphas_cumprod=self.gammas,
                                       schedule='uniform', 
                                       num_timesteps=self.num_timesteps,
                                       module_name=self.module_name)
        elif self.sample_type == 'dpmsolver' or self.sample_type == 'dpmsolver++':
            assert self.dpm_timesteps < self.num_timesteps, 'num_timesteps must greater than dpm_timesteps'
            from utils.sampler.dpm_solver import DPMSolverSampler
            self.algorithm_type = self.sample_type
            self.sampler = DPMSolverSampler(model=self.denoise_fn, 
                                            alphas_cumprod=self.gammas,
                                            device=device)

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, label=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, label))  # TODO

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, label=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, label=label)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, 
                    y_t=None, 
                    y_0=None, 
                    mask=None, 
                    label=None, 
                    sample_num=8, 
                    eta=0.0, 
                    ratio=1.0, 
                    gd_w=0.0,
                    **kwargs):
        # pdb.set_trace()
        if gd_w > 0.0:
            label_l = list()
            for i in range(label.shape[0]):
                label_l.append(0)
            for i in range(label.shape[0]):
                label_l.append(label[i])
            label = torch.tensor(label_l).to(label.device)
        if self.sample_type == 'ddim':
            y_t, ret_arr = self.sampler.sample(y_cond=y_cond, y_t=y_t, y_0=y_0, 
                                               mask=mask, label=label, context=None,
                                               sample_num=sample_num, ddim_num_steps=self.ddim_timesteps,
                                               eta=eta, ratio=ratio, gd_w=gd_w)          
            return y_t, ret_arr
        
        if self.sample_type == 'dpmsolver' or self.sample_type == 'dpmsolver++':
            y_t, ret_arr = self.sampler.sample(S=self.dpm_timesteps, x_cond=y_cond, x_t=y_t, x_0=y_0, 
                                               mask=mask, label=label, context=None,
                                               sample_num=sample_num, ratio=ratio, gd_w=gd_w, module='gd')
            return y_t, ret_arr
        
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond, label=label)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr
    
    @torch.no_grad()
    def restoration_ddim(self, y_cond, y_t=None, y_0=None, mask=None, label=None, ddim_timesteps=500, eta=0.0, ratio=1.0, sample_num=8):
        '''
        eta controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
        ddim_timesteps controls how many timesteps used in the process. Its value must less than self.num_timesteps
        '''
        # skip = max(0, self.num_timesteps // ddim_timesteps)
        # # pdb.set_trace()
        # seq = range(0, self.num_timesteps, skip)[1:]
        # seq_next = range(0, self.num_timesteps, skip)[:-1]

        sample_inter = (len(self.time_seq) // sample_num)
        y_t = default(y_t, lambda: torch.randn_like(y_cond))

        if ratio < 1.0:
            num_timesteps = int(self.num_timesteps * ratio)
            assert self.ddim_timesteps <= num_timesteps, 'num_timesteps must greater than ddim_timesteps'

            c = num_timesteps // self.ddim_timesteps
            ddim_timesteps_seq = np.asarray(list(range(0, num_timesteps, c)))

            self.time_seq = ddim_timesteps_seq[1:]
            self.time_seq_next = ddim_timesteps_seq[0:-1]
  
            t = (torch.ones(y_t.size(0)) * self.time_seq[-1]).to(y_t.device).long()
            at = extract(self.gammas, t, x_shape=y_0.shape)
            y_t = self.q_sample(y_0=y_0, sample_gammas=at)
            if mask is not None:
                y_t = y_t * mask + y_0 * (1 - mask)

        ret_arr = y_t
        n = y_t.size(0)
        for i, j in tqdm(zip(reversed(self.time_seq), reversed(self.time_seq_next)), desc='    sampling loop time step', total=len(self.time_seq)):
            t = (torch.ones(n) * i).to(y_t.device).long()
            next_t = (torch.ones(n) * j).to(y_t.device).long()
            at = extract(self.gammas, t, x_shape=(1, 1))
            at_next = extract(self.gammas, next_t, x_shape=(1, 1))

            et = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), at, label)

            at = at.unsqueeze(-1).unsqueeze(-1)
            at_next = at_next.unsqueeze(-1).unsqueeze(-1)
            y0_t = (y_t - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            y_t = at_next.sqrt() * y0_t + c1 * torch.randn_like(y_t) + c2 * et
            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            if i == self.time_seq[0]:
                ret_arr[-y_t.shape[0]:, :, :, :] =  y_t

        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None, label=None):
        # pdb.set_trace()
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas, label)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas, label)
            loss = self.loss_fn(noise, noise_hat)
        return loss


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


