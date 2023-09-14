import sys
sys.path.append("")

import os
import numpy as np
import json
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from tools.engine import Engine
from models.lora_diffusion.LoRA import inject_trainable_lora, init_lora_ups_down
from models.ddpm import extract
from utils.img_proc.image_process import shape_to_mask, crop_image, points2mask
import core.util as Util
import copy

import pdb

class DragDiff(Engine):
    def __init__(self, opt, img_sz=256):
        super(DragDiff, self).__init__(opt, img_sz=img_sz)
        
        lora_config = self.opt['model']['which_networks'][0].get('lora_config', {})
        self.lora_config = lora_config
        
        if lora_config["flag"] != "not_use": 
            ## it is very strange that when I set the requires_grad=True, it will throw the error:            
            # self.model.netG.denoise_fn.requires_grad_(False)
            if self.model.netG.cond_fn is not None:
                self.model.netG.cond_fn.requires_grad_(False)
            
            if lora_config["flag"] == "infer":
                assert lora_config["loras"] is not None, "In infer stage, we must have a lora model."
            unet_lora_params, _ = inject_trainable_lora(model=self.model.netG.denoise_fn,
                                                            loras=lora_config["loras"],
                                                            r=lora_config["lora_rank"],
                                                            target_replace_module=lora_config["target_replace_module"],
                                                            dropout_p=lora_config["lora_dropout_p"],
                                                            scale=lora_config["lora_scale"])

        if lora_config["flag"] == "ft":   
            opt_params = list(unet_lora_params[0])
            for i in range(1, len(unet_lora_params)):
                opt_params += list(unet_lora_params[i])
            self.lora_optG = torch.optim.AdamW(opt_params, lr=lora_config['lora_lr'])
            self.lora_loss_fn = torch.nn.MSELoss()

    def set_params(self,
                   n_actual_inference_step=100,
                   n_pix_step=40,
                   lam=0.1,
                   drag_lr=0.01,
                   r_m=1,
                   r_p=3):
        
        self.n_actual_inference_step = n_actual_inference_step
        self.n_pix_step = n_pix_step
        self.lam = lam
        self.drag_lr = drag_lr
        self.r_m = r_m
        self.r_p = r_p

        t = self.model.netG.sampler.get_time(time=n_actual_inference_step, ddim_num_steps=self.model.netG.ddim_timesteps)
        self.t = (torch.ones(1) * t).to(self.device).long()

    def set_points(self,
                   handle_points,
                   target_points):
        assert len(handle_points) == len(target_points), "number of handle point must equals target points"

        self.handle_points = torch.tensor(copy.deepcopy(handle_points), dtype=torch.float)
        self.handle_points_init = torch.tensor(copy.deepcopy(handle_points), dtype=torch.float)
        self.target_points = torch.tensor(target_points, dtype=torch.float)

    def labelme2inputs(self, imp, show_img=False):
        '''
        in labelme, the point is (x, y), type is list
        but in DragDiff, the point is (y, x), type is tensor
        '''
        jsp = imp.replace('.jpg', '.json')
        cv_img = cv2.imread(imp)
        with open(jsp, 'r', encoding='utf-8') as jf:
            info = json.load(jf)

        handle_points = list()
        target_points = list()
        d_handle_points = dict()
        d_target_points = dict()
        all_points = list()
        for shape in info['shapes']:
            if 'src' in shape['label']:
                d_handle_points[shape['label']] = shape['points'][0]
                all_points.append(shape['points'][0])
            elif 'dst' in shape['label']:
                d_target_points[shape['label']] = shape['points'][0]
                all_points.append(shape['points'][0])
            else:
                label = shape['label']
                points = shape['points']
                all_points.extend(points)
                shape_type = shape['shape_type']
                mask = points2mask(points, shape_type, label, im_crop_sz=cv_img.shape[0])
                # cond_img = mask
        x_list = [k[0] for k in all_points]
        y_list = [k[1] for k in all_points]
        x1 = int(min(x_list))
        y1 = int(min(y_list))
        x2 = int(max(x_list))
        y2 = int(max(y_list))
        # mask = np.zeros((1, cv_img.shape[0], cv_img.shape[1]))
        mask[:, y1:y2, x1:x2] = 1.0
        # mask[:, 64:192, 64:192] = 1.0
        cond_img = copy.deepcopy(mask)
        handle_points = [[int(p[1][1]), int(p[1][0])] for p in sorted(d_handle_points.items(), key=lambda d: d[0], reverse=False)]
        target_points = [[int(p[1][1]), int(p[1][0])] for p in sorted(d_target_points.items(), key=lambda d: d[0], reverse=False)]
        # pdb.set_trace()
        
        if show_img:
            alpha = mask.transpose(1, 2, 0)
            show_img = cv_img * alpha + cv_img * (1 - alpha) * 0.8
            show_img = show_img.astype('uint8')
            point_size = 1
            point_color = (0, 0, 255) # BGR
            thickness = 4 # could be 0 、4、8

            for point in handle_points:
                cv2.circle(show_img, (point[1], point[0]), point_size, point_color, thickness)
            point_color = (0, 255, 0) # BGR
            for point in target_points:
                cv2.circle(show_img, (point[1], point[0]), point_size, point_color, thickness)
            cv2.imshow('img', show_img)
            cv2.waitKey(0)

        return cv_img, cond_img, mask, label, handle_points, target_points
                        
    def train_lora(self, show_loss=False):
        '''
        in fact, when train lora, we do not using mask, the task will be generate task, 
        i.e. mask and cond_image is all 1
        it maybe wrong
        '''
        init_lora_ups_down(model=self.model.netG.denoise_fn, 
                           target_replace_module=self.lora_config["target_replace_module"],
                           r=self.lora_config["lora_rank"])
          
        # self.model.netG.denoise_fn.eval()  
        mask_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(self.device)
        cond_image = torch.ones(1, 1, self.im_crop_sz, self.im_crop_sz).to(self.device)     
        for sep in range(self.lora_config['lora_steps']):            
            ## use_ldm
            if self.model.netG.first_stage_fn is not None:
                self.model.netG.first_stage_fn.eval()
                with torch.no_grad():
                    y_0 = self.model.netG.first_stage_fn.encode(self.gt_image).sample()
                    y_0 = y_0.detach()

                scale_factor = 1.0 * y_0.shape[2] / self.mask_image.shape[2]
                mask = F.interpolate(mask_image, scale_factor=scale_factor, mode="bilinear")
                mask = mask.detach()

                y_cond = F.interpolate(cond_image, scale_factor=scale_factor, mode="bilinear")
                y_cond = y_cond.detach()
            else:
                y_0 = self.gt_image
                mask = mask_image.detach()
                y_cond = cond_image.detach()

            # sampling from p(gammas)
            b, *_ = y_0.shape
            t = torch.randint(1, self.model.netG.num_timesteps, (b,), device=y_0.device).long()
            sample_gammas = extract(self.model.netG.gammas, t, x_shape=y_0.shape)

            noise = torch.randn_like(y_0)
            y_noisy = self.model.netG.q_sample(y_0=y_0, 
                                               sample_gammas=sample_gammas.view(-1, 1, 1, 1),
                                               noise=noise)
        
            context = None
            if self.model.netG.cond_fn is not None:
                assert self.text is not None, 'We must have text while in condition guided mode.'
                context = self.model.netG.cond_fn(self.text)

            if mask is not None:
                y_noisy = y_0 * (1.0 - mask) + y_noisy * mask

            if y_cond is not None:
                y_noisy = torch.cat([y_noisy, y_cond], dim=1)
            
            if self.model.netG.module_name == 'sd_v15':
                noise_hat = self.model.netG.denoise_fn(y_noisy, t, y=self.label, context=context)

            if mask is not None:
                loss = self.lora_loss_fn(mask * noise, mask * noise_hat)
            else:
                loss = self.lora_loss_fn(noise, noise_hat)
            if show_loss:
                print("in step " + str(sep) + ", loss is " + str(loss.cpu().item()))
                
            self.lora_optG.zero_grad()
            loss.backward()
            self.lora_optG.step()
            # print("linear is ====> " + str(self.model.netG.denoise_fn.input_blocks[1][1].to_q.linear.weight.sum().item()))
            # print("lora_up is ====> " + str(self.model.netG.denoise_fn.input_blocks[1][1].to_q.lora_up.weight.sum().item()))
            # print("lora_down is ====> " + str(self.model.netG.denoise_fn.input_blocks[1][1].to_q.lora_down.weight.sum().item()))
            # pdb.set_trace()

    def get_latent(self,
                     img, 
                     mask=None, 
                     label=None, 
                     text=None):
        # step 1: set input
        cond_img = mask
        self.set_input(cv_img=img,
                       cond_img=cond_img,
                       mask=mask,
                       label=label,
                       text=text)
        
        # step 2: train LoRA
        if self.lora_config["flag"] != "not_use":
            self.train_lora(show_loss=True)

        # step 3: get invert latent code
        with torch.no_grad():
            self.latent_codes = self.image2latent(gt_image=self.gt_image, 
                                                label=self.label,
                                                inference_step=self.n_actual_inference_step, 
                                                return_invert_sample=False)
        
    def forward_unet_features(self, is_scale=True):
        y_cond = torch.ones(1, 1, self.latent_codes.shape[2], self.latent_codes.shape[3]).to(self.latent_codes.device)
        y_t_con = torch.cat([self.latent_codes, y_cond], dim=1)
        # pdb.set_trace()
        # self.model.netG.denoise_fn.eval()
        et, F0 = self.model.netG.denoise_fn(y_t_con, timesteps=self.t, y=self.label, context=None, return_last2layers=True)
        if is_scale:
            F0 = F.interpolate(F0, (self.im_crop_sz, self.im_crop_sz), mode="bilinear")

        y_prev_0 = self.model.netG.sampler.step(y_t=self.latent_codes, 
                                                time=self.n_actual_inference_step, 
                                                et=et, 
                                                ddim_num_steps=self.model.netG.ddim_timesteps)
        # if self.model.netG.first_stage_fn is not None:
        #     F0 = self.model.netG.first_stage_fn.decode(y_prev_0)
        return F0, y_prev_0
    
    
    def preworks(self, is_scale=True):
        with torch.no_grad():
            self.F0, self.y_prev_0 = self.forward_unet_features(is_scale=is_scale)

        # prepare optimizable init_code and optimizer
        self.latent_codes.requires_grad_(True)
        # self.model.netG.denoise_fn.requires_grad_(False)
        self.drag_optimizer = torch.optim.Adam([self.latent_codes], lr=self.drag_lr)  
        self.drag_train_step = 0


    def point_tracking(self, F0, F1, handle_points, handle_points_init):
        '''
        this part do not need backforward
        '''
        with torch.no_grad():
            for i in range(len(handle_points)):
                pi0, pi = handle_points_init[i], handle_points[i]
                f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

                r1, r2 = int(pi[0]) - self.r_p, int(pi[0]) + self.r_p + 1
                c1, c2 = int(pi[1]) - self.r_p, int(pi[1]) + self.r_p + 1
                F1_neighbor = F1[:, :, r1:r2, c1:c2]
                all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
                all_dist = all_dist.squeeze(dim=0)
                # WARNING: no boundary protection right now
                row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
                handle_points[i][0] = pi[0] - self.r_p + row
                handle_points[i][1] = pi[1] - self.r_p + col
            return handle_points

    def check_handle_reach_target(self, handle_points, target_points):
        # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
        all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
        return (torch.tensor(all_dist) < 1.0).all()

    def interpolate_feature_patch(self, feat, y, x, r):
        """obtain the bilinear interpolated feature patch centered around (x, y) with radius r"""
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())

        Ia = feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        Ib = feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1]
        Ic = feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1]
        Id = feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]

        return Ia * wa + Ib * wb + Ic * wc + Id * wd


    def drag_diffusion_update(self, iters=5, show_loss=False):
        '''
        return a flag: 0 means trainning can contunue
                       1 means trainning is finished due to the n_pix_step
                       2 means trainning is finished due to the handle_points had reached to the target points
        '''
        if self.drag_train_step >= self.n_pix_step:
            return 1
        
        # return True if all handle points have reached the targets
        if self.check_handle_reach_target(self.handle_points, self.target_points):
            return 2

        # interp_mask = F.interpolate(self.mask_image, (self.latent_codes.shape[2], self.latent_codes.shape[3]), mode='nearest')
        interp_mask = F.interpolate(self.mask_image, (self.latent_codes.shape[2], self.latent_codes.shape[3]), mode='bilinear')

        # iters = 5
        for i in range(iters):
            loss = 0.0
            F1, y_prev = self.forward_unet_features()     
            if self.drag_train_step != 0:
                self.handle_points = self.point_tracking(self.F0, F1, self.handle_points, self.handle_points_init)
            # print(self.handle_points) 
            for i in range(len(self.handle_points)):
                pi, ti = self.handle_points[i], self.target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 1:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[:, :, int(pi[0]) - self.r_m:int(pi[0]) + self.r_m + 1, int(pi[1]) - self.r_m:int(pi[1]) + self.r_m + 1].detach()
                f1_patch = self.interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_m)
                loss += ((2 * self.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += self.lam * ((y_prev - self.y_prev_0) * (1.0 - interp_mask)).abs().sum()
            if show_loss:
                print('loss total=%f' % (loss.item()))
            # pdb.set_trace()

            self.drag_optimizer.zero_grad()
            loss.backward()
            self.drag_optimizer.step()
        # pdb.set_trace()
        
        self.drag_train_step += 1

        return 0
    
    # def forward_unet_features_shr(self,):
    #     y_cond = torch.ones(1, 1, self.latent_codes.shape[2], self.latent_codes.shape[3]).to(self.latent_codes.device).detach()
    #     y_t_con = torch.cat([self.latent_codes, y_cond], dim=1)

    #     # self.model.netG.denoise_fn.eval()
    #     et = self.model.netG.denoise_fn(y_t_con, timesteps=self.t, y=self.label, context=None, return_last2layers=False)

    #     y_prev_0 = self.model.netG.sampler.step(y_t=self.latent_codes, 
    #                                             time=self.n_actual_inference_step, 
    #                                             et=et, 
    #                                             ddim_num_steps=self.model.netG.ddim_timesteps)
        
    #     if self.model.netG.first_stage_fn is not None:
    #         ldm_scale_factor = 0.31723
    #         y_prev_0 = 1.0 / ldm_scale_factor * y_prev_0
    #         out_prev = self.model.netG.first_stage_fn.decode(y_prev_0)
    #     else:
    #         out_prev = y_prev_0
    #     return out_prev
    
    # def preworks_shr(self,):
    #     with torch.no_grad():
    #         self.out_prev_0 = self.forward_unet_features_shr()            

    #     # prepare optimizable init_code and optimizer
    #     self.latent_codes.requires_grad_(True)
    #     # self.model.netG.denoise_fn.requires_grad_(False)
    #     # self.model.netG.first_stage_fn.requires_grad_(False)
    #     self.drag_optimizer = torch.optim.Adam([self.latent_codes], lr=self.drag_lr)  
    #     self.drag_train_step = 0
    
    # def drag_diffusion_update_shr(self, iters=5, show_loss=False):
    #     '''
    #     return a flag: 0 means trainning can contunue
    #                    1 means trainning is finished due to the n_pix_step
    #                    2 means trainning is finished due to the handle_points had reached to the target points
    #     '''
    #     if self.drag_train_step >= self.n_pix_step:
    #         return 1
        
    #     # return True if all handle points have reached the targets
    #     if self.check_handle_reach_target(self.handle_points, self.target_points):
    #         return 2

    #     for i in range(iters):
    #         loss = 0.0
    #         out_prev = self.forward_unet_features_shr()   
    #         # output = Util.tensor2img(out_prev.cpu().detach())
    #         # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    #         # cv2.imshow('temp_out', output)
    #         # cv2.waitKey(0)
    #         print("latent.sum is =====> " + str(self.latent_codes.sum().item()))
            
    #         if self.drag_train_step != 0:
    #             self.handle_points = self.point_tracking(self.out_prev_0, out_prev, self.handle_points, self.handle_points_init)
    #         print(handle_points)  
    #         # pdb.set_trace()
            
    #         for i in range(len(self.handle_points)):
    #             pi, ti = self.handle_points[i], self.target_points[i]
    #             # skip if the distance between target and source is less than 1
    #             if (ti - pi).norm() < 1:
    #                 continue

    #             di = (ti - pi) / (ti - pi).norm()

    #             # motion supervision
    #             f0_patch = out_prev[:, :, int(pi[0]) - self.r_m:int(pi[0]) + self.r_m + 1, int(pi[1]) - self.r_m:int(pi[1]) + self.r_m + 1].detach()
    #             f1_patch = self.interpolate_feature_patch(out_prev, pi[0] + di[0], pi[1] + di[1], self.r_m)
    #             loss += ((2 * self.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

    #         # masked region must stay unchanged
    #         loss += self.lam * ((out_prev - self.out_prev_0) * (1.0 - self.mask_image)).abs().sum()
    #         if show_loss:
    #             print('loss total=%f' % (loss.item()))

    #         self.drag_optimizer.zero_grad()
    #         loss.backward()
    #         self.drag_optimizer.step()
        
    #     self.drag_train_step += 1

    #     return 0

    def pipeline(self, 
                 img, 
                 handle_points,
                 target_points,
                 mask=None, 
                 label=None, 
                 text=None, 
                 n_actual_inference_step=50,
                 n_pix_step=40,
                 lam=0.1,
                 drag_lr=0.01,
                 r_m=1,
                 r_p=3):
        
        # step 1: setup drag params
        self.set_params(n_actual_inference_step=n_actual_inference_step,
                        n_pix_step=n_pix_step,
                        lam=lam,
                        drag_lr=drag_lr,
                        r_m=r_m,
                        r_p=r_p)
        
        # step 2-4: set input, train LoRA, get invert latent code
        self.get_latent(img=img, 
                        mask=mask, 
                        label=label, 
                        text=text)
        
        # step 5: setup drag points
        self.set_points(handle_points=handle_points,
                        target_points=target_points)
        
        # step 6: get gt and preworks
        # self.preworks(is_scale=True)
        self.preworks()
        
        # step 7: update latent step by step
        img_list = []
        output = img.copy()
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 0 、4、8
        for point in self.handle_points:
            cv2.circle(output, (int(point[1]), int(point[0])), point_size, point_color, thickness)
        point_color = (0, 255, 0) # BGR
        for point in self.target_points:
            cv2.circle(output, (int(point[1]), int(point[0])), point_size, point_color, thickness)
        img_list.append(output)
        
        iters = 1
        while (True):
            flag = self.drag_diffusion_update(iters=iters, show_loss=True)
            # iters = 1
            if flag > 0:
                height, width, _ = img_list[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video = cv2.VideoWriter('./test/video_name.mp4', fourcc, 10, (width, height))

                for image in img_list:
                    video.write(image)
                video.release()

                break

            output = self.latent2image(latent=self.latent_codes,
                                       label=self.label,
                                       inference_step=n_actual_inference_step)
            mask = self.mask_image.squeeze(0).squeeze(0).unsqueeze(2).cpu().numpy()
            output = img * (1 - mask) + output * mask
            output = output.astype('uint8')

            point_size = 1
            point_color = (0, 0, 255) # BGR
            thickness = 4 # 0 、4、8
            for point in self.handle_points:
                cv2.circle(output, (int(point[1]), int(point[0])), point_size, point_color, thickness)
            point_color = (0, 255, 0) # BGR
            for point in self.target_points:
                cv2.circle(output, (int(point[1]), int(point[0])), point_size, point_color, thickness)
            img_list.append(output)
            im_res = np.concatenate([img_list[0], img_list[-1]], 1)
            cv2.imshow('temp', im_res)
            cv2.waitKey(50)
            # pdb.set_trace()

        # step 8: latent to image
        output = self.latent2image(latent=self.latent_codes, 
                                   label=self.label,
                                   inference_step=n_actual_inference_step)
        mask = self.mask_image.squeeze(0).squeeze(0).unsqueeze(2).cpu().numpy()
        output = img * (1 - mask) + output * mask
        output = output.astype('uint8')

        return output
    
if __name__ == "__main__":  
    config = './config/drag/drag_inpainting_ldm_ch192.json'
    drag_engine = DragDiff(config, img_sz=256) 

    imp = './test/aokeng_0_damian_0372-0008-01.jpg'
    imp = './test/pengshang_0_cemian_0303-0011-06.jpg'
    # imp = './test/693_16903_cemian_0224-0008-07.jpg'
    # imp = './test/aokeng_7_damian_0315-0015-01.jpg'
    # imp = './test/pengshang_1_cemian_0719-0004-06.jpg'
    # imp = './test/huashang_1_damian_0031-13.jpg'
    cv_img, cond_img, mask, label, handle_points, target_points = drag_engine.labelme2inputs(imp, show_img=False)

    ## default
    drag_engine.pipeline(img=cv_img, 
                         handle_points=handle_points,
                         target_points=target_points,
                         mask=mask, 
                         label=label,
                         n_actual_inference_step=15,
                         n_pix_step=100,
                         lam=0.1,
                         drag_lr=0.01,
                         r_m=1,
                         r_p=2)
    
    ## shr
    # drag_engine.pipeline(img=cv_img, 
    #                      handle_points=handle_points,
    #                      target_points=target_points,
    #                      mask=mask, 
    #                      label=label,
    #                      n_actual_inference_step=20,
    #                      n_pix_step=100,
    #                      lam=0.1,
    #                      drag_lr=0.01,
    #                      r_m=3,
    #                      r_p=2)
