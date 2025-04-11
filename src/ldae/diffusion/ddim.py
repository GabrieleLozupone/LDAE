# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------


import torch
import numpy as np

from functools import partial
from tqdm import tqdm

"""
Originally ported from here: https://github.com/ckczzj/PDAE/tree/master and adapted for the LDAE framework.
"""

class DDIM:
    """
    Denoising Diffusion Implicit Models (DDIM) implementation for the Latent Diffusion Autoencoders (LDAE) framework.
    
    DDIM enables deterministic sampling and encoding in the diffusion process, which is crucial for:
    1. Efficient sampling in the compressed latent space
    2. Encoding real brain MRI scans into the latent space
    3. Semantic manipulation of brain scans (e.g., simulating disease progression)
    4. Interpolation between different brain scans
    
    This implementation includes specialized methods for the LDAE framework, such as
    shifted sampling/encoding with gradient information from the semantic encoder.
    """
    def __init__(self, betas, timestep_map, device):
        """
        Initialize the DDIM sampler with diffusion parameters.
        
        Args:
            betas (numpy.ndarray): Noise schedule for the diffusion process.
            timestep_map (torch.Tensor): Mapping between original timesteps and potentially accelerated sampling steps.
            device (torch.device): The device (CPU/GPU) to perform computations on.
        """
        super().__init__()
        self.device = device
        self.timestep_map = timestep_map.to(self.device)
        self.timesteps = betas.shape[0] - 1

        # length = timesteps + 1
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 1. will never be used
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)  # 0. will never be used

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))

        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        """
        Extract time-dependent coefficients from a schedule at timestep t and reshape for broadcasting.
        
        Args:
            schedule (torch.Tensor): The schedule containing coefficients for all timesteps.
            t (torch.Tensor): The timestep(s) to extract coefficients for.
            x_shape (tuple): The shape of the tensor to broadcast the coefficients to.
            
        Returns:
            torch.Tensor: Coefficients at timestep t, reshaped for broadcasting.
        """
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    def t_transform(self, t):
        """
        Transform original timesteps to potentially accelerated sampling steps.
        
        Used to map between the original diffusion timesteps and a potentially reduced
        set of timesteps for faster sampling.
        
        Args:
            t (torch.Tensor): Original timesteps.
            
        Returns:
            torch.Tensor: Transformed timesteps.
        """
        new_t = self.timestep_map[t]
        return new_t

    def ddim_sample(self, denoise_fn, x_t, t, condition=None):
        """
        Perform a single step of DDIM sampling (deterministic sampling).
        
        This function implements one step of the DDIM reverse process, taking a noisy latent
        at timestep t and producing a less noisy latent at timestep t-1.
        
        Args:
            denoise_fn (callable): The denoising network (UNet) that predicts noise.
            x_t (torch.Tensor): The latent at timestep t.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor, optional): Conditional information (e.g., semantic code) to guide sampling.
            
        Returns:
            torch.Tensor: The latent at timestep t-1.
        """
        shape = x_t.shape
        predicted_noise = denoise_fn(x_t, self.t_transform(t), condition)
        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

    def ddim_sample_loop(self, denoise_fn, x_T, condition=None):
        """
        Perform complete DDIM sampling from pure noise to a clean sample.
        
        This iteratively applies ddim_sample to reverse the diffusion process from
        pure noise at timestep T to a clean sample at timestep 0.
        
        Args:
            denoise_fn (callable): The denoising network (UNet) that predicts noise.
            x_T (torch.Tensor): Initial noise tensor (at timestep T).
            condition (torch.Tensor, optional): Conditional information to guide sampling.
            
        Returns:
            torch.Tensor: The final generated clean sample at timestep 0.
        """
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.ddim_sample(denoise_fn, img, t, condition)
        return img

    def ddim_encode(self, denoise_fn, x_t, t, condition=None):
        """
        Perform a single step of DDIM encoding.
        
        This enables encoding a clean image into the diffusion process, moving from
        timestep t to timestep t+1. Used for DDIM inversion to project real images
        into the latent space.
        
        Args:
            denoise_fn (callable): The denoising network (UNet) that predicts noise.
            x_t (torch.Tensor): The latent at timestep t.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor, optional): Conditional information for guided encoding.
            
        Returns:
            torch.Tensor: The latent at timestep t+1.
        """
        shape = x_t.shape
        predicted_noise = denoise_fn(x_t, self.t_transform(t), condition)
        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_next = self.extract_coef_at_t(self.alphas_cumprod_next, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * new_predicted_noise

    def ddim_encode_loop(self, denoise_fn, x_0, condition=None, disable_tqdm=False):
        """
        Perform complete DDIM encoding from a clean sample to pure noise.
        
        This function iteratively applies ddim_encode to move from a clean sample at timestep 0
        to a pure noise sample at timestep T. Used for DDIM inversion of real brain MRI scans
        to enable manipulation and interpolation.
        
        Args:
            denoise_fn (callable): The denoising network (UNet) that predicts noise.
            x_0 (torch.Tensor): The clean input sample at timestep 0.
            condition (torch.Tensor, optional): Conditional information for guided encoding.
            disable_tqdm (bool, optional): Whether to disable the progress bar.
            
        Returns:
            torch.Tensor: The encoded noise at timestep T.
        """
        shape = x_0.shape
        batch_size = shape[0]
        x_t = x_0
        for i in tqdm(range(0, self.timesteps), desc='encoding loop time step', total=self.timesteps, disable=disable_tqdm):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.ddim_encode(denoise_fn, x_t, t, condition)
        return x_t

    def shift_ddim_sample(self, decoder, z, x_t, t, use_shift=True):
        """
        Perform a DDIM sampling step with semantic guidance via gradient shifting.
        
        This method uses a gradient estimator to guide the diffusion process toward
        semantically meaningful latents, which is key for the LDAE framework's ability
        to perform semantic manipulation of brain MRIs.
        
        Args:
            decoder (callable): The decoder network that predicts noise and gradient.
            z (torch.Tensor): The semantic code used for conditioning.
            x_t (torch.Tensor): The latent at timestep t.
            t (torch.Tensor): The current timestep.
            use_shift (bool, optional): Whether to apply the gradient shift for semantic guidance.
            
        Returns:
            torch.Tensor: The latent at timestep t-1, semantically guided if use_shift is True.
        """
        shape = x_t.shape
        predicted_noise, gradient = decoder(x_t, self.t_transform(t), z)
        if use_shift:
            coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = predicted_noise - coef * gradient

        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

    def shift_ddim_sample_loop(self, decoder, z, x_T, stop_percent=0.0, disable_tqdm=False):
        """
        Perform complete DDIM sampling with semantic guidance from the gradient estimator.
        
        This function enables the generation of brain MRIs conditioned on a semantic code z,
        with guidance from the gradient estimator for semantic control. The stop_percent
        parameter allows for controlling when to stop applying the semantic guidance.
        
        Args:
            decoder (callable): The decoder network that predicts noise and gradient.
            z (torch.Tensor): The semantic code used for conditioning.
            x_T (torch.Tensor): Initial noise tensor (at timestep T).
            stop_percent (float, optional): Percentage of timesteps at which to stop applying semantic guidance.
            disable_tqdm (bool, optional): Whether to disable the progress bar.
            
        Returns:
            torch.Tensor: The generated brain MRI scan conditioned on the semantic code z.
        """
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T

        stop_step = int(stop_percent * self.timesteps)

        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps, disable=disable_tqdm):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.shift_ddim_sample(decoder, z, img, t, use_shift=True if (i - 1) >= stop_step else False)
        return img

    def shift_ddim_encode(self, decoder, z, x_t, t):
        """
        Perform a single step of DDIM encoding with semantic guidance.
        
        This method enables encoding a real brain MRI scan into the diffusion process
        with guidance from the semantic encoder, moving from timestep t to t+1.
        
        Args:
            decoder (callable): The decoder network that predicts noise and gradient.
            z (torch.Tensor): The semantic code used for conditioning.
            x_t (torch.Tensor): The latent at timestep t.
            t (torch.Tensor): The current timestep.
            
        Returns:
            torch.Tensor: The latent at timestep t+1, with semantic guidance.
        """
        shape = x_t.shape
        predicted_noise, gradient = decoder(x_t, self.t_transform(t), z)
        coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
        predicted_noise = predicted_noise - coef * gradient

        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_next = self.extract_coef_at_t(self.alphas_cumprod_next, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * new_predicted_noise

    def shift_ddim_encode_loop(self, decoder, z, x_0, disable_tqdm=False):
        """
        Perform complete DDIM encoding with semantic guidance.
        
        This function iteratively applies shift_ddim_encode to move from a clean brain MRI scan
        at timestep 0 to a noise sample at timestep T, guided by the semantic code z.
        Used for semantically-guided DDIM inversion of real brain MRI scans.
        
        Args:
            decoder (callable): The decoder network that predicts noise and gradient.
            z (torch.Tensor): The semantic code used for conditioning.
            x_0 (torch.Tensor): The clean input sample at timestep 0.
            disable_tqdm (bool, optional): Whether to disable the progress bar.
            
        Returns:
            torch.Tensor: The encoded noise at timestep T, with semantic guidance.
        """
        shape = x_0.shape
        batch_size = shape[0]
        x_t = x_0
        for i in tqdm(range(0, self.timesteps), desc='encoding loop time step', total=self.timesteps, disable=disable_tqdm):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.shift_ddim_encode(decoder, z, x_t, t)
        return x_t

    def shift_ddim_trajectory_interpolation(self, decoder, z_1, z_2, x_T, alpha):
        """
        Perform interpolation between two semantic codes during the diffusion process.
        
        This enables smooth interpolation between two brain MRI semantics (e.g., different
        disease states or ages) by interpolating between their semantic codes during sampling.
        Key for creating smooth transitions between different brain conditions.
        
        Args:
            decoder (callable): The decoder network that predicts noise and gradient.
            z_1 (torch.Tensor): The first semantic code.
            z_2 (torch.Tensor): The second semantic code.
            x_T (torch.Tensor): Initial noise tensor (at timestep T).
            alpha (float): Interpolation coefficient between z_1 and z_2 (0.0 = z_1, 1.0 = z_2).
            
        Returns:
            torch.Tensor: A brain MRI scan representing the interpolation between z_1 and z_2.
        """
        shape = x_T.shape
        batch_size = shape[0]
        x_t = x_T

        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            predicted_noise, gradient_1 = decoder(x_t, self.t_transform(t), z_1)
            _, gradient_2 = decoder(x_t, self.t_transform(t), z_2)
            gradient = (1.0 - alpha) * gradient_1 + alpha * gradient_2
            coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = predicted_noise - coef * gradient

            predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                            self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
            predicted_x_0 = predicted_x_0.clamp(-1, 1)

            new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t,
                                                          shape) * x_t - predicted_x_0) / \
                                  self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

            alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

            x_t = predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

        return x_t

    def latent_ddim_sample(self, latent_denoise_fn, z_t, t):
        """
        Perform a single step of DDIM sampling in the latent space.
        
        This function is specifically designed for sampling in the compressed latent space
        of the AutoencoderKL, which is crucial for the efficiency of the LDAE framework.
        
        Args:
            latent_denoise_fn (callable): The denoising network operating in the latent space.
            z_t (torch.Tensor): The latent at timestep t.
            t (torch.Tensor): The current timestep.
            
        Returns:
            torch.Tensor: The latent at timestep t-1.
        """
        shape = z_t.shape
        predicted_noise = latent_denoise_fn(z_t, self.t_transform(t))
        predicted_z_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * z_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_z_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * predicted_noise

    def latent_ddim_sample_loop(self, latent_denoise_fn, z_T):
        """
        Perform complete DDIM sampling in the latent space from pure noise to a clean latent.
        
        This function iteratively applies latent_ddim_sample to generate clean latent
        representations from noise, which can then be decoded by the AutoencoderKL into
        high-quality brain MRI scans.
        
        Args:
            latent_denoise_fn (callable): The denoising network operating in the latent space.
            z_T (torch.Tensor): Initial noise tensor in the latent space (at timestep T).
            
        Returns:
            torch.Tensor: The final generated clean latent at timestep 0.
        """
        shape = z_T.shape
        batch_size = shape[0]
        z = z_T
        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            z = self.ddim_sample(latent_denoise_fn, z, t)
        return z
