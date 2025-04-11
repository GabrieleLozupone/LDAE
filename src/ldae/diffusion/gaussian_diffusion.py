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

import math
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from .ddim import DDIM

"""
Originally ported from here: https://github.com/ckczzj/PDAE/tree/master and adapted for the LDAE framework.
"""

def calculate_theta(a, b):
    """
    Calculate the angle (theta) between two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        The angle in radians between tensors 'a' and 'b'
    """
    return torch.arccos(torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b)))


def slerp(a, b, alpha):
    """
    Spherical Linear Interpolation between two tensors.
    
    Used for smooth interpolation between two latent representations in the stochastic space.
    
    Args:
        a: First tensor
        b: Second tensor
        alpha: Interpolation factor between 0.0 and 1.0
        
    Returns:
        Interpolated tensor between 'a' and 'b'
    """
    theta = calculate_theta(a, b)
    sin_theta = torch.sin(theta)
    return a * torch.sin((1.0 - alpha) * theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta


def lerp(a, b, alpha):
    """
    Linear Interpolation between two tensors.
    
    Used for interpolation between two semantic representations.
    
    Args:
        a: First tensor
        b: Second tensor
        alpha: Interpolation factor between 0.0 and 1.0
        
    Returns:
        Interpolated tensor between 'a' and 'b'
    """
    return (1.0 - alpha) * a + alpha * b

class GaussianDiffusion:
    """
    GaussianDiffusion is the main class implementing the diffusion process for Latent Diffusion Autoencoders (LDAE), Pretrained Diffusion Autoencoder (PDAE) e Diffusion Autoencoders (DAE).
    
    This class handles various operations related to the diffusion process, including:
    - Forward diffusion (adding noise in a controlled manner)
    - Reverse diffusion (denoising process)
    - Training utility functions 
    - DDPM and DDIM sampling
    - Representation learning
    - Semantic manipulation
    - Latent space interpolation
    
    The diffusion process is used in the LDAE framework to model the distribution of compressed latent representations
    of 3D MRI scans, enabling high-quality reconstruction, manipulation, and generation.
    """
    def __init__(self, config, device):
        """
        Initialize the GaussianDiffusion model.
        
        Args:
            config (dict): Configuration dictionary containing parameters like:
                - timesteps: Number of diffusion timesteps
                - betas_type: Type of noise schedule ("linear" or "cosine")
            device (torch.device): Device to run computations on
        """
        super().__init__()
        self.device = device
        self.timesteps = config["timesteps"]
        betas_type = config["betas_type"]
        if betas_type == "linear":
            betas = np.linspace(0.0001, 0.02, self.timesteps)
        elif betas_type == "cosine":
            betas = []
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            max_beta = 0.999
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            betas = np.array(betas)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.to_torch = to_torch

        # Pre-calculate diffusion parameters for efficiency
        self.alphas = to_torch(alphas)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = to_torch(posterior_variance)
        # clip the log because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_log_variance_clipped = to_torch(posterior_log_variance_clipped)

        self.x_0_posterior_mean_x_0_coef = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.x_0_posterior_mean_x_t_coef = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.noise_posterior_mean_x_t_coef = to_torch(np.sqrt(1. / alphas))
        self.noise_posterior_mean_noise_coef = to_torch(betas / (np.sqrt(alphas) * np.sqrt(1. - alphas_cumprod)))

        self.shift_coef = to_torch(- np.sqrt(alphas) * (1. - alphas_cumprod_prev) / np.sqrt(1. - alphas_cumprod))

        # SNR-weighted loss coefficients
        snr = alphas_cumprod / (1. - alphas_cumprod)
        gamma = 0.1
        self.weight = to_torch(snr ** gamma / (1. + snr))

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        """
        Extract coefficients at specific timesteps and reshape them to match the input tensor shape.
        
        Args:
            schedule (torch.Tensor): Tensor containing the coefficients for all timesteps
            t (torch.Tensor): Selected timesteps to extract
            x_shape (tuple): Shape of the input tensor
            
        Returns:
            torch.Tensor: Coefficients reshaped to broadcast with the input tensor
        """
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        """
        Convert DDPM parameters to DDIM parameters by selecting a subset of timesteps.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            original_alphas_cumprod (np.ndarray): Original cumulative alphas from DDPM
            
        Returns:
            tuple: (new_betas, timestep_map) - DDIM-specific betas and a mapping from DDIM to DDPM timesteps
        """
        original_timesteps = original_alphas_cumprod.shape[0]
        ddim_step = int(ddim_style[len("ddim"):])
        # data: x_{-1}  noisy latents: x_{0}, x_{1}, x_{2}, ..., x_{T-2}, x_{T-1}
        # encode: treat input x_{-1} as starting point x_{0}
        # sample: treat ending point x_{0} as output x_{-1}
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, ddim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)

    # x_start: batch_size x channel x height x width
    # t: batch_size
    def q_sample(self, x_0, t, noise):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Adds noise to the initial sample x_0 according to the diffusion schedule.
        
        Args:
            x_0 (torch.Tensor): Initial sample (clean image/latent)
            t (torch.Tensor): Timesteps to compute noise for
            noise (torch.Tensor): Random noise to add
            
        Returns:
            torch.Tensor: Noisy sample x_t at the specified timestep
        """
        shape = x_0.shape
        return (
                self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
                + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def q_posterior_mean(self, x_0, x_t, t):
        """
        Compute the mean of the posterior distribution q(x_{t-1} | x_t, x_0).
        
        This is used during the reverse process to compute the mean for sampling x_{t-1} given x_t and x_0.
        
        Args:
            x_0 (torch.Tensor): Initial clean sample
            x_t (torch.Tensor): Current noisy sample
            t (torch.Tensor): Current timestep
            
        Returns:
            torch.Tensor: Mean of the posterior distribution
        """
        shape = x_t.shape
        return self.extract_coef_at_t(self.x_0_posterior_mean_x_0_coef, t, shape) * x_0 \
            + self.extract_coef_at_t(self.x_0_posterior_mean_x_t_coef, t, shape) * x_t

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def noise_p_sample(self, x_t, t, predicted_noise, learned_range=None):
        """
        Sample from p(x_{t-1} | x_t) using the predicted noise.
        
        This is the primary reverse diffusion sampling step, removing noise from x_t to generate x_{t-1}.
        
        Args:
            x_t (torch.Tensor): Current noisy sample at timestep t
            t (torch.Tensor): Current timestep
            predicted_noise (torch.Tensor): Noise predicted by the denoising model
            learned_range (torch.Tensor, optional): Model-predicted variance parameter
            
        Returns:
            torch.Tensor: Sample x_{t-1} with less noise
        """
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def x_0_clip_p_sample(self, x_t, t, predicted_noise, learned_range=None, clip_x_0=True):
        """
        Sample from p(x_{t-1} | x_t) using the x_0 prediction.
        
        Alternative sampling method that first predicts x_0, clips it, and then samples x_{t-1}.
        
        Args:
            x_t (torch.Tensor): Current noisy sample at timestep t
            t (torch.Tensor): Current timestep
            predicted_noise (torch.Tensor): Noise predicted by the denoising model
            learned_range (torch.Tensor, optional): Model-predicted variance parameter
            clip_x_0 (bool): Whether to clip the predicted x_0 to [-1, 1]
            
        Returns:
            torch.Tensor: Sample x_{t-1} with less noise
        """
        shape = x_t.shape

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
        if clip_x_0:
            predicted_x_0.clamp_(-1, 1)
        predicted_mean = self.q_posterior_mean(predicted_x_0, x_t, t)
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def learned_range_to_log_variance(self, learned_range, t):
        """
        Convert model-predicted range parameter to log variance.
        
        The learned_range is a model output in [-1, 1] that is converted to variance in log space.
        
        Args:
            learned_range (torch.Tensor): Model-predicted range parameter in [-1, 1]
            t (torch.Tensor): Current timestep
            
        Returns:
            torch.Tensor: Log variance for the posterior distribution
        """
        shape = learned_range.shape
        min_log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        max_log_variance = self.extract_coef_at_t(torch.log(self.betas), t, shape)
        # The learned_range is [-1, 1] for [min_var, max_var].
        frac = (learned_range + 1) / 2
        return min_log_variance + frac * (max_log_variance - min_log_variance)

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        """
        Convert predicted noise to a prediction of the clean sample x_0.
        
        Args:
            x_t (torch.Tensor): Current noisy sample at timestep t
            t (torch.Tensor): Current timestep
            predicted_noise (torch.Tensor): Noise predicted by the denoising model
            
        Returns:
            torch.Tensor: Predicted clean sample x_0
        """
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
            - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def predicted_noise_to_predicted_mean(self, x_t, t, predicted_noise):
        """
        Convert predicted noise to the mean of the posterior distribution.
        
        Args:
            x_t (torch.Tensor): Current noisy sample at timestep t
            t (torch.Tensor): Current timestep
            predicted_noise (torch.Tensor): Noise predicted by the denoising model
            
        Returns:
            torch.Tensor: Predicted mean for the posterior distribution p(x_{t-1} | x_t)
        """
        shape = x_t.shape
        return self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise

    def p_loss(self, noise, predicted_noise, weight=None, loss_type="l2"):
        """
        Calculate the loss between the true noise and the predicted noise.
        
        This is the main training objective for diffusion models.
        
        Args:
            noise (torch.Tensor): True noise added during the forward process
            predicted_noise (torch.Tensor): Noise predicted by the denoising model
            weight (torch.Tensor, optional): Optional weighting factor for the loss
            loss_type (str): Type of loss function to use ('l1' or 'l2')
            
        Returns:
            torch.Tensor: Computed loss value
        """
        if loss_type == 'l1':
            return (noise - predicted_noise).abs().mean()
        elif loss_type == 'l2':
            if weight is not None:
                return torch.mean(weight * (noise - predicted_noise) ** 2)
            else:
                return torch.mean((noise - predicted_noise) ** 2)
        else:
            raise NotImplementedError

    """
        test pretrained dpms
    """

    def test_pretrained_dpms(self, ddim_style, denoise_fn, x_T, condition=None):
        """
        Test pretrained diffusion probabilistic models (DPMs) using DDIM sampling.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            denoise_fn (callable): Denoising function (UNet)
            x_T (torch.Tensor): Starting noise tensor
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            torch.Tensor: Generated sample
        """
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    """
        ddim
    """

    def ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        """
        DDIM (Denoising Diffusion Implicit Models) sampling process.
        
        Generates samples using the more efficient DDIM sampling algorithm,
        which uses fewer steps than DDPM while maintaining high sample quality.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            denoise_fn (callable): Denoising function (UNet)
            x_T (torch.Tensor): Starting noise tensor
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            torch.Tensor: Generated sample
        """
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(denoise_fn, x_T, condition)

    def ddim_encode(self, ddim_style, denoise_fn, x_0, condition=None):
        """
        DDIM encoding process (deterministically map a clean sample to a noisy latent).
        
        This is the inverse of the DDIM sampling process, used to encode a clean image 
        into the latent space for manipulation.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            denoise_fn (callable): Denoising function (UNet)
            x_0 (torch.Tensor): Clean input tensor to encode
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            torch.Tensor: Encoded latent in noise space
        """
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_encode_loop(denoise_fn, x_0, condition)

    """
        regular
    """

    def regular_train_one_batch(self, denoise_fn, x_0, condition=None):
        """
        Train one batch of data using the standard diffusion training objective.
        
        This is the standard training method for diffusion models, where the model learns
        to predict the noise added during the forward process.
        
        Args:
            denoise_fn (callable): Denoising function (UNet)
            x_0 (torch.Tensor): Clean input tensor
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            dict: Dictionary containing the prediction loss
        """
        shape = x_0.shape
        batch_size = shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_t, t, condition)

        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def regular_ddim_encode(self, ddim_style, denoise_fn, x_0, disable_tqdm=False):
        """
        Standard DDIM encoding process with option to disable progress bar.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            denoise_fn (callable): Denoising function (UNet)
            x_0 (torch.Tensor): Clean input tensor to encode
            disable_tqdm (bool): Whether to disable the progress bar
            
        Returns:
            torch.Tensor: Encoded latent in noise space
        """
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_encode_loop(denoise_fn, x_0, disable_tqdm=disable_tqdm)

    def regular_ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        """
        Standard DDIM sampling process.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            denoise_fn (callable): Denoising function (UNet)
            x_T (torch.Tensor): Starting noise tensor
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            torch.Tensor: Generated sample
        """
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    def regular_ddpm_sample(self, denoise_fn, x_T, condition=None):
        """
        DDPM (Denoising Diffusion Probabilistic Models) sampling process.
        
        This is the original, slower sampling process that uses all timesteps defined in the model.
        
        Args:
            denoise_fn (callable): Denoising function (UNet)
            x_T (torch.Tensor): Starting noise tensor
            condition (torch.Tensor, optional): Conditional input to the denoising function
            
        Returns:
            torch.Tensor: Generated sample
        """
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            output = denoise_fn(img, t, condition)
            if output.shape[1] == 2 * shape[1]:
                predicted_noise, learned_range = torch.split(output, shape[1], dim=1)
            else:
                predicted_noise = output
                learned_range = None
            img = self.noise_p_sample(img, t, predicted_noise, learned_range)
        return img

    """
        representation learning
    """

    def representation_learning_train_one_batch(self, encoder, decoder, x_0):
        """
        Train one batch for representation learning in the diffusion framework.
        
        This method implements the training procedure for PDAE where the semantic encoder generates
        conditional information for the gradient estimator to guide the diffusion process.
        
        Args:
            encoder (nn.Module): Semantic encoder network that extracts semantic latent codes
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor
            
        Returns:
            dict: Dictionary containing the prediction loss
        """
        shape = x_0.shape
        batch_size = shape[0]

        z = encoder(x_0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        predicted_noise, gradient = decoder(x_t, t, z)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)

        # weight = None
        weight = self.extract_coef_at_t(self.weight, t, shape)

        prediction_loss = self.p_loss(noise, predicted_noise + shift_coef * gradient, weight=weight)

        return {
            'prediction_loss': prediction_loss
        }

    def latent_representation_learning_train_one_batch(self, encoder, decoder, x_0, z_0):
        """
        Train one batch for latent representation learning in LDAE.
        
        Unlike standard representation learning, this operates on the compressed latent (z_0)
        rather than directly on the input image. The semantic encoder still extracts features from x_0.
        
        Args:
            encoder (nn.Module): Semantic encoder network that extracts conditional embeddings
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor (MRI scan)
            z_0 (torch.Tensor): Compressed latent representation from AutoencoderKL
            
        Returns:
            dict: Dictionary containing the prediction loss
        """
        shape = x_0.shape
        batch_size = shape[0]

        emb = encoder(x_0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(x_0=z_0, t=t, noise=noise)

        predicted_noise, gradient = decoder(z_t, t, emb)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)

        # weight = None
        weight = self.extract_coef_at_t(self.weight, t, shape)

        prediction_loss = self.p_loss(noise, predicted_noise + shift_coef * gradient, weight=weight)

        return {
            'prediction_loss': prediction_loss
        }

    def representation_learning_ddpm_sample(self, encoder, decoder, x_0, x_T, z=None):
        """
        DDPM sampling guided by the semantic representation for representation learning.
        
        This method uses the full DDPM sampling with the conditional gradient guidance.
        
        Args:
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Reference clean tensor to extract semantic features from
            x_T (torch.Tensor): Starting noise tensor
            z (torch.Tensor, optional): Pre-computed semantic embedding
            
        Returns:
            torch.Tensor: Generated sample guided by the semantic features
        """
        shape = x_0.shape
        batch_size = shape[0]

        if z is None:
            z = encoder(x_0)
        img = x_T

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise, gradient = decoder(img, t, z)
            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            img = self.noise_p_sample(img, t, predicted_noise + shift_coef * gradient)
        return img

    def representation_learning_ddim_sample(self, ddim_style, encoder, decoder, x_0, x_T, z=None, stop_percent=0.0, disable_tqdm=False):
        """
        DDIM sampling guided by the semantic representation for representation learning.
        
        More efficient version of the sampling process using DDIM with semantic guidance.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Reference clean tensor to extract semantic features from
            x_T (torch.Tensor): Starting noise tensor
            z (torch.Tensor, optional): Pre-computed semantic embedding
            stop_percent (float): Percentage of steps to stop applying the gradient guidance
            disable_tqdm (bool): Whether to disable the progress bar
            
        Returns:
            torch.Tensor: Generated sample guided by the semantic features
        """
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_sample_loop(decoder, z, x_T, stop_percent=stop_percent, disable_tqdm=disable_tqdm)

    def representation_learning_diffae_sample(self, ddim_style, encoder, unet, x_0, x_T, z=None, disable_tqdm=False):
        """
        DDIM sampling for DAE using a standard diffusion autoencoder approach.
        
        This method is similar to the original DiffAE implementation using semantic guidance.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            encoder (nn.Module): Semantic encoder network
            unet (nn.Module): Conditional UNet for denoising
            x_0 (torch.Tensor): Reference clean tensor to extract semantic features from
            x_T (torch.Tensor): Starting noise tensor
            z (torch.Tensor, optional): Pre-computed semantic embedding
            disable_tqdm (bool): Whether to disable the progress bar
            
        Returns:
            torch.Tensor: Generated sample guided by the semantic features
        """
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(unet, x_T, z)

    def representation_learning_diffae_encode(self, ddim_style, encoder, unet, x_0, z=None):
        """
        DDIM encoding for DAE using a standard diffusion autoencoder approach.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            encoder (nn.Module): Semantic encoder network
            unet (nn.Module): Conditional UNet for denoising
            x_0 (torch.Tensor): Clean input tensor to encode
            z (torch.Tensor, optional): Pre-computed semantic embedding
            
        Returns:
            torch.Tensor: Encoded latent in noise space
        """
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_encode_loop(unet, x_0, z)

    def representation_learning_ddim_encode(self, ddim_style, encoder, decoder, x_0, z=None):
        """
        DDIM encoding guided by the semantic representation for representation learning (PDAE approach).
        
        This encodes a clean sample to a noisy latent guided by the semantic representation.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor to encode
            z (torch.Tensor, optional): Pre-computed semantic embedding
            
        Returns:
            torch.Tensor: Encoded latent in noise space
        """
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_encode_loop(decoder, z, x_0)

    def latent_representation_learning_ddim_encode(self, ddim_style, encoder, decoder, x_0, z_0, style=None, disable_tqdm=False):
        """
        DDIM encoding for latent representation learning in LDAE.
        
        Encodes a compressed latent z_0 to a noisy latent z_T guided by semantic features from x_0.
        This operates in the latent space of the autoencoder.
        
        Args:
            ddim_style (str): String in format 'ddim{steps}' indicating number of DDIM steps
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor to extract semantic features from
            z_0 (torch.Tensor): Compressed latent representation from AutoencoderKL
            style (torch.Tensor, optional): Pre-computed semantic embedding
            disable_tqdm (bool): Whether to disable the progress bar
            
        Returns:
            torch.Tensor: Encoded latent in noise space
        """
        if style is None:
            style = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_encode_loop(decoder, style, z_0, disable_tqdm=disable_tqdm)

    def representation_learning_autoencoding(self, encoder_ddim_style, decoder_ddim_style, encoder, decoder, x_0):
        """
        Perform autoencoding using representation learning in PDAE approaches.
        
        This function first encodes the input to a latent noise representation, then decodes it back.
        It operates on the original image space (not the compressed latent).
        
        Args:
            encoder_ddim_style (str): DDIM style for encoding
            decoder_ddim_style (str): DDIM style for decoding
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor to encode and reconstruct
            
        Returns:
            torch.Tensor: Reconstructed output
        """
        z = encoder(x_0)
        inferred_x_T = self.representation_learning_ddim_encode(encoder_ddim_style, encoder, decoder, x_0, z)
        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, inferred_x_T, z)

    def latent_representation_learning_autoencoding(self, encoder_ddim_style, decoder_ddim_style, encoder, decoder, x_0, z_0, disable_tqdm=False):
        """
        Perform autoencoding in latent space using LDAE framework.
        
        This operates on the compressed latent from the AutoencoderKL, guided by semantic features.
        It encodes z_0 to a noisy latent, then decodes it back to a clean latent.
        
        Args:
            encoder_ddim_style (str): DDIM style for encoding
            decoder_ddim_style (str): DDIM style for decoding
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor to extract semantic features from
            z_0 (torch.Tensor): Compressed latent representation from AutoencoderKL
            disable_tqdm (bool): Whether to disable the progress bar
            
        Returns:
            torch.Tensor: Reconstructed compressed latent
        """
        y_sem = encoder(x_0)
        inferred_z_T = self.latent_representation_learning_ddim_encode(encoder_ddim_style, None, decoder, None, z_0, y_sem, disable_tqdm)
        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, inferred_z_T, y_sem, disable_tqdm=disable_tqdm)
    
    def representation_learning_gap_measure(self, encoder, decoder, x_0):
        """
        Measure the posterior mean gap for representation learning evaluation.
        
        This method quantifies how well the model approximates the true posterior distribution
        by measuring the gap between the true posterior mean and the predicted posterior mean,
        both with and without the gradient guidance from the semantic encoder.
        
        Args:
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor
            
        Returns:
            tuple: (predicted_posterior_mean_gap, autoencoder_posterior_mean_gap) - 
                  Lists of mean squared errors between true posterior and predicted posteriors
        """
        shape = x_0.shape
        batch_size = shape[0]
        z = encoder(x_0)

        predicted_posterior_mean_gap = []
        autoencoder_posterior_mean_gap = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.q_sample(x_0, t, torch.rand_like(x_0))
            predicted_noise, gradient = decoder(x_t, t, z)

            predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
            predicted_posterior_mean = self.q_posterior_mean(predicted_x_0, x_t, t)

            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
            autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)
            autoencoder_predicted_posterior_mean = self.q_posterior_mean(autoencoder_predicted_x_0, x_t, t)

            true_posterior_mean = self.q_posterior_mean(x_0, x_t, t)

            predicted_posterior_mean_gap.append(
                torch.mean((true_posterior_mean - predicted_posterior_mean) ** 2, dim=[0, 1, 2, 3]).cpu().item())
            autoencoder_posterior_mean_gap.append(
                torch.mean((true_posterior_mean - autoencoder_predicted_posterior_mean) ** 2,
                           dim=[0, 1, 2, 3]).cpu().item())

        return predicted_posterior_mean_gap, autoencoder_posterior_mean_gap

    def representation_learning_denoise_one_step(self, encoder, decoder, x_0, timestep_list):
        """
        Perform one-step denoising with representation learning for visualization.
        
        This method compares regular denoising vs. denoising guided by the semantic representation
        at specified timesteps, useful for visualizing the impact of semantic guidance.
        
        Args:
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Clean input tensor
            timestep_list (list): List of timesteps to denoise at
            
        Returns:
            tuple: (predicted_x_0, autoencoder_predicted_x_0) - 
                  Predicted clean samples without and with semantic guidance
        """
        shape = x_0.shape

        t = torch.tensor(timestep_list, device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_0, t, noise=torch.randn_like(x_0))
        z = encoder(x_0)
        predicted_noise, gradient = decoder(x_t, t, z)

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
        autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
        autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)

        return predicted_x_0, autoencoder_predicted_x_0

    """
        latent (If you want to train another DDPM on the learned semantic code)
    """

    @property
    def latent_diffusion_config(self):
        """
        Configure parameters for latent semantic diffusion.
        
        This property defines the diffusion parameters specifically for the latent semantic space diffusion
        process, which may differ from the parameters used for the regular diffusion in voxel space.
        
        Returns:
            dict: Dictionary containing latent semantic diffusion parameters
        """
        timesteps = 1000
        betas = np.array([0.008] * timesteps)
        # betas = np.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        loss_type = "l1"

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        return {
            "timesteps": timesteps,
            "betas": betas,
            "alphas_cumprod": to_torch(alphas_cumprod),
            "sqrt_alphas_cumprod": to_torch(sqrt_alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": to_torch(sqrt_one_minus_alphas_cumprod),
            "loss_type": loss_type,
        }

    def normalize(self, z, mean, std):
        """
        Normalize a latent vector with the provided mean and standard deviation.
        
        Args:
            z (torch.Tensor): Latent vector to normalize
            mean (torch.Tensor): Mean for normalization
            std (torch.Tensor): Standard deviation for normalization
            
        Returns:
            torch.Tensor: Normalized latent vector
        """
        return (z - mean) / std

    def denormalize(self, z, mean, std):
        """
        Denormalize a latent vector with the provided mean and standard deviation.
        
        Args:
            z (torch.Tensor): Normalized latent vector
            mean (torch.Tensor): Mean for denormalization
            std (torch.Tensor): Standard deviation for denormalization
            
        Returns:
            torch.Tensor: Denormalized latent vector
        """
        return z * std + mean

    def latent_diffusion_train_one_batch(self, latent_denoise_fn, encoder, x_0, latents_mean, latents_std):
        """
        Train one batch for latent semantic diffusion.
        
        This trains a diffusion model that operates directly on the semantic latent vectors 
        produced by the encoder.
        
        Args:
            latent_denoise_fn (nn.Module): Denoising network for the latent space
            encoder (nn.Module): Semantic encoder network
            x_0 (torch.Tensor): Clean input tensor
            latents_mean (torch.Tensor): Mean of semantic latents for normalization
            latents_std (torch.Tensor): Standard deviation of semantic latents for normalization
            
        Returns:
            dict: Dictionary containing the prediction loss
        """
        timesteps = self.latent_diffusion_config["timesteps"]

        sqrt_alphas_cumprod = self.latent_diffusion_config["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod = self.latent_diffusion_config["sqrt_one_minus_alphas_cumprod"]

        z_0 = encoder(x_0)
        z_0 = z_0.detach()
        z_0 = self.normalize(z_0, latents_mean, latents_std)

        shape = z_0.shape
        batch_size = shape[0]

        t = torch.randint(0, timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_0)

        z_t = self.extract_coef_at_t(sqrt_alphas_cumprod, t, shape) * z_0 \
              + self.extract_coef_at_t(sqrt_one_minus_alphas_cumprod, t, shape) * noise

        predicted_noise = latent_denoise_fn(z_t, t)

        prediction_loss = self.p_loss(noise, predicted_noise, loss_type=self.latent_diffusion_config["loss_type"])

        return {
            'prediction_loss': prediction_loss,
        }

    def latent_diffusion_sample(self, latent_ddim_style, decoder_ddim_style, latent_denoise_fn, decoder, x_T,
                                latents_mean, latents_std):
        """
        Generate samples using the latent semantic diffusion model.
        
        This samples a semantic latent vector from pure noise using the latent diffusion model,
        then uses this vector to guide the generation.
        
        Args:
            latent_ddim_style (str): DDIM style for latent diffusion sampling
            decoder_ddim_style (str): DDIM style for voxel-space diffusion sampling
            latent_denoise_fn (nn.Module): Denoising network for the latent semantic space
            decoder (nn.Module): Unet + Gradient estimator network (conditional decoder)
            x_T (torch.Tensor): Starting noise tensor for the voxel space
            latents_mean (torch.Tensor): Mean of semantic latents for denormalization
            latents_std (torch.Tensor): Standard deviation of semantic latents for denormalization
            
        Returns:
            torch.Tensor: Generated brain MRI sample
        """
        alphas_cumprod = self.latent_diffusion_config["alphas_cumprod"]

        batch_size = x_T.shape[0]
        input_channel = latent_denoise_fn.input_channel
        z_T = torch.randn((batch_size, input_channel), device=self.device)

        z_T.clamp_(-1.0, 1.0)  # may slightly improve sample quality

        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(latent_ddim_style, alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        z = ddim.latent_ddim_sample_loop(latent_denoise_fn, z_T)

        z = self.denormalize(z, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, x_T, z,
                                                        stop_percent=0.3)

    """
        manipulation
    """

    def manipulation_train_one_batch(self, classifier, encoder, x_0, label, latents_mean, latents_std):
        """
        Train a classifier on the semantic embedding for manipulation tasks.
        
        This method enables learning a direction in the semantic space that corresponds to
        a specific attribute or class (e.g., Alzheimer's disease).
        
        Args:
            classifier (nn.Module): Binary classifier model
            encoder (nn.Module): Semantic encoder network
            x_0 (torch.Tensor): Clean input tensor
            label (torch.Tensor): Binary labels for the classifier
            latents_mean (torch.Tensor): Mean of semantic latents for normalization
            latents_std (torch.Tensor): Standard deviation of semantic latents for normalization
            
        Returns:
            dict: Dictionary containing the classification loss
        """
        z = encoder(x_0)
        z = z.detach()
        z_norm = self.normalize(z, latents_mean, latents_std)

        prediction = classifier(z_norm)

        gt = torch.where(label > 0, torch.ones_like(label).float(), torch.zeros_like(label).float())
        loss = F.binary_cross_entropy_with_logits(prediction, gt)
        return {
            'bce_loss': loss,
        }

    def manipulation_sample(self, ddim_style, classifier_weight, encoder, decoder, x_0, inferred_x_T, latents_mean,
                            latents_std, class_id, scale):
        """
        Manipulate the semantic latent vector to modify attributes in the generated sample (PDAE approach).
        
        This method shifts the semantic embedding along a direction corresponding to a specific
        attribute (e.g., adding or removing signs of Alzheimer's disease) and generates 
        a modified brain scan.
        
        Args:
            ddim_style (str): DDIM style for voxel-space diffusion sampling
            classifier_weight (torch.Tensor): Weight vector of the trained classifier
            encoder (nn.Module): Semantic encoder network
            decoder (nn.Module): Gradient estimator network (conditional decoder)
            x_0 (torch.Tensor): Input tensor to extract semantic features from
            inferred_x_T (torch.Tensor): Noisy latent obtained from DDIM encoding
            latents_mean (torch.Tensor): Mean of semantic latents for normalization
            latents_std (torch.Tensor): Standard deviation of semantic latents for normalization
            class_id (int): Class ID to manipulate towards (e.g., 0 for CN, 1 for AD)
            scale (float): Scaling factor for the manipulation strength
            
        Returns:
            torch.Tensor: Manipulated sample
        """
        z = encoder(x_0)
        z_norm = self.normalize(z, latents_mean, latents_std)

        import math
        z_norm_manipulated = z_norm + scale * math.sqrt(512) * F.normalize(classifier_weight[class_id][None, :], dim=1)
        z_manipulated = self.denormalize(z_norm_manipulated, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, inferred_x_T, z_manipulated,
                                                        stop_percent=0.0)
    
    def latent_semantic_manipulation(self, ddim_style, encoder, decoder, x_0, z_0, mean, std, direction, scale, disable_tqdm=False):
        """
        Manipulate the latent representation of the input image by adding a scaled direction vector to the normalized
        latent representation.
        Args:
            encoder: The encoder model to extract the latent representation.
            decoder: The decoder model to reconstruct the image.
            x_0: The input image tensor.
            z_0: The latent representation of the input image.
            mean: The mean value for normalization.
            std: The standard deviation value for normalization.
            direction: The direction vector for manipulation.
            scale: The scaling factor for the direction vector.
        """
        y_sem = encoder(x_0)
        inferred_z_T = self.latent_representation_learning_ddim_encode(ddim_style, None, decoder, None, z_0, y_sem, disable_tqdm)
        norm_y_sem = self.normalize(y_sem, mean, std)
        # Check if the y_sem have batch dimension and get the embedding dim
        if len(norm_y_sem.shape) == 2:
            _, embedding_dim = norm_y_sem.shape
        elif len(norm_y_sem.shape) == 1:
            embedding_dim = norm_y_sem.shape[0]
        else:
            raise ValueError("y_sem should have 1 or 2 dimensions")
        manipulation = norm_y_sem + scale * math.sqrt(embedding_dim) * F.normalize(direction, dim=1)
        y_sem_manipulated = self.denormalize(manipulation, mean, std)
        return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, inferred_z_T, y_sem_manipulated, disable_tqdm=disable_tqdm)
    
    """
        Interpolation
    """

    def latent_lerp_slerp_interpolation(self, ddim_style, encoder, decoder, x_0, x_1, z_0, z_1, alpha, disable_tqdm=False):
        """
        Lerp and Slerp interpolation between two images. Lerp will be applied on the semantic representation and Slerp
        on the latent stochastic code.
        Args:
            encoder: The encoder model to extract the latent representation.
            decoder: The decoder model to reconstruct the image.
            x_0: The first input image tensor.
            x_1: The second input image tensor.
            z_0: The latent representation of the first input image.
            z_1: The latent representation of the second input image.
            alpha: The interpolation factor (0.0 to 1.0).
        """
        y_sem_0 = encoder(x_0)
        y_sem_1 = encoder(x_1)
        inferred_z_T_0 = self.latent_representation_learning_ddim_encode(ddim_style, None, decoder, None, z_0, y_sem_0, disable_tqdm)
        inferred_z_T_1 = self.latent_representation_learning_ddim_encode(ddim_style, None, decoder, None, z_1, y_sem_1, disable_tqdm)
        y_sem = lerp(y_sem_0, y_sem_1, alpha)
        z_T = slerp(inferred_z_T_0, inferred_z_T_1, alpha)
        return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, z_T, y_sem, disable_tqdm=disable_tqdm)
    
    def latent_lerp_lerp_interpolation(self, ddim_style, encoder, decoder, x_0, x_1, z_0, z_1, alpha, disable_tqdm=False):
        """
        Lerp and Lerp interpolation between two images. Lerp will be applied on both the semantic representation and stochastic one.
        Args:
            encoder: The encoder model to extract the latent representation.
            decoder: The decoder model to reconstruct the image.
            x_0: The first input image tensor.
            x_1: The second input image tensor.
            z_0: The latent representation of the first input image.
            z_1: The latent representation of the second input image.
            alpha: The interpolation factor (0.0 to 1.0).
        """
        y_sem_0 = encoder(x_0)
        y_sem_1 = encoder(x_1)
        inferred_z_T_0 = self.latent_representation_learning_ddim_encode(ddim_style, None, decoder, None, z_0, y_sem_0, disable_tqdm)
        inferred_z_T_1 = self.latent_representation_learning_ddim_encode(ddim_style, None, decoder, None, z_1, y_sem_1, disable_tqdm)
        y_sem = lerp(y_sem_0, y_sem_1, alpha)
        z_T = lerp(inferred_z_T_0, inferred_z_T_1, alpha)
        return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, z_T, y_sem, disable_tqdm=disable_tqdm)
    
    def representation_learning_ddim_trajectory_interpolation(self, ddim_style, decoder, z_1, z_2, x_T, alpha):
        """
        Perform trajectory interpolation between two semantic embeddings for PDAE.
        
        This method uses the DDIM framework to interpolate between two semantic embeddings
        during the reverse diffusion process, creating a smooth transition between two brain states.
        
        Args:
            ddim_style (str): DDIM style for voxel-space diffusion sampling
            decoder (nn.Module): Gradient estimator network (conditional decoder)
            z_1 (torch.Tensor): First semantic embedding
            z_2 (torch.Tensor): Second semantic embedding
            x_T (torch.Tensor): Noisy latent to start the reverse diffusion from
            alpha (float): Interpolation factor between 0.0 and 1.0
            
        Returns:
            torch.Tensor: Interpolated sample
        """
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_trajectory_interpolation(decoder, z_1, z_2, x_T, alpha)
