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


import lightning as L
import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.networks.layers import Act
import torch.nn.functional as F
import wandb


class LitAutoencoderKL(L.LightningModule):
    """
    The LigtningModule for the AutoencoderKL model training. This module is used to train/fine-tune the compression
    model discussed in the paper.
    """
    def __init__(self,
                 aekl_args,
                 kl_weight=1e-6,
                 disc_layers=3,
                 disc_num_channels=32,
                 disc_kernel_size=4,
                 perceptual_loss_network="squeeze",
                 perceptual_loss_fake_3d_ratio=0.25,
                 patch_adversarial_loss_criterion="least_squares",
                 adv_weight=0.01,
                 perceptual_weight=0.001,
                 lr_g=1e-4,
                 weight_decay_g=1e-3,
                 lr_d=5e-4,
                 weight_decay_d=1e-3,
                 accumulate_grad_batches=4,
                 load_autoenc_from_checkpoint=None):
        super().__init__()

        self.model = AutoencoderKL(**aekl_args)
        self.kl_weight = kl_weight

        if load_autoenc_from_checkpoint is not None:
            self.model.load_state_dict(torch.load(load_autoenc_from_checkpoint))

        self.discriminator = PatchDiscriminator(
            spatial_dims=aekl_args["spatial_dims"],
            num_layers_d=disc_layers,
            num_channels=disc_num_channels,
            in_channels=aekl_args["in_channels"],
            out_channels=aekl_args["out_channels"],
            kernel_size=disc_kernel_size,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=aekl_args["spatial_dims"],
                                              network_type=perceptual_loss_network,
                                              fake_3d_ratio=perceptual_loss_fake_3d_ratio)
        self.perceptual_weight = perceptual_weight

        self.adv_loss = PatchAdversarialLoss(criterion=patch_adversarial_loss_criterion, no_activation_leastsq=True)
        self.adv_weight = adv_weight

        self.lr_g = lr_g
        self.weight_decay_g = weight_decay_g
        self.lr_d = lr_d
        self.weight_decay_d = weight_decay_d
        self.accumulate_grad_batches = accumulate_grad_batches

        # Disable automatic optimization to control the two optimizers separately
        self.automatic_optimization = False
        # Initialize step counters to synchronize the processes
        self.generator_step_counter = 0
        self.discriminator_step_counter = 0
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_g)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        return optimizer_g, optimizer_d

    def training_step(self, batch, batch_idx):
        # Get the optimizers
        opt_g, opt_d = self.optimizers()

        x = batch["image"]

        # Generator part
        recon, z_mu, z_sigma = self(x)
        logits_fake = self.discriminator(recon.contiguous().float())[-1]

        recons_loss = F.l1_loss(recon.float(), x.float())
        p_loss = self.perceptual_loss(recon.float(), x.float())
        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        kl_loss = 0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(z_sigma ** 2) - 1, dim=[1, 2, 3, 4])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss_g = (recons_loss + (self.kl_weight * kl_loss) + (self.perceptual_weight * p_loss) +
                  (self.adv_weight * generator_loss))

        # scale losses by 1/N (for N batches of gradient accumulation)
        loss_g /= self.accumulate_grad_batches
        self.manual_backward(loss_g)

        # Increment and synchronize generator step counter
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_g.step()
            opt_g.zero_grad()

        # Discriminator part
        logits_fake = self.discriminator(recon.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d = 0.5 * (loss_d_fake + loss_d_real) * self.adv_weight

        # scale losses by 1/N (for N batches of gradient accumulation)
        loss_d /= self.accumulate_grad_batches
        self.manual_backward(loss_d)

        # Increment and synchronize discriminator step counter
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_d.step()
            opt_d.zero_grad()

        # Log losses
        self.log("train_loss_g", loss_g*self.accumulate_grad_batches, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_d", loss_d*self.accumulate_grad_batches, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("kl_loss", kl_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("perceptual_loss", p_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_recons", recons_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        recon, z_mu, z_sigma = self(x)
        recons_loss = F.l1_loss(recon.float(), x.float())

        # Log loss
        self.log("val_loss", recons_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Log center slice of the first image in the batch (assuming batch size > 0)
        if self.global_rank == 0 and batch_idx == 0:
            # Extract the center slice of the first image (assuming 3D volume in shape [B, C, H, W, D])
            img = x[0, 0].cpu().numpy()  # First image, first channel
            recon_img = recon[0, 0].cpu().numpy()  # First reconstructed image, first channel

            # Get center slices along each axis
            images_real = [img[:, :, img.shape[2] // 2], img[:, img.shape[1] // 2, :], img[img.shape[0] // 2, :, :]]
            captions_real = ["Center axial real", "Center coronal real", "Center sagittal real"]

            images_recon = [recon_img[:, :, recon_img.shape[2] // 2], recon_img[:, recon_img.shape[1] // 2, :],
                            recon_img[recon_img.shape[0] // 2, :, :]]
            captions_recon = ["Center axial recon", "Center coronal recon", "Center sagittal recon"]

            # Log each image separately using Wandb
            wandb_logger = self.logger.experiment  # Get the Wandb experiment
            for i, (img_real, img_recon, cap_real, cap_recon) in enumerate(
                    zip(images_real, images_recon, captions_real, captions_recon)):
                # Use wandb.Image to log
                wandb_logger.log({
                    f"val_real_slice_{i}": wandb.Image(img_real, caption=cap_real),
                    f"val_recon_slice_{i}": wandb.Image(img_recon, caption=cap_recon),
                })

        return recons_loss

    def forward(self, x):
        return self.model(x)
