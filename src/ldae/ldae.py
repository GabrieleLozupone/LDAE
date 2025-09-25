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
import wandb
import copy  # For copying model parameters

from src.ldae.nets import AttentionSemanticEncoder
from src.ldae.nets import UNet, ShiftUNet
from src.ldae.diffusion import GaussianDiffusion
from generative.networks.nets import AutoencoderKL
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric, FIDMetric
from generative.losses import PerceptualLoss


class LatentDiffusionAutoencoders(L.LightningModule):
    def __init__(self,
                 enc_args,
                 unet_args,
                 vae_args,
                 vae_path,
                 timesteps_args,
                 lr=2.5e-5,
                 mode="pretrain",
                 pretrained_path=None,
                 ema_decay=0.9999,
                 ema_update_after_step=1):
        super().__init__()
        self.mode = mode
        self.lr = lr
        self.timesteps_args = timesteps_args
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step

        # Set the metrics
        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=3)
        self.mssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=3)

        # Diffusion helper
        self.gaussian_diffusion = GaussianDiffusion(
            timesteps_args, device=self.device
        )

        self.encoder = None
        self.unet = None
        self.decoder = None

        # ------------------- Autoencoder (VAE) -------------------
        self.vae = AutoencoderKL(**vae_args)
        if vae_path is not None:
            print(f"Loading VAE from {vae_path}")
            self.vae.load_state_dict(torch.load(vae_path))
            print("VAE model loaded successfully.")

        # Init the scale factor
        self.scale_factor = torch.tensor(0.8730)
        """
                Computes the scaling factor for normalizing the latent space representation of inputs.

                This function encodes the inputs from the first batch of the provided dataloader using the
                specified autoencoder. It then calculates the reciprocal of the standard deviation of the
                resulting latent variables. This scaling factor is used to normalize the latent space,
                ensuring that the latent variables have unit standard deviation. Such normalization is
                essential for the stable training of diffusion models in the latent space, as it standardizes
                the latent variables, making the diffusion process more stable and effective.

                See https://arxiv.org/pdf/2112.10752 Appendix G Details on Autoencoder Models for more details.
        """

        # ------------------- Pretrain Models -------------------
        if mode == "pretrain":
            self.unet = UNet(**unet_args)

            # Create EMA copy of UNet (disable grad, so it won't update by backprop)
            self.ema_unet = copy.deepcopy(self.unet).eval()
            for p in self.ema_unet.parameters():
                p.requires_grad = False

        # ------------------- Representation-Learning Models -------------------
        elif mode == "representation-learning":
            # Load the pretrained denoising model
            weights = None
            try:
                data = torch.load(pretrained_path)
                weights = {key.replace('ema_unet.', ''): value for key, value in data['state_dict'].items()}
            except FileNotFoundError:
                print(f"Pretrained model not found at {pretrained_path}")
            # ShiftUNet decoder
            self.decoder = ShiftUNet(
                latent_dim=enc_args["emb_chans"],
                **unet_args,
            )
            if weights is not None:
                self.decoder.load_state_dict(weights, strict=False)
                print("Successfully loaded the pretrained decoder model.")
            else:
                print("No weights loaded for the decoder.")

            # Semantic encoder
            self.encoder = AttentionSemanticEncoder(**enc_args)

            # Create EMA copies (disable grad, so they won't update by backprop)
            self.ema_encoder = copy.deepcopy(self.encoder).eval()
            for p in self.ema_encoder.parameters():
                p.requires_grad = False

            self.ema_decoder = copy.deepcopy(self.decoder).eval()
            for p in self.ema_decoder.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Save hyperparameters
        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        """
        Called when fit begins. Re-initialize the diffusion if needed and set train modes.
        """
        self.gaussian_diffusion = GaussianDiffusion(
            self.timesteps_args, device=self.device
        )
        print("on_fit_start device:", self.device)

        if self.mode == "representation-learning":
            self.encoder.train()
            self.decoder.set_train_mode()
            print("Encoder and decoder set to training mode.")

    def on_test_start(self) -> None:
        self.gaussian_diffusion = GaussianDiffusion(
            self.timesteps_args, device=self.device
        )
        # Add LPIPS metric
        self.lpips = PerceptualLoss(spatial_dims=3,
                                    network_type="squeeze",
                                    fake_3d_ratio=0.5)
        self.lpips = self.lpips.to(self.device)
        print("LPIPS model loaded successfully.")
        # Add FID metric if testing pretraining stage
        if self.mode == "pretrain":
            try:
                self.fid_model = torch.hub.load("Warvito/MedicalNet-models", "medicalnet_resnet50_23datasets")
                self.fid_model = self.fid_model.to(self.device)
                self.fid_model.eval()
                # Add also lists to store the features
                self.real_features = torch.tensor([]).to(self.device)
                self.generated_features = torch.tensor([]).to(self.device)
                self.reconstructed_features = torch.tensor([]).to(self.device)
                print("FID model loaded successfully.")
            except Exception as e:
                print(f"Error loading FID model: {e} trying to load from local folder")
                try:
                    self.fid_model = torch.hub.load("pretrained", "resnet_50_23dataset.pth")
                    self.fid_model = self.fid_model.to(self.device)
                    self.fid_model.eval()
                    # Add also lists to store the features
                    self.real_features = torch.tensor([]).to(self.device)
                    self.generated_features = torch.tensor([]).to(self.device)
                    self.reconstructed_features = torch.tensor([]).to(self.device)
                    print("FID model loaded successfully from local folder.")
                except:
                    print("Error loading FID model from local folder")
            del self.unet
            print("Deleted UNet model, maintaining only EMA model.")
        elif self.mode == "representation-learning":
            # Delete the non-EMA models
            del self.encoder
            del self.decoder
            print("Deleted encoder and decoder models, maintaining only EMA models.")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @torch.no_grad()
    def decode(self, z):
        # Decode latents using the VAE (stage 2)
        z = z / self.scale_factor
        self.vae.eval()
        with torch.no_grad():
            recon = self.vae.decode_stage_2_outputs(z)
        return recon

    def training_step(self, batch, batch_idx):
        """
        Standard training loop:
          - Either pretraining the UNet in latent space
          - Or learning a latent representation (encoder + ShiftUNet)
        """
        z0 = batch["latent"]
        z0 = z0 * self.scale_factor
        output = None

        # ------------------- 1) Denoiser pretraining -------------------
        if self.mode == "pretrain":
            output = self.gaussian_diffusion.regular_train_one_batch(
                denoise_fn=self.unet,
                x_0=z0,
            )

        # ------------------- 2) Representation learning -------------------
        elif self.mode == "representation-learning" or self.mode == "refining":
            x0 = batch["image"]
            output = self.gaussian_diffusion.latent_representation_learning_train_one_batch(
                encoder=self.encoder,
                decoder=self.decoder,
                x_0=x0,
                z_0=z0,
            )

        if output is None:
            raise ValueError(f"Invalid mode: {self.mode}")

        loss = output['prediction_loss']
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        After each training batch, update the EMA parameters if the step
        matches the `ema_update_after_step` criteria.
        """
        # Global step in PL counts across epochs
        if self.global_step % self.ema_update_after_step == 0 and self.global_step > 0:
            if self.mode == "pretrain":
                self.update_ema(
                    ema_model=self.ema_unet,
                    current_model=self.unet,
                    decay=self.ema_decay
                )
            elif self.mode == "representation-learning":
                self.update_ema(
                    ema_model=self.ema_encoder,
                    current_model=self.encoder,
                    decay=self.ema_decay
                )
                self.update_ema(
                    ema_model=self.ema_decoder,
                    current_model=self.decoder,
                    decay=self.ema_decay
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

    @torch.no_grad()
    def update_ema(self, ema_model, current_model, decay):
        """
        Update EMA model parameters by a factor of `decay`.
        """
        ema_state_dict = ema_model.state_dict()
        current_state_dict = current_model.state_dict()

        for key, param in current_state_dict.items():
            if param.dtype in (torch.float16, torch.float32, torch.float64):
                ema_state_dict[key].data.mul_(decay).add_(param.data, alpha=1 - decay)

    def validation_step(self, batch, batch_idx):
        """
        Here we run inference to visualize outputs.
        """
        z0 = batch["latent"]
        z0 = z0 * self.scale_factor

        # For visualization (only for batch_idx == 0 on global_rank == 0)
        if batch_idx == 0 and self.global_rank == 0:
            print("Generating image...")
            z_0 = None
            if self.mode == "pretrain":
                # Use the EMA version (or the main unet) for sampling
                z_0 = self.gaussian_diffusion.regular_ddim_sample(
                    ddim_style='ddim50',
                    denoise_fn=self.ema_unet if hasattr(self, 'ema_unet') else self.unet,
                    x_T=torch.randn_like(z0),
                )
            elif self.mode == "representation-learning":
                x0 = batch["image"]
                # Use the EMA version (or the main encoder/decoder) for sampling
                z_0 = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style='ddim50',
                    encoder=self.ema_encoder if hasattr(self, 'ema_encoder') else self.encoder,
                    decoder=self.ema_decoder if hasattr(self, 'ema_decoder') else self.decoder,
                    x_0=x0,
                    x_T=torch.randn_like(z0),
                    disable_tqdm=True
                )
                # Log the original for comparison
                self.log_image_batch(
                    self.decode(z0).cpu().numpy(),
                    caption="original"
                )

            if z_0 is not None:
                print("Logging image...")
                if self.mode == "pretrain":
                    self.log_image_batch(
                        self.decode(z_0[0:min(4, z_0.shape[0])]).cpu().numpy(),
                        caption="generated"
                    )
                elif self.mode == "representation-learning":
                    self.log_image_batch(
                        self.decode(z_0).cpu().numpy(),
                        caption="generated"
                    )
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        if self.mode == "representation-learning":
            x0 = batch["image"]
            zt = self.gaussian_diffusion.representation_learning_ddim_sample(
                ddim_style='ddim100',
                encoder=self.ema_encoder if hasattr(self, 'ema_encoder') else self.encoder,
                decoder=self.ema_decoder if hasattr(self, 'ema_decoder') else self.decoder,
                x_0=x0,
                x_T=torch.randn_like(z0),
                disable_tqdm=True
            )
            recon = self.decode(zt)
            ssim = self.ssim(recon, self.decode(z0))
            self.log("val_ssim", ssim.mean(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return ssim

    def test_step(self, batch, batch_idx):
        """
        Here we test the model reconstruction and generation capabilities.
        """
        z0 = batch["latent"]
        z0 = z0 * self.scale_factor

        x0 = batch["image"]

        if self.mode == "pretrain":
            z_T = self.gaussian_diffusion.regular_ddim_encode(
                ddim_style='ddim100',
                denoise_fn=self.ema_unet,
                x_0=z0,
                disable_tqdm=True
            )
            # Use the EMA version (or the main unet) for sampling
            z_0 = self.gaussian_diffusion.regular_ddim_sample(
                ddim_style='ddim100',
                denoise_fn=self.ema_unet,
                x_T=torch.randn_like(z0),
            )
            x0_gen = self.decode(z_0)
            x0_recon = self.decode(z0)
            x0_real = x0.permute(0, 1, 3, 4, 2)
            # Log 10 images in total
            if batch_idx < 10:
                self.log_image_batch(x0_gen.cpu().numpy(), caption="generated")
                self.log_image_batch(x0_recon.cpu().numpy(), caption="reconstructed")
                self.log_image_batch(x0_real.cpu().numpy(), caption="real")
            # Compute the features with pretrained model
            gen_features = self.fid_model(x0_gen)
            gen_features = torch.nn.functional.adaptive_avg_pool3d(gen_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            real_features = self.fid_model(x0_real)
            real_features = torch.nn.functional.adaptive_avg_pool3d(real_features, 1).squeeze(-1).squeeze(-1).squeeze(
                -1)
            recon_features = self.fid_model(x0_recon)
            recon_features = torch.nn.functional.adaptive_avg_pool3d(recon_features, 1).squeeze(-1).squeeze(-1).squeeze(
                -1)
            # Append the features to the lists
            self.real_features = torch.cat([self.real_features, real_features], dim=0)
            self.generated_features = torch.cat([self.generated_features, gen_features], dim=0)
            self.reconstructed_features = torch.cat([self.reconstructed_features, recon_features], dim=0)
            # Now get reconstruction metrics with regular DDIM
            z_infer_ddim = self.gaussian_diffusion.regular_ddim_sample(
                ddim_style='ddim100',
                denoise_fn=self.ema_unet,
                x_T=z_T,
            )
            x_infer_ddim = self.decode(z_infer_ddim)
            mse = torch.nn.functional.mse_loss(x_infer_ddim, x0_real)
            ssim = self.ssim(x_infer_ddim, x0_real)
            mssim = self.mssim(x_infer_ddim, x0_real)
            lpips = self.lpips(x_infer_ddim, x0_real)
            if batch_idx < 10:
                self.log_image_batch(x_infer_ddim.cpu().numpy(), caption="inferred_ddim")
            self.log("test_ddim_mse", mse.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_ssim", ssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_mssim", mssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_lpips", lpips.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == "representation-learning":
            # Use the EMA version (or the main encoder/decoder) for sampling
            print("Encoding the image...")
            z_T = self.gaussian_diffusion.latent_representation_learning_ddim_encode(
                ddim_style='ddim100',
                encoder=self.ema_encoder,
                decoder=self.ema_decoder,
                x_0=x0,
                z_0=z0,
                disable_tqdm=True
            )
            print("Decoding with LDAE...")
            z0_hat = self.gaussian_diffusion.representation_learning_ddim_sample(
                ddim_style='ddim100',
                encoder=self.ema_encoder,
                decoder=self.ema_decoder,
                x_0=x0,
                x_T=z_T,
                disable_tqdm=True
            )
            z0_semantic = self.gaussian_diffusion.representation_learning_ddim_sample(
                ddim_style='ddim100',
                encoder=self.ema_encoder,
                decoder=self.ema_decoder,
                x_0=x0,
                x_T=torch.randn_like(z0),
                disable_tqdm=True
            )

            x0_semantic = self.decode(z0_semantic)
            x0_hat = self.decode(z0_hat)
            x0_recon = self.decode(z0)
            x0_real = x0.permute(0, 1, 3, 4, 2)  # [B, C, H, W, D]

            if batch_idx < 20:
                self.log_image_batch(x0_semantic.cpu().numpy(), caption="semantic")
                self.log_image_batch(x0_hat.cpu().numpy(), caption="inferred")
                self.log_image_batch(x0_recon.cpu().numpy(), caption="reconstructed")
                self.log_image_batch(x0_real.cpu().numpy(), caption="real")

            # Compute reconstruction metrics starting from the semantic space
            ldae_mse_semantic_original = torch.nn.functional.mse_loss(x0_semantic, x0_real)
            ldae_ssim_semantic_original = self.ssim(x0_semantic, x0_real)
            ldae_mssim_semantic_original = self.mssim(x0_semantic, x0_real)
            ldae_lpips_semantic_original = self.lpips(x0_semantic, x0_real)
            self.log("test_mse_ldae_semantic_original", ldae_mse_semantic_original.mean(), prog_bar=True, on_step=True,
                     on_epoch=True, sync_dist=True)
            self.log("test_ssim_ldae_semantic_original", ldae_ssim_semantic_original.mean(), prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_mssim_ldae_semantic_original", ldae_mssim_semantic_original.mean(), prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_lpips_ldae_semantic_original", ldae_lpips_semantic_original.mean(), prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True)

            # Compute reconstruction metrics starting from semantic + z_T inferred
            ldae_mse_original = torch.nn.functional.mse_loss(x0_hat, x0_real)
            ldae_ssim_original = self.ssim(x0_hat, x0_real)
            ldea_mssim_original = self.mssim(x0_hat, x0_real)
            ldae_lpips_original = self.lpips(x0_hat, x0_real)
            self.log("test_mse_ldae_original", ldae_mse_original.mean(), prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True)
            self.log("test_ssim_ldae_original", ldae_ssim_original.mean(), prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True)
            self.log("test_mssim_ldae_original", ldea_mssim_original.mean(), prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True)
            self.log("test_lpips_ldae_original", ldae_lpips_original.mean(), prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True)

            # Compute reconstruction metrics of the autoencoder
            aekl_mse = torch.nn.functional.mse_loss(x0_recon, x0_real)
            aekl_ssim = self.ssim(x0_recon, x0_real)
            aekl_mssim = self.mssim(x0_recon, x0_real)
            aekl_lpips = self.lpips(x0_recon, x0_real)
            self.log("test_mse_aekl", aekl_mse.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ssim_aekl", aekl_ssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_mssim_aekl", aekl_mssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_lpips_aekl", aekl_lpips.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def on_test_epoch_end(self):
        if self.mode == "pretrain":
            fid = FIDMetric()
            fid_original = fid(self.generated_features, self.real_features)
            fid_reconstructed = fid(self.generated_features, self.reconstructed_features)
            self.log("test_fid_original", fid_original, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_fid_reconstructed", fid_reconstructed, prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True)

    def log_image(self, recon, caption="generated"):
        """
        Helper for logging 3D volume slices to Weights & Biases.
        """
        images_recon = [
            recon[:, :, recon.shape[2] // 2],
            recon[:, recon.shape[1] // 2, :],
            recon[recon.shape[0] // 2, :, :]
        ]
        captions_recon = [
            f"Center axial {caption}",
            f"Center coronal {caption}",
            f"Center sagittal {caption}"
        ]

        # Log each image separately using Wandb
        wandb_logger = self.logger.experiment
        for i, (img_recon, cap_recon) in enumerate(zip(images_recon, captions_recon)):
            wandb_logger.log({
                f"val_{caption}_slice_{i}": wandb.Image(img_recon, caption=cap_recon),
            })

    def log_image_batch(self, recons, caption="generated"):
        """
        Log the center slices (axial, coronal, and sagittal) for an entire batch
        of 3D volumes to Weights & Biases.

        Args:
            recons (np.ndarray or torch.Tensor): A 4D array of shape [B, H, W, D].
            caption (str): Optional string to append to logging captions.
        """
        # If you're using PyTorch Tensors, you may want to convert to CPU & numpy:
        # recons = recons.detach().cpu().numpy()

        # Unpack shapes
        B, C, H, W, D = recons.shape

        # Grab center slices for each item in the batch
        # Axial slice is the middle slice along depth D
        axial_slices = recons[:, 0, :, :, D // 2]  # shape [B, H, W]

        # Coronal slice is the middle slice along width W
        # (note: shape [B, H, D])
        coronal_slices = recons[:, 0, :, W // 2, :]

        # Sagittal slice is the middle slice along height H
        # (note: shape [B, W, D])
        sagittal_slices = recons[:, 0, H // 2, :, :]

        # Prepare logger
        wandb_logger = self.logger.experiment

        # Helper to turn a batch of 2D slices into a list of wandb.Images
        def slices_to_wandb_images(slices_2d, view_name):
            """
            slices_2d is a 3D array of shape [B, height, width] for that particular view.
            We'll make a list of wandb.Image objects, one per batch item.
            """
            images = []
            for idx in range(B):
                # Each slice is a 2D array: [height, width]
                img_2d = slices_2d[idx]
                # Convert to wandb.Image, optionally applying a caption
                images.append(
                    wandb.Image(img_2d, caption=f"{caption} {view_name}, batch idx {idx}")
                )
            return images

        # Convert the slices for each view into a list of wandb.Images
        axial_wandb_images = slices_to_wandb_images(axial_slices, "axial")
        coronal_wandb_images = slices_to_wandb_images(coronal_slices, "coronal")
        sagittal_wandb_images = slices_to_wandb_images(sagittal_slices, "sagittal")

        # Finally, log them to W&B. Each key will appear as a gallery of images in the UI.
        wandb_logger.log({
            f"val_{caption}_axial": axial_wandb_images,
            f"val_{caption}_coronal": coronal_wandb_images,
            f"val_{caption}_sagittal": sagittal_wandb_images,
        })

    def configure_optimizers(self):
        """
        Return optimizer for either the UNet or (encoder + ShiftUNet).
        """
        if self.mode == "pretrain":
            return torch.optim.Adam([
                {"params": self.unet.parameters()},
            ], lr=self.lr, eps=1.0e-08, betas=(0.9, 0.999), weight_decay=0.0)
        elif self.mode == "representation-learning":
            return torch.optim.Adam([
                {"params": self.encoder.parameters()},
                {"params": self.decoder.label_emb.parameters()},
                {"params": self.decoder.shift_middle_block.parameters()},
                {"params": self.decoder.shift_output_blocks.parameters()},
                {"params": self.decoder.shift_out.parameters()},
            ], lr=self.lr, eps=1.0e-08, betas=(0.9, 0.999), weight_decay=0.0)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
