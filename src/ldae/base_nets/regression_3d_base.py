# --------------------------------------------------------------------------------
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
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .backbone_base import BackboneBaseModule


class LitRegression3DBase(L.LightningModule):
    """
    This is a base class for 3D regression tasks. It's used to train, validate, and test a model on 3D data.
    """
    def __init__(self,
                 backbone_args,
                 embedding_dim=768,
                 load_size=224,
                 lr=1e-4,
                 weight_decay=1e-3,
                 epochs=30,
                 warmup_epochs=5,
                 warmup_start_lr=1e-5,
                 age_min=50.4,  # Set your dataset mean age here
                 age_max=97.3):  # Set your dataset max here
        super().__init__()

        self.embedding_dim = embedding_dim
        self.load_size = load_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.age_min = age_min
        self.age_max = age_max

        self.save_hyperparameters()

        if backbone_args is not None:
            self.backbone = BackboneBaseModule(**backbone_args)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.backbone(x)

    def normalize(self, x):
        return (x - self.hparams.age_min) / self.hparams.age_max

    def unnormalize(self, x):
        return x * self.hparams.age_max + self.hparams.age_min

    def remove_nan_entries(self, pred, target):
        """
        Removes entries (rows) where either prediction or target has NaNs.
        pred, target: [batch_size, 1]
        Returns the filtered (pred, target).
        """
        # valid_mask is True only if the row has *no* NaNs in pred or target
        valid_mask = ~torch.isnan(pred).any(dim=1) & ~torch.isnan(target).any(dim=1)
        pred = pred[valid_mask]
        target = target[valid_mask]
        return pred, target

    def shared_step(self, batch, stage: str):
        x, y = batch["image"], batch["age"]
        # Normalize and unsqueeze
        y = self.normalize(y).unsqueeze(1)

        # Forward pass
        pred_norm = self(x)

        # Unnormalize
        pred = self.unnormalize(pred_norm)
        y_true = self.unnormalize(y)

        # ----- Handle NaNs by removing them -----
        pred, y_true = self.remove_nan_entries(pred, y_true)

        # If the entire batch got removed due to NaNs, skip
        if pred.numel() == 0:
            return None  # Skip backprop

        # Compute base loss
        loss = self.loss_fn(pred, y_true)

        # Compute metrics via functional approach
        mae_value = F.l1_loss(pred, y_true, reduction='mean')
        rmse_value = torch.sqrt(F.mse_loss(pred, y_true, reduction='mean'))

        # Log everything
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_mae", mae_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_rmse", rmse_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        """Define and return your optimizer(s) here"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-7,
        )
        return optimizer
