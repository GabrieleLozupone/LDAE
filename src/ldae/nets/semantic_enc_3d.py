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

import torch.nn as nn
from src.ldae.modules.module import (AttentionBlock, normalization, View)


class SemanticEncoder3D(nn.Module):
    """
    A simple 3D CNN encoder for semantic representation learning.
    """
    def __init__(self, latent_dim=768, shape=(128, 160, 128)):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(

            nn.Conv3d(1, 64, (3, 3, 3), (2, 2, 2), 1),          # batch_size x 64 x 64 x 64 x 64

            normalization(64),
            nn.SiLU(True),
            nn.Conv3d(64, 128, (3, 3, 3), (2, 2, 2), 1),          # batch_size x 128 x 32 x 32 x 32

            normalization(128),
            nn.SiLU(True),
            nn.Conv3d(128, 256, (3, 3, 3), (2, 2, 2), 1),         # batch_size x 256 x 16 x 16 x 16

            AttentionBlock(256, 4, -1, False),

            normalization(256),
            nn.SiLU(True),
            nn.Conv3d(256, 256, (3, 3, 3), (2, 2, 3), 1),          # batch_size x 256 x 8 x 8 x 8

            normalization(256),
            nn.SiLU(True),
            nn.Conv3d(256, 256, (3, 3, 3), (2, 2, 3), 1),          # batch_size x 256 x 4 x 4 x 4

            normalization(256),
            nn.SiLU(True),
            View((-1, 256 * shape[0] // 32 * shape[1] // 32 * shape[2] // 32)),
            nn.Linear(256 * shape[0] // 32 * shape[1] // 32 * shape[2] // 32, self.latent_dim)
        )

    # x: batch_size x 3 x 128 x 128
    def forward(self, x):
        # batch_size x latent_dim
        return self.encoder(x)
