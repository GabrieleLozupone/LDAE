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

from src.ldae.modules import SliceDistributed, PositionalEncoding, SoftAttention
from src.ldae.base_nets.backbone_base import BackboneBaseModule
import torch.nn as nn


class AttentionSemanticEncoder(nn.Module):
    """
    This module is used to extract features from 3D images. It uses a 2D backbone to extract compact features maps
    from the images, then it computes a global representation of using soft attention. The global representation and the
    original ones are projected into Q, K, V and used to simulate self-attention with cross-attention.
    Args:
        backbone_args (dict): The arguments to pass to the BackboneBaseModule.
        emb_chans (int): The dimension of the final embedding.
        num_heads (int): The number of transformer heads to use.
        attn_dropout (float): The dropout rate to apply to the attention weights.
        seq_len (int): The length of the sequence (Number of slices in original paper).
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, D, 1, H, W)`.
    Outputs:
        y_sem (torch.Tensor): The output of the module with shape `(batch_size, 768)`.
    """

    def __init__(self,
                 backbone_args,
                 emb_chans=768,
                 num_heads=8,
                 attn_dropout=0.1,
                 seq_len=128):
        super().__init__()
        self.embedding_dim = emb_chans
        # Set the model parameters
        self.backbone = BackboneBaseModule(**backbone_args)
        # Add the modules
        self.slice_distributed = SliceDistributed(self.backbone)
        self.positional_encoding = PositionalEncoding(d_model=self.embedding_dim, max_length=seq_len)
        # Learnable projections for Q, K, V
        self.query_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        # Attention
        self.soft_attention = SoftAttention(input_size=self.embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads,
                                                     dropout=attn_dropout,
                                                     batch_first=True)
        # Layer Normalization after Cross-Attention
        self.layer_norm_after_attention = nn.LayerNorm(self.embedding_dim)

    def forward(self, x):
        # 1. Pass the input through the backbone
        x = self.slice_distributed(x)
        # 2. Compute the attention
        x = self.positional_encoding(x)
        # 3. Compute global vector with soft attention
        q, attention_weights = self.soft_attention(x)
        # 4. Linear projections for Q, K, V
        q = self.query_projection(q.unsqueeze(1))
        k = self.key_projection(x)
        v = self.value_projection(x)
        # 5. Compute self-attention using with cross-attention
        x, _ = self.cross_attention(q, k, v)
        # 6. Layer Normalization
        x = self.layer_norm_after_attention(x)
        # 7. Squeeze the second dimension
        x = x.squeeze(1)
        return x
