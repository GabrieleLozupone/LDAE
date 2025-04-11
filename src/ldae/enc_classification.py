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
from src.ldae.base_nets import LitClassification3DBase
import torch.nn as nn
import torch


class EncoderClassifier(LitClassification3DBase):
    """
    This module is used to classify 3D images. It uses a backbone to extract features from the images, then it computes
    a global representation of the input using soft attention. The global vector is then projected into Q, K, V and used
    to compute self-attention with cross-attention. Finally, it applies a classifier to the output of the attention
    fusion. !!!Note: See the LitClassification3DBase class for the inherited attributes and methods.
    Args:
        num_heads (int): The number of transformer heads to use.
        attn_dropout (float): The dropout rate to apply to the attention weights.
        return_attention_weights (bool): Whether to return the attention weights.
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, num_slices, 3, 224, 224)`.
    Outputs:
        logits (torch.Tensor): The output of the module with shape `(batch_size, num_classes)`.
    """

    def __init__(self,
                 num_heads=8,
                 attn_dropout=0.1,
                 pretrained_semantic_path=None,
                 return_attention_weights=False, **kwargs):
        super().__init__(**kwargs)
        # Set the model parameters
        self.return_attention_weights = return_attention_weights
        # Add the modules
        self.slice_distributed = SliceDistributed(self.backbone)
        self.positional_encoding = PositionalEncoding(d_model=self.embedding_dim, max_length=self.load_size)
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
        # Classifier
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)  # Final linear layer

        if pretrained_semantic_path is not None:
            self.load_state_dict(torch.load(pretrained_semantic_path, map_location="cpu"), strict=False)
            print(f"Loaded pretrained semantic encoder weights from {pretrained_semantic_path}")

        self.save_hyperparameters()

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
        x = x.squeeze(1)  # e.g. (32, 1, 512) -> (32, 512)
        # 8. Apply the classifier
        x = self.classifier(x)  # e.g. (32, 512) -> (32, num_classes)
        # Return the output
        if self.return_attention_weights:
            return x, attention_weights
        return x
