# ==========================================================
# STEP 8 : Prototype-guided Cross Attention
# ==========================================================

import torch
import torch.nn as nn

class PrototypeCrossAttention(nn.Module):

    def __init__(self, feature_dim=2048):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, prototype, query_feature):

        # query_feature :
        # [B,C,H,W]

        B,C,H,W = query_feature.shape

        # menjadi
        # [B,625,2048]

        query = query_feature.flatten(2).permute(0,2,1)

        # prototype
        # [B,2048]

        prototype = prototype.unsqueeze(1)

        attended,_ = self.attention(
            query=query,
            key=prototype,
            value=prototype
        )

        return attended
