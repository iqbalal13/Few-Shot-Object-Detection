# ==========================================================
# STEP 8 : Prototype-guided Cross Attention
# ==========================================================

import torch
import torch.nn as nn

class PrototypeCrossAttention(nn.Module):

    def __init__(self, feature_dim=CONFIG["hidden_dim"]):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, prototype, query_feature):

        # query_feature :
        # [B,256,H,W]

        B, C, H, W = query_feature.shape

        # menjadi
        # [B,H*W,256]

        query = query_feature.flatten(2).permute(0,2,1)

        # prototype
        # [B,256]

        prototype = prototype.unsqueeze(1)

        attended, _ = self.attention(
            query=query,
            key=prototype,
            value=prototype
        )

        return attended
