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

        # ---------------------------------------------
        # query_feature
        # Shape : [B,256,H,W]
        # ---------------------------------------------
        B, C, H, W = query_feature.shape

        # ---------------------------------------------
        # Flatten menjadi token
        # [B,256,H,W] -> [B,H*W,256]
        # ---------------------------------------------
        query = query_feature.flatten(2).permute(0, 2, 1)

        # ---------------------------------------------
        # Prototype
        # [B,256] -> [B,1,256]
        # ---------------------------------------------
        prototype = prototype.unsqueeze(1)

        # ---------------------------------------------
        # Prototype-guided Cross Attention
        # ---------------------------------------------
        attended, _ = self.attention(
            query=query,
            key=prototype,
            value=prototype
        )

        # ---------------------------------------------
        # Residual Connection
        # Mempertahankan informasi query asli
        # ---------------------------------------------
        output = query + attended

        return output
