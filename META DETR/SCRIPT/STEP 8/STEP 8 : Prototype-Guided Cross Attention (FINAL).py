# ==========================================================
# STEP 8 : Prototype-Guided Cross Attention (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class PrototypeGuidedCrossAttention(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        query_tokens,
        support_tokens,
        prototype
    ):

        # --------------------------------------------------
        # Prototype:
        #
        # [B,256]
        # ->
        # [B,1,256]
        # --------------------------------------------------

        prototype_token = \
            prototype.unsqueeze(1)

        # --------------------------------------------------
        # Support memory:
        #
        # Global prototype
        # +
        # Spatial support tokens
        #
        # [B,1+HW,256]
        # --------------------------------------------------

        support_memory = torch.cat(
            [
                prototype_token,
                support_tokens
            ],
            dim=1
        )

        # --------------------------------------------------
        # Cross Attention
        #
        # Q = query image tokens
        #
        # K,V = support representation
        # --------------------------------------------------

        attended, _ = self.cross_attention(
            query=query_tokens,
            key=support_memory,
            value=support_memory,
            need_weights=False
        )

        # --------------------------------------------------
        # Residual connection
        # --------------------------------------------------

        output = self.norm(
            query_tokens
            +
            self.dropout(attended)
        )

        return output


cross_attention = PrototypeGuidedCrossAttention(

    hidden_dim=CONFIG["hidden_dim"],

    num_heads=CONFIG["num_heads"],

    dropout=CONFIG["dropout"]

).to(CONFIG["device"])


print("=" * 70)
print("STEP 8 : Prototype-Guided Cross Attention Ready")
print("=" * 70)
