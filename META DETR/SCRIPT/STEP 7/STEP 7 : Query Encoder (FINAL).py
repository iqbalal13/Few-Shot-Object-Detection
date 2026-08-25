# ==========================================================
# STEP 7 : Query Encoder (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class QueryEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=256
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        query_feature
    ):

        # --------------------------------------------------
        # query_feature
        # [B,256,H,W]
        # --------------------------------------------------

        B, C, H, W = query_feature.shape

        # --------------------------------------------------
        # Flatten spatial map into Transformer tokens
        #
        # [B,256,H,W]
        # ->
        # [B,H*W,256]
        # --------------------------------------------------

        query_tokens = (
            query_feature
            .flatten(2)
            .transpose(1, 2)
        )

        query_tokens = self.norm(
            query_tokens
        )

        spatial_shape = (H, W)

        return query_tokens, spatial_shape


query_encoder = QueryEncoder(
    hidden_dim=CONFIG["hidden_dim"]
).to(CONFIG["device"])


print("=" * 70)
print("STEP 7 : Query Encoder Ready")
print("=" * 70)
