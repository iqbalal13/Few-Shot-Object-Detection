# ==========================================================
# STEP 8.5 : 2D Sine Positional Encoding (FINAL)
# ==========================================================

import math
import torch
import torch.nn as nn


class PositionEmbeddingSine2D(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        scale=2 * math.pi
    ):
        super().__init__()

        assert hidden_dim % 2 == 0

        self.num_pos_feats = \
            hidden_dim // 2

        self.temperature = temperature

        self.normalize = normalize

        self.scale = scale

    def forward(
        self,
        feature_map
    ):

        # feature_map:
        # [B,C,H,W]

        B, C, H, W = feature_map.shape

        device = feature_map.device

        # --------------------------------------------------
        # Spatial coordinates
        # --------------------------------------------------

        y_embed = torch.arange(
            1,
            H + 1,
            dtype=torch.float32,
            device=device
        )

        x_embed = torch.arange(
            1,
            W + 1,
            dtype=torch.float32,
            device=device
        )

        y_embed, x_embed = torch.meshgrid(
            y_embed,
            x_embed,
            indexing="ij"
        )

        if self.normalize:

            eps = 1e-6

            y_embed = (
                y_embed
                /
                (H + eps)
                *
                self.scale
            )

            x_embed = (
                x_embed
                /
                (W + eps)
                *
                self.scale
            )

        # --------------------------------------------------
        # Frequency dimensions
        # --------------------------------------------------

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=device
        )

        dim_t = self.temperature ** (
            2
            *
            torch.div(
                dim_t,
                2,
                rounding_mode="floor"
            )
            /
            self.num_pos_feats
        )

        # --------------------------------------------------
        # X positional encoding
        # --------------------------------------------------

        pos_x = (
            x_embed[:, :, None]
            /
            dim_t
        )

        pos_x = torch.stack(
            (
                pos_x[:, :, 0::2].sin(),
                pos_x[:, :, 1::2].cos()
            ),
            dim=3
        ).flatten(2)

        # --------------------------------------------------
        # Y positional encoding
        # --------------------------------------------------

        pos_y = (
            y_embed[:, :, None]
            /
            dim_t
        )

        pos_y = torch.stack(
            (
                pos_y[:, :, 0::2].sin(),
                pos_y[:, :, 1::2].cos()
            ),
            dim=3
        ).flatten(2)

        # --------------------------------------------------
        # Combine Y + X
        #
        # [H,W,256]
        # --------------------------------------------------

        pos = torch.cat(
            (
                pos_y,
                pos_x
            ),
            dim=2
        )

        # --------------------------------------------------
        # Batch dimension
        #
        # [B,H,W,256]
        # ->
        # [B,H*W,256]
        # --------------------------------------------------

        pos = (
            pos
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
        )

        pos = pos.flatten(
            1,
            2
        )

        return pos


position_encoding = PositionEmbeddingSine2D(
    hidden_dim=CONFIG["hidden_dim"]
).to(CONFIG["device"])


print("=" * 70)
print("STEP 8.5 : 2D Positional Encoding Ready")
print("=" * 70)
