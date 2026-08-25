# ==========================================================
# STEP 9 : Transformer Encoder (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # Expected:
        # [B, H*W, hidden_dim]

        if x.dim() != 3:
            raise ValueError(
                f"Transformer Encoder expects [B,N,C], got {x.shape}"
            )

        return self.encoder(x)


# ==========================================================
# Initialize
# ==========================================================

transformer_encoder = TransformerEncoder().to(
    CONFIG["device"]
)

print("=" * 70)
print("STEP 9 : Transformer Encoder Ready")
print("=" * 70)
print(transformer_encoder)
