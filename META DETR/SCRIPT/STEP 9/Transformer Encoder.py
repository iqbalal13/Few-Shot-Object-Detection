# ==========================================================
# STEP 9 : Transformer Encoder
# ==========================================================

import torch.nn as nn

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        return self.encoder(x)
