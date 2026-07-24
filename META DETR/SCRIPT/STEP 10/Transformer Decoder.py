# ==========================================================
# STEP 10 : Transformer Decoder
# ==========================================================

import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=8,
        num_layers=6,
        num_queries=100
    ):

        super().__init__()

        self.object_queries = nn.Parameter(
            torch.randn(num_queries, hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(self, encoder_output):

        batch_size = encoder_output.size(0)

        queries = self.object_queries.unsqueeze(0).repeat(
            batch_size,
            1,
            1
        )

        output = self.decoder(
            tgt=queries,
            memory=encoder_output
        )

        return output
