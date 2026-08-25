# ==========================================================
# STEP 10 : Transformer Decoder + Object Queries (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_decoder_layers"],
        num_queries=CONFIG["num_queries"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    ):
        super().__init__()

        # ==================================================
        # Learnable DETR Object Queries
        # ==================================================

        self.object_queries = nn.Embedding(
            num_queries,
            hidden_dim
        )

        # ==================================================
        # Transformer Decoder
        # ==================================================

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, memory):

        # memory:
        # [B, H*W, 256]

        batch_size = memory.shape[0]

        object_queries = (
            self.object_queries
            .weight
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        output = self.decoder(
            tgt=object_queries,
            memory=memory
        )

        # [B, num_queries, hidden_dim]

        return output


# ==========================================================
# Initialize
# ==========================================================

transformer_decoder = TransformerDecoder().to(
    CONFIG["device"]
)

print("=" * 70)
print("STEP 10 : Transformer Decoder Ready")
print("=" * 70)
print(transformer_decoder)
