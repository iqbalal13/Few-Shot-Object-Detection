# ==========================================================
# STEP 8 : DETR-Style Object Transformer Decoder
#
# - 100 learned object-query positions
# - NO support conditioning here
#
# Self-attention:
# Q,K = tgt + object_query_pos
#
# Cross-attention:
# Q = tgt + object_query_pos
# K = memory + spatial_pos
# V = memory
# ==========================================================


class DETRTransformerDecoderLayer(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=
            CONFIG[
                "hidden_dim"
            ],
        num_heads=
            CONFIG[
                "num_heads"
            ],
        dim_feedforward=
            CONFIG[
                "dim_feedforward"
            ],
        dropout=
            CONFIG[
                "dropout"
            ]
    ):
        super().__init__()


        self.self_attn = (
            nn.MultiheadAttention(

                embed_dim=
                    hidden_dim,

                num_heads=
                    num_heads,

                dropout=
                    dropout,

                batch_first=True
            )
        )


        self.cross_attn = (
            nn.MultiheadAttention(

                embed_dim=
                    hidden_dim,

                num_heads=
                    num_heads,

                dropout=
                    dropout,

                batch_first=True
            )
        )


        self.linear1 = nn.Linear(

            hidden_dim,

            dim_feedforward
        )


        self.linear2 = nn.Linear(

            dim_feedforward,

            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.dropout1 = nn.Dropout(
            dropout
        )


        self.dropout2 = nn.Dropout(
            dropout
        )


        self.dropout3 = nn.Dropout(
            dropout
        )


        self.norm1 = nn.LayerNorm(
            hidden_dim
        )


        self.norm2 = nn.LayerNorm(
            hidden_dim
        )


        self.norm3 = nn.LayerNorm(
            hidden_dim
        )


        self.activation = nn.ReLU()


    @staticmethod
    def with_pos_embed(
        tensor,
        pos
    ):

        if pos is None:
            return tensor

        return tensor + pos


    def forward(
        self,
        tgt,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=None
    ):

        # --------------------------------------------------
        # SELF ATTENTION
        # --------------------------------------------------

        q = self.with_pos_embed(
            tgt,
            query_pos
        )

        k = self.with_pos_embed(
            tgt,
            query_pos
        )


        tgt2 = self.self_attn(

            query=q,

            key=k,

            value=tgt,

            need_weights=False
        )[0]


        tgt = (

            tgt
            +
            self.dropout1(
                tgt2
            )
        )


        tgt = self.norm1(
            tgt
        )


        # --------------------------------------------------
        # CROSS ATTENTION
        # --------------------------------------------------

        q = self.with_pos_embed(
            tgt,
            query_pos
        )


        k = self.with_pos_embed(
            memory,
            memory_pos
        )


        tgt2 = self.cross_attn(

            query=q,

            key=k,

            value=memory,

            key_padding_mask=
                memory_key_padding_mask,

            need_weights=False
        )[0]


        tgt = (

            tgt
            +
            self.dropout2(
                tgt2
            )
        )


        tgt = self.norm2(
            tgt
        )


        # --------------------------------------------------
        # FFN
        # --------------------------------------------------

        tgt2 = self.linear2(

            self.dropout(

                self.activation(

                    self.linear1(
                        tgt
                    )
                )
            )
        )


        tgt = (

            tgt
            +
            self.dropout3(
                tgt2
            )
        )


        tgt = self.norm3(
            tgt
        )


        return tgt


# ==========================================================
# FULL DECODER
# ==========================================================

class DETRObjectDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim=
            CONFIG[
                "hidden_dim"
            ],
        num_queries=
            CONFIG[
                "num_queries"
            ],
        num_layers=
            CONFIG[
                "num_decoder_layers"
            ]
    ):
        super().__init__()


        self.num_queries = (
            num_queries
        )


        # Learned positional embeddings
        # for DETR object queries.

        self.object_queries = (
            nn.Embedding(

                num_queries,

                hidden_dim
            )
        )


        base_layer = (
            DETRTransformerDecoderLayer()
        )


        self.layers = nn.ModuleList([

            copy.deepcopy(
                base_layer
            )

            for _ in range(
                num_layers
            )
        ])


        self.final_norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )


    def forward(
        self,
        memory,
        memory_pos,
        memory_key_padding_mask=None
    ):

        B = memory.shape[0]


        query_pos = (

            self.object_queries
            .weight

            .unsqueeze(0)

            .expand(
                B,
                -1,
                -1
            )
        )


        # DETR target/object content
        # begins at zero.

        tgt = torch.zeros_like(
            query_pos
        )


        for layer in self.layers:

            tgt = layer(

                tgt=
                    tgt,

                memory=
                    memory,

                memory_pos=
                    memory_pos,

                query_pos=
                    query_pos,

                memory_key_padding_mask=
                    memory_key_padding_mask
            )


        tgt = self.final_norm(
            tgt
        )


        return tgt


print("=" * 70)
print("STEP 8 : DETR OBJECT DECODER DEFINED")
print("=" * 70)
