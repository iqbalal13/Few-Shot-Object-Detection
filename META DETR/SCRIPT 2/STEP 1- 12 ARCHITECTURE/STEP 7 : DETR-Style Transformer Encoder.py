# ==========================================================
# STEP 7 : DETR-Style Transformer Encoder
#
# Every layer:
#
# Q = content + position
# K = content + position
# V = content
# ==========================================================


class DETRTransformerEncoderLayer(
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


        self.norm1 = nn.LayerNorm(
            hidden_dim
        )


        self.norm2 = nn.LayerNorm(
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
        src,
        pos=None,
        key_padding_mask=None
    ):

        q = self.with_pos_embed(
            src,
            pos
        )

        k = self.with_pos_embed(
            src,
            pos
        )


        src2 = self.self_attn(

            query=q,

            key=k,

            value=src,

            key_padding_mask=
                key_padding_mask,

            need_weights=False
        )[0]


        src = (

            src
            +
            self.dropout1(
                src2
            )
        )


        src = self.norm1(
            src
        )


        src2 = self.linear2(

            self.dropout(

                self.activation(

                    self.linear1(
                        src
                    )
                )
            )
        )


        src = (

            src
            +
            self.dropout2(
                src2
            )
        )


        src = self.norm2(
            src
        )


        return src


# ==========================================================
# FULL ENCODER
# ==========================================================

class DETRTransformerEncoder(
    nn.Module
):

    def __init__(
        self,
        num_layers=
            CONFIG[
                "num_encoder_layers"
            ]
    ):
        super().__init__()


        base_layer = (
            DETRTransformerEncoderLayer()
        )


        self.layers = nn.ModuleList([

            copy.deepcopy(
                base_layer
            )

            for _ in range(
                num_layers
            )
        ])


    def forward(
        self,
        src,
        pos,
        key_padding_mask=None
    ):

        output = src


        for layer in self.layers:

            output = layer(

                src=
                    output,

                pos=
                    pos,

                key_padding_mask=
                    key_padding_mask
            )


        return output


print("=" * 70)
print("STEP 7 : DETR-STYLE TRANSFORMER ENCODER DEFINED")
print("=" * 70)
