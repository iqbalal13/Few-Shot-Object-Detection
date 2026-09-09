# ==========================================================
# STEP 5 : Support Prototype Encoder
#
# support feature map
#      ↓
# GAP
#      ↓
# 2048 → 512 → 256
#      ↓
# L2 normalized semantic prototype P
# ==========================================================

class SupportPrototypeEncoder(nn.Module):

    def __init__(
        self,
        in_channels=
            CONFIG[
                "backbone_out_channels"
            ],
        hidden_dim=
            CONFIG[
                "support_hidden_dim"
            ],
        output_dim=
            CONFIG[
                "hidden_dim"
            ],
        dropout=
            CONFIG[
                "dropout"
            ]
    ):
        super().__init__()


        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )


        self.projector = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            )
        )


    def forward(
        self,
        support_feature_map
    ):

        if support_feature_map.dim() != 4:

            raise ValueError(
                "Support encoder expects "
                "[B,C,H,W]."
            )


        pooled = (

            self.pool(
                support_feature_map
            )

            .flatten(1)
        )


        prototype = (

            self.projector(
                pooled
            )
        )


        prototype = F.normalize(

            prototype,

            p=2,

            dim=-1
        )


        return prototype


print("=" * 70)
print("STEP 5 : SUPPORT PROTOTYPE ENCODER DEFINED")
print("=" * 70)
