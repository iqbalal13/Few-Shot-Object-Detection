# ==========================================================
# STEP 5 : Support Prototype Encoder
#          + Padding-Mask-Aware Global Average Pooling
#
# support feature map
#       ↓
# masked global average pool
#       ↓
# 2048 -> 512 -> 256
#       ↓
# L2-normalized support prototype
#
# IMPORTANT:
# Letterbox padding must NOT contribute to the support
# prototype. If no mask is supplied, behavior falls back to
# ordinary global average pooling.
# ==========================================================

class SupportPrototypeEncoder(
    nn.Module
):

    def __init__(
        self,
        in_channels=CONFIG[
            "backbone_out_channels"
        ],
        hidden_dim=CONFIG[
            "support_hidden_dim"
        ],
        output_dim=CONFIG[
            "hidden_dim"
        ],
        dropout=CONFIG[
            "dropout"
        ],
    ):
        super().__init__()

        self.pool = (
            nn.AdaptiveAvgPool2d(1)
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
            ),
        )

    def forward(
        self,
        support_feature_map,
        padding_mask=None,
    ):

        if (
            support_feature_map.dim()
            != 4
        ):
            raise ValueError(
                "Support encoder expects "
                "[B,C,H,W]."
            )

        if padding_mask is None:

            pooled = (
                self.pool(
                    support_feature_map
                )
                .flatten(1)
            )

        else:

            if padding_mask.dim() != 3:
                raise ValueError(
                    "support padding_mask must be [B,H,W]."
                )

            mask = (
                padding_mask
                .to(
                    support_feature_map.device
                )
                .bool()
            )

            if (
                mask.shape[-2:]
                !=
                support_feature_map.shape[-2:]
            ):

                mask = (
                    F.interpolate(
                        mask[:, None].float(),
                        size=(
                            support_feature_map.shape[-2],
                            support_feature_map.shape[-1],
                        ),
                        mode="nearest",
                    )
                    [:, 0]
                    .bool()
                )

            valid = (
                (~mask)
                .to(
                    support_feature_map.dtype
                )
                [:, None]
            )

            valid_count = (
                valid.sum(
                    dim=(2, 3)
                )
                .clamp_min(1.0)
            )

            pooled = (
                (
                    support_feature_map
                    *
                    valid
                )
                .sum(
                    dim=(2, 3)
                )
                /
                valid_count
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
print("STEP 5 : MASK-AWARE SUPPORT PROTOTYPE ENCODER DEFINED")
print("=" * 70)
