# ==========================================================
# STEP 6 : Query Feature Encoder
#          + Mask-Aware 2D Sine Position Encoding
# ==========================================================

class PositionEmbeddingSine2D(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=CONFIG[
            "hidden_dim"
        ],
        temperature=CONFIG[
            "position_temperature"
        ],
        normalize=CONFIG[
            "position_normalize"
        ],
        scale=CONFIG[
            "position_scale"
        ],
    ):
        super().__init__()

        if hidden_dim % 4 != 0:
            raise ValueError(
                "hidden_dim must be "
                "divisible by 4."
            )

        self.num_pos_feats = (
            hidden_dim // 2
        )

        self.temperature = (
            float(temperature)
        )

        self.normalize = bool(
            normalize
        )

        self.scale = float(scale)

    def forward(
        self,
        feature_map,
        padding_mask=None
    ):

        if feature_map.dim() != 4:
            raise ValueError(
                "feature_map must be "
                "[B,C,H,W]."
            )

        B, _, H, W = (
            feature_map.shape
        )

        device = (
            feature_map.device
        )

        # False = valid
        # True  = padding

        if padding_mask is None:

            padding_mask = torch.zeros(
                (B, H, W),
                dtype=torch.bool,
                device=device
            )

        else:

            padding_mask = (
                padding_mask
                .to(device)
                .bool()
            )

            if (
                padding_mask.shape[-2:]
                != (H, W)
            ):

                padding_mask = (
                    F.interpolate(
                        padding_mask[
                            :, None
                        ].float(),
                        size=(H, W),
                        mode="nearest",
                    )
                    [:, 0]
                    .bool()
                )

        not_mask = (
            ~padding_mask
        )

        y_embed = (
            not_mask
            .cumsum(
                dim=1,
                dtype=torch.float32
            )
        )

        x_embed = (
            not_mask
            .cumsum(
                dim=2,
                dtype=torch.float32
            )
        )

        if self.normalize:

            eps = 1e-6

            y_embed = (
                y_embed
                /
                (
                    y_embed[
                        :, -1:, :
                    ]
                    + eps
                )
                * self.scale
            )

            x_embed = (
                x_embed
                /
                (
                    x_embed[
                        :, :, -1:
                    ]
                    + eps
                )
                * self.scale
            )

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=device
        )

        dim_t = (
            self.temperature
            **
            (
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
        )

        pos_x = (
            x_embed[..., None]
            /
            dim_t
        )

        pos_y = (
            y_embed[..., None]
            /
            dim_t
        )

        pos_x = torch.stack(
            (
                pos_x[..., 0::2].sin(),
                pos_x[..., 1::2].cos(),
            ),
            dim=-1
        ).flatten(3)

        pos_y = torch.stack(
            (
                pos_y[..., 0::2].sin(),
                pos_y[..., 1::2].cos(),
            ),
            dim=-1
        ).flatten(3)

        # [B,H,W,D]
        pos = torch.cat(
            (
                pos_y,
                pos_x
            ),
            dim=-1
        )

        # [B,H*W,D]
        pos = pos.flatten(
            1,
            2
        )

        mask_flat = (
            padding_mask.flatten(1)
        )

        return (
            pos,
            mask_flat
        )


# ==========================================================
# QUERY FEATURE ENCODER
# ==========================================================

class QueryFeatureEncoder(
    nn.Module
):

    def __init__(
        self,
        in_channels=CONFIG[
            "backbone_out_channels"
        ],
        hidden_dim=CONFIG[
            "hidden_dim"
        ],
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=1
        )

        self.position_embedding = (
            PositionEmbeddingSine2D(
                hidden_dim=hidden_dim
            )
        )

    def forward(
        self,
        query_feature_map,
        padding_mask=None
    ):

        projected = (
            self.input_proj(
                query_feature_map
            )
        )

        _, _, H, W = (
            projected.shape
        )

        query_tokens = (
            projected
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        (
            query_pos,
            mask_flat

        ) = self.position_embedding(

            projected,

            padding_mask=
                padding_mask
        )

        if (
            query_tokens.shape
            != query_pos.shape
        ):
            raise RuntimeError(
                "Query token and position "
                "shapes differ."
            )

        return (
            query_tokens,
            query_pos,
            mask_flat,
            (H, W),
        )


print("=" * 70)
print("STEP 6 : QUERY ENCODER + 2D POSITION DEFINED")
print("=" * 70)
