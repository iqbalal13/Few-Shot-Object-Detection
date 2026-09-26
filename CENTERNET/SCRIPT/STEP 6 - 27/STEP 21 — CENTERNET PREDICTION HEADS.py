# ============================================================
# STEP 21 — CENTERNET PREDICTION HEADS
# ============================================================

print("=" * 70)
print("STEP 21 — BUILD CENTERNET PREDICTION HEADS")
print("=" * 70)


class CenterNetHead(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        final_bias=None
    ):
        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=True
            )
        )

        # Small initialization for final layer
        nn.init.normal_(
            self.head[-1].weight,
            std=0.001
        )

        if final_bias is not None:

            nn.init.constant_(
                self.head[-1].bias,
                final_bias
            )

        else:

            nn.init.constant_(
                self.head[-1].bias,
                0
            )

    def forward(self, x):
        return self.head(x)


class CenterNetPredictionHeads(nn.Module):

    def __init__(
        self,
        in_channels=256,
        head_channels=256,
        num_classes=80
    ):
        super().__init__()

        # Center heatmap
        self.heatmap_head = CenterNetHead(
            in_channels,
            head_channels,
            num_classes,

            # Common CenterNet heatmap prior:
            # sigmoid(-2.19) ≈ 0.10
            final_bias=-2.19
        )

        # Width / height
        self.wh_head = CenterNetHead(
            in_channels,
            head_channels,
            2,
            final_bias=0.0
        )

        # Center sub-pixel offset
        self.offset_head = CenterNetHead(
            in_channels,
            head_channels,
            2,
            final_bias=0.0
        )

    def forward(self, x):

        return {
            "heatmap": self.heatmap_head(x),
            "wh": self.wh_head(x),
            "offset": self.offset_head(x)
        }


prediction_heads = CenterNetPredictionHeads(
    in_channels=256,
    head_channels=256,
    num_classes=CONFIG["num_classes"]
).to(device)


# ------------------------------------------------------------
# Head shape sanity test
# ------------------------------------------------------------

dummy_feature = torch.randn(
    1,
    256,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    device=device
)

prediction_heads.eval()

with torch.no_grad():

    head_outputs = prediction_heads(
        dummy_feature
    )

print(
    "Heatmap :",
    head_outputs["heatmap"].shape
)

print(
    "WH      :",
    head_outputs["wh"].shape
)

print(
    "Offset  :",
    head_outputs["offset"].shape
)

assert head_outputs["heatmap"].shape == (
    1,
    80,
    160,
    160
)

assert head_outputs["wh"].shape == (
    1,
    2,
    160,
    160
)

assert head_outputs["offset"].shape == (
    1,
    2,
    160,
    160
)

del dummy_feature
del head_outputs

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSTEP 21 PASSED")
