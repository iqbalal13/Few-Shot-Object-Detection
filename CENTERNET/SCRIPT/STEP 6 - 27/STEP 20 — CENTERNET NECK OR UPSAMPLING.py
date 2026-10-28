# ============================================================
# STEP 20 — CENTERNET NECK / UPSAMPLING
# ============================================================

print("=" * 70)
print("STEP 20 — BUILD CENTERNET UPSAMPLING NECK")
print("=" * 70)


class CenterNetNeck(nn.Module):

    def __init__(
        self,
        in_channels=2048,
        hidden_channels=256
    ):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.out_channels = hidden_channels

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x


centernet_neck = CenterNetNeck(
    in_channels=2048,
    hidden_channels=256
).to(device)


# ------------------------------------------------------------
# Shape sanity test
# ------------------------------------------------------------

dummy_feature = torch.randn(
    1,
    2048,
    20,
    20,
    device=device
)

centernet_neck.eval()

with torch.no_grad():

    neck_output = centernet_neck(
        dummy_feature
    )

print("Input feature :", dummy_feature.shape)
print("Neck output   :", neck_output.shape)

assert tuple(
    neck_output.shape
) == (
    1,
    256,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH
)

del dummy_feature
del neck_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSTEP 20 PASSED")
