
# STEP 20 — Gate-zero residual C4/C3 fusion
import torch
import torch.nn as nn

class CenterNetNeck(nn.Module):
    def __init__(
        self,
        in_channels=2048,
        hidden_channels=256
    ):
        super().__init__()

        def deconv(cin):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    cin,
                    hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )

        self.deconv1 = deconv(in_channels)
        self.deconv2 = deconv(hidden_channels)
        self.deconv3 = deconv(hidden_channels)

        # New B2 lateral paths
        self.lateral_c4 = nn.Conv2d(
            1024, hidden_channels, kernel_size=1
        )
        self.lateral_c3 = nn.Conv2d(
            512, hidden_channels, kernel_size=1
        )

        nn.init.kaiming_normal_(
            self.lateral_c4.weight,
            mode="fan_out"
        )
        nn.init.kaiming_normal_(
            self.lateral_c3.weight,
            mode="fan_out"
        )
        nn.init.zeros_(self.lateral_c4.bias)
        nn.init.zeros_(self.lateral_c3.bias)

        # Zero-initialized trainable scalar gates.
        self.gate_c4 = nn.Parameter(
            torch.tensor(float(CONFIG["b2_gate_init"]))
        )
        self.gate_c3 = nn.Parameter(
            torch.tensor(float(CONFIG["b2_gate_init"]))
        )

        self.out_channels = hidden_channels

    def forward(self, features):
        assert isinstance(features, dict)

        x = self.deconv1(features["c5"])

        x = (
            x
            + self.gate_c4
            * self.lateral_c4(features["c4"])
        )

        x = self.deconv2(x)

        x = (
            x
            + self.gate_c3
            * self.lateral_c3(features["c3"])
        )

        return self.deconv3(x)


centernet_neck = CenterNetNeck().to(device).eval()

with torch.inference_mode():
    samples = {
        "c3": torch.randn(
            1, 512, 80, 80, device=device
        ),
        "c4": torch.randn(
            1, 1024, 40, 40, device=device
        ),
        "c5": torch.randn(
            1, 2048, 20, 20, device=device
        )
    }

    out = centernet_neck(samples)
    assert out.shape == (1, 256, 160, 160)

    old_path = centernet_neck.deconv3(
        centernet_neck.deconv2(
            centernet_neck.deconv1(samples["c5"])
        )
    )

    assert torch.allclose(
        out, old_path,
        atol=1e-6,
        rtol=1e-6
    )

print("STEP 20 PASS: zero-gate path equals original deconv")

del out, old_path, samples

if torch.cuda.is_available():
    torch.cuda.empty_cache()
