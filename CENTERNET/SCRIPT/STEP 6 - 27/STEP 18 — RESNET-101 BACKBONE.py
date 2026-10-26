
# STEP 18 — ResNet-101 C3/C4/C5
import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = resnet101(
            weights=(
                ResNet101_Weights.DEFAULT
                if pretrained else None
            )
        )

        # Keep old state_dict key names.
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.out_channels = 2048
        self.output_stride = 32

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)

            if i == 5:
                c3 = x
            elif i == 6:
                c4 = x

        return {
            "c3": c3,
            "c4": c4,
            "c5": x
        }

backbone = ResNet101Backbone(
    pretrained=CONFIG["backbone_pretrained"]
).to(device)

print("STEP 18 PASS: ResNet101 features C3/C4/C5")
