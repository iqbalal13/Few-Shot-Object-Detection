# ============================================================
# STEP 18 — RESNET-101 BACKBONE
# ImageNet-pretrained backbone for CenterNet
# ============================================================

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

print("=" * 70)
print("STEP 18 — BUILD RESNET-101 BACKBONE")
print("=" * 70)


class ResNet101Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            ResNet101_Weights.DEFAULT
            if pretrained
            else None
        )

        resnet = resnet101(
            weights=weights
        )

        # Keep convolutional feature extractor only.
        # Remove:
        # - avgpool
        # - fully connected classification layer
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.out_channels = 2048
        self.output_stride = 32

    def forward(self, x):
        return self.features(x)


backbone = ResNet101Backbone(
    pretrained=CONFIG["backbone_pretrained"]
).to(device)

total_params = sum(
    p.numel()
    for p in backbone.parameters()
)

trainable_params = sum(
    p.numel()
    for p in backbone.parameters()
    if p.requires_grad
)

print("Backbone             : ResNet-101")
print("Initial weights      : ImageNet pretrained")
print("Output channels      :", backbone.out_channels)
print("Backbone stride      :", backbone.output_stride)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

assert backbone.out_channels == 2048
assert backbone.output_stride == 32

print("\nSTEP 18 PASSED")
