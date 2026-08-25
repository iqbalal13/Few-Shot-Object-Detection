# ==========================================================
# STEP 4 : Shared ResNet-101 Backbone (FINAL)
# ==========================================================

import torch
import torch.nn as nn
import torchvision.models as models


class SharedResNet101Backbone(nn.Module):

    def __init__(self, hidden_dim=256):
        super().__init__()

        # --------------------------------------------------
        # ImageNet pretrained ResNet-101
        # --------------------------------------------------
        resnet = models.resnet101(
            weights=models.ResNet101_Weights.DEFAULT
        )

        # --------------------------------------------------
        # Remove Average Pooling + FC classification layer
        #
        # Output:
        # [B, 2048, H/32, W/32]
        # --------------------------------------------------
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # --------------------------------------------------
        # DETR input projection
        #
        # 2048 -> hidden_dim (256)
        # --------------------------------------------------
        self.input_projection = nn.Conv2d(
            in_channels=2048,
            out_channels=hidden_dim,
            kernel_size=1
        )

    def forward(self, images):

        features = self.feature_extractor(images)

        features = self.input_projection(features)

        return features


# ==========================================================
# Initialize ONE shared backbone
# ==========================================================

backbone = SharedResNet101Backbone(
    hidden_dim=CONFIG["hidden_dim"]
).to(CONFIG["device"])


print("=" * 70)
print("STEP 4 : Shared ResNet-101 Backbone Ready")
print("=" * 70)

print(backbone.__class__.__name__)
