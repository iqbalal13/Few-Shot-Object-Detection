# ==========================================================
# STEP 13 : Refactor Backbone
# ==========================================================

import torch.nn as nn

class Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet101(
            weights=models.ResNet101_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Input Projection
        self.input_projection = nn.Conv2d(
            2048,
            256,
            kernel_size=1
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.input_projection(x)

        return x
