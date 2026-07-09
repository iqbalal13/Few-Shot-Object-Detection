# ==========================================================
# STEP 5 : Prototype Feature Extraction
# ==========================================================

import torch
import torch.nn as nn

class PrototypeFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, feature_map):

        prototype = self.gap(feature_map)

        prototype = prototype.flatten(1)

        return prototype
