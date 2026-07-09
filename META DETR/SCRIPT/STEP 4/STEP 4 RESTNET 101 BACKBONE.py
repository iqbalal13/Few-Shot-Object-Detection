# ==========================================================
# STEP 4 : Build Shared ResNet-101 Backbone
# ==========================================================

import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet-101
resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

print(resnet101)
