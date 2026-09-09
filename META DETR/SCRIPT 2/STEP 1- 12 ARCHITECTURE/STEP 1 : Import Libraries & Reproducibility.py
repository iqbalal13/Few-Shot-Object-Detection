# ==========================================================
# STEP 1 : Import Libraries & Reproducibility
# FINAL CLEAN NOTEBOOK
# ==========================================================

import os
import math
import copy
import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchvision.models import (
    resnet101,
    ResNet101_Weights
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("STEP 1 : IMPORTS & REPRODUCIBILITY READY")
print("=" * 70)

print("PyTorch     :", torch.__version__)
print("TorchVision :", torchvision.__version__)
print("Device      :", DEVICE)
print("Seed        :", SEED)

if torch.cuda.is_available():
    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )

print("=" * 70)
