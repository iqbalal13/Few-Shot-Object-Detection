# ============================================================
# STEP 1 — ENVIRONMENT SETUP
# CenterNet + ResNet-101 + MS COCO 2017
# ============================================================

!pip -q install pycocotools

import os
import sys
import random
import json
import time
import math
import shutil
import zipfile
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights

from pycocotools.coco import COCO

print("=" * 60)
print("ENVIRONMENT READY")
print("=" * 60)
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("NumPy       :", np.__version__)
print("pycocotools : OK")
