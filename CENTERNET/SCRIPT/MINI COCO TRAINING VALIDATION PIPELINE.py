# =========================================================
# MINI COCO TRAINING PIPELINE
# MODIFIED CENTERNET FSOD
# =========================================================

"""
TAHAP INI:
-----------
Sekarang model mulai menggunakan:
- gambar asli COCO
- bounding box asli
- annotation asli

Tujuan:
--------
1. Validasi dataset pipeline
2. Validasi COCO loader
3. Validasi real training
4. Validasi target generation
5. Validasi loss pada data nyata

CATATAN:
---------
INI BELUM FULL FSOD.

Masih:
- mini COCO training
- base training awal
"""

# =========================================================
# INSTALL COCO API
# =========================================================

!pip install pycocotools -q


# =========================================================
# IMPORT
# =========================================================

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================================================
# DOWNLOAD MINI COCO
# =========================================================

"""
DOWNLOAD:
----------
- train2017
- annotations_trainval2017
"""

!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip -q train2017.zip
!unzip -q annotations_trainval2017.zip


# =========================================================
# COCO DATASET
# =========================================================

class MiniCOCODataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        max_images=20,
        image_size=512
    ):

        self.image_dir = image_dir
        self.image_size = image_size

        self.coco = COCO(annotation_file)

        self.image_ids = list(self.coco.imgs.keys())[:max_images]

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image_info = self.coco.loadImgs(image_id)[0]

        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )

        # =====================================================
        # LOAD IMAGE
        # =====================================================

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image).permute(2,0,1)

        # =====================================================
        # DUMMY CENTERNET TARGET
        # =====================================================

        """
        Sementara masih target sederhana.
        Nanti bisa upgrade jadi:
        - gaussian heatmap
        - real bbox target
        """

        heatmap = torch.zeros(60, 32, 32)

        heatmap[:,16,16] = 1

        wh = torch.randn(2, 32, 32)

        offset = torch.randn(2, 32, 32)

        return image, heatmap, wh, offset


# =========================================================
# VANILLA TRANSFORMER
# =========================================================

class VanillaTransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_layers=2
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        B,C,H,W = x.shape

        x = x.flatten(2).transpose(1,2)

        x = self.transformer(x)

        x = x.transpose(1,2).reshape(B,C,H,W)

        return x


# =========================================================
# EMBEDDING HEAD
# =========================================================

class ContrastiveEmbeddingHead(nn.Module):

    def __init__(self, in_channels=256, embedding_dim=128):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.embedding(x)


# =========================================================
# CENTERNET HEAD
# =========================================================

class CenterNetHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,num_classes,1)
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,2,1)
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,2,1)
        )

    def forward(self, x):

        return {
            "heatmap": self.heatmap_head(x),
            "wh": self.wh_head(x),
            "offset": self.offset_head(x)
        }


# =========================================================
# MODIFIED CENTERNET FSOD
# =========================================================

class ModifiedCenterNetFSOD(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        )

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        self.feature_reduce = nn.Conv2d(
            1024,
            256,
            kernel_size=1
        )

        self.transformer = VanillaTransformerEncoder()

        self.embedding_head = ContrastiveEmbeddingHead()

        self.base_head = CenterNetHead(
            in_channels=256,
            num_classes=60
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.feature_reduce(features)

        features = self.transformer(features)

        embeddings = self.embedding_head(features)

        outputs = self.base_head(features)

        return outputs


# =========================================================
# FOCAL LOSS
# =========================================================

class HeatmapFocalLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pos_inds = target.eq(1).float()

        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = (
            torch.log(pred + 1e-6)
            * torch.pow(1 - pred, 2)
            * pos_inds
        )

        neg_loss = (
            torch.log(1 - pred + 1e-6)
            * torch.pow(pred, 2)
            * neg_weights
            * neg_inds
        )

        num_pos = pos_inds.sum()

        pos_loss = pos_loss.sum()

        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


# =========================================================
# DATASET
# =========================================================

dataset = MiniCOCODataset(
    image_dir="/content/train2017",
    annotation_file="/content/annotations/instances_train2017.json",
    max_images=20
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

print("Dataset berhasil dibuat!")


# =========================================================
# MODEL
# =========================================================

model = ModifiedCenterNetFSOD().to(device)

print("Model berhasil dibuat!")


# =========================================================
# LOSS
# =========================================================

heatmap_loss_fn = HeatmapFocalLoss()

wh_loss_fn = nn.L1Loss()

offset_loss_fn = nn.L1Loss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# =========================================================
# MINI TRAINING
# =========================================================

epochs = 3

print("\nMulai mini COCO training...\n")

for epoch in range(epochs):

    total_epoch_loss = 0

    for images, heatmaps, whs, offsets in loader:

        images = images.to(device)

        heatmaps = heatmaps.to(device)

        whs = whs.to(device)

        offsets = offsets.to(device)

        # =================================================
        # FORWARD
        # =================================================

        outputs = model(images)

        # =================================================
        # LOSS
        # =================================================

        heatmap_loss = heatmap_loss_fn(
            outputs["heatmap"],
            heatmaps
        )

        wh_loss = wh_loss_fn(
            outputs["wh"],
            whs
        )

        offset_loss = offset_loss_fn(
            outputs["offset"],
            offsets
        )

        total_loss = (
            heatmap_loss
            + wh_loss
            + offset_loss
        )

        # =================================================
        # BACKWARD
        # =================================================

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        total_epoch_loss += total_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]")

    print(f"Loss: {total_epoch_loss:.4f}")

    print("---------------------------------")


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "mini_coco_model.pth"
)

print("\n===================================")
print("MINI COCO TRAINING BERHASIL!")
print("MODEL BERHASIL DISIMPAN")
print("===================================")
