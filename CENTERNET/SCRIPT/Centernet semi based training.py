# =========================================================
# SEMI BASE TRAINING
# MODIFIED CENTERNET FSOD
# FIXED VERSION
# =========================================================

!pip install pycocotools -q

# =========================================================
# IMPORT
# =========================================================

import os
import cv2
import math
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
# GAUSSIAN UTILS
# =========================================================

def gaussian2D(shape, sigma=1):

    m, n = [(ss - 1.) / 2. for ss in shape]

    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_gaussian(heatmap, center, radius):

    diameter = 2 * radius + 1

    gaussian = gaussian2D(
        (diameter, diameter),
        sigma=diameter / 6
    )

    x, y = center

    height, width = heatmap.shape[0:2]

    left = min(x, radius)
    right = min(width - x, radius + 1)

    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    masked_heatmap = heatmap[
        y-top:y+bottom,
        x-left:x+right
    ]

    masked_gaussian = gaussian[
        radius-top:radius+bottom,
        radius-left:radius+right
    ]

    if (
        min(masked_gaussian.shape) > 0
        and min(masked_heatmap.shape) > 0
    ):
        np.maximum(
            masked_heatmap,
            masked_gaussian,
            out=masked_heatmap
        )


# =========================================================
# COCO DATASET
# =========================================================

class SemiCOCODataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        max_images=5000,
        image_size=512,
        num_classes=80
    ):

        self.image_dir = image_dir

        self.image_size = image_size

        self.output_size = 32

        self.num_classes = num_classes

        self.coco = COCO(annotation_file)

        self.image_ids = list(
            self.coco.imgs.keys()
        )[:max_images]

        # =====================================================
        # CATEGORY MAPPING
        # =====================================================

        self.cat_ids = sorted(
            self.coco.getCatIds()
        )

        self.cat_id_to_label = {
            cat_id: idx
            for idx, cat_id in enumerate(self.cat_ids)
        }

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

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image).permute(2,0,1)

        # =====================================================
        # TARGET
        # =====================================================

        heatmap = np.zeros(
            (
                self.num_classes,
                self.output_size,
                self.output_size
            ),
            dtype=np.float32
        )

        wh = np.zeros(
            (
                2,
                self.output_size,
                self.output_size
            ),
            dtype=np.float32
        )

        offset = np.zeros(
            (
                2,
                self.output_size,
                self.output_size
            ),
            dtype=np.float32
        )

        # =====================================================
        # LOAD ANNOTATIONS
        # =====================================================

        ann_ids = self.coco.getAnnIds(
            imgIds=image_id
        )

        anns = self.coco.loadAnns(ann_ids)

        # =====================================================
        # LOOP OBJECTS
        # =====================================================

        for ann in anns:

            bbox = ann["bbox"]

            category_id = self.cat_id_to_label[
                ann["category_id"]
            ]

            x, y, w, h = bbox

            # resize bbox
            x = x * self.output_size / original_w
            y = y * self.output_size / original_h
            w = w * self.output_size / original_w
            h = h * self.output_size / original_h

            center_x = x + w / 2
            center_y = y + h / 2

            center_int_x = int(center_x)
            center_int_y = int(center_y)

            # skip invalid
            if (
                center_int_x >= self.output_size
                or center_int_y >= self.output_size
            ):
                continue

            radius = max(
                1,
                int(min(w, h) / 2)
            )

            # draw heatmap
            draw_gaussian(
                heatmap[category_id],
                (center_int_x, center_int_y),
                radius
            )

            # width-height
            wh[0, center_int_y, center_int_x] = w
            wh[1, center_int_y, center_int_x] = h

            # offset
            offset[0, center_int_y, center_int_x] = (
                center_x - center_int_x
            )

            offset[1, center_int_y, center_int_x] = (
                center_y - center_int_y
            )

        heatmap = torch.tensor(heatmap)

        wh = torch.tensor(wh)

        offset = torch.tensor(offset)

        return image, heatmap, wh, offset


# =========================================================
# TRANSFORMER
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
# MODEL
# =========================================================

class ModifiedCenterNet(nn.Module):

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

        self.reduce = nn.Conv2d(
            1024,
            256,
            1
        )

        self.transformer = VanillaTransformerEncoder()

        self.head = CenterNetHead(
            in_channels=256,
            num_classes=80
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.reduce(x)

        x = self.transformer(x)

        outputs = self.head(x)

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

        neg_weights = torch.pow(
            1 - target,
            4
        )

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

dataset = SemiCOCODataset(
    image_dir="/content/train2017",
    annotation_file="/content/annotations/instances_train2017.json",
    max_images=5000
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

model = ModifiedCenterNet().to(device)

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
# TRAINING
# =========================================================

epochs = 5

print("\nMulai semi base training...\n")

for epoch in range(epochs):

    total_epoch_loss = 0

    model.train()

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

    print(f"Total Loss: {total_epoch_loss:.4f}")

    print("---------------------------------")


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "semi_base_training.pth"
)

print("\n===================================")
print("SEMI BASE TRAINING BERHASIL!")
print("MODEL BERHASIL DISIMPAN")
print("===================================")
