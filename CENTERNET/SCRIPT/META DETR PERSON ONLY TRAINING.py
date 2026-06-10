# =========================================================
# FINAL LOCKED TRAINING PIPELINE
# ADAPTED DETR RESNET101
# META-DETR INSPIRED
# PERSON ONLY
# 5000 IMAGES
# =========================================================

# =========================================================
# INSTALL
# =========================================================

!pip install transformers pycocotools timm accelerate -q

# =========================================================
# IMPORT
# =========================================================

import os
import torch

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pycocotools.coco import COCO

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor
)

# =========================================================
# CUDA MEMORY OPTIMIZATION
# =========================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()

# =========================================================
# DEVICE
# =========================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

print("DEVICE:", device)

# =========================================================
# DOWNLOAD COCO
# =========================================================

!mkdir -p /content/coco

%cd /content/coco

# =====================================
# DOWNLOAD TRAIN2017
# =====================================

if not os.path.exists(
    "/content/coco/train2017"
):

    !wget -q http://images.cocodataset.org/zips/train2017.zip

    !unzip -q train2017.zip

# =====================================
# DOWNLOAD ANNOTATIONS
# =====================================

if not os.path.exists(
    "/content/coco/annotations"
):

    !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    !unzip -q annotations_trainval2017.zip

print("COCO READY")

# =========================================================
# PATH
# =========================================================

IMAGE_DIR = "/content/coco/train2017"

ANN_FILE = "/content/coco/annotations/instances_train2017.json"

# =========================================================
# DETR IMAGE PROCESSOR
# =========================================================

processor = DetrImageProcessor.from_pretrained(

    "facebook/detr-resnet-101",

    size={

        "shortest_edge": 300,
        "longest_edge": 500

    }

)

# =========================================================
# COCO PERSON DATASET
# =========================================================

class COCO_Person_Dataset(Dataset):

    def __init__(

        self,
        image_dir,
        ann_file

    ):

        self.image_dir = image_dir

        self.coco = COCO(
            ann_file
        )

        # =====================================
        # PERSON CATEGORY
        # =====================================

        self.person_cat = 1

        # =====================================
        # PERSON IMAGE IDS
        # =====================================

        self.image_ids = self.coco.getImgIds(

            catIds=[self.person_cat]

        )

        print(
            "TOTAL PERSON IMAGES:",
            len(self.image_ids)
        )

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, idx):

        # =====================================
        # IMAGE ID
        # =====================================

        image_id = self.image_ids[idx]

        # =====================================
        # IMAGE INFO
        # =====================================

        image_info = self.coco.loadImgs(
            image_id
        )[0]

        image_path = os.path.join(

            self.image_dir,
            image_info["file_name"]

        )

        # =====================================
        # LOAD IMAGE
        # =====================================

        image = Image.open(
            image_path
        ).convert("RGB")

        # =====================================
        # PERSON ANNOTATIONS
        # =====================================

        ann_ids = self.coco.getAnnIds(

            imgIds=image_id,
            catIds=[self.person_cat]

        )

        anns = self.coco.loadAnns(
            ann_ids
        )

        # =====================================
        # DETR ENCODING
        # =====================================

        encoding = processor(

            images=image,

            annotations={

                "image_id": image_id,
                "annotations": anns

            },

            return_tensors="pt"

        )

        pixel_values = encoding[
            "pixel_values"
        ].squeeze()

        labels = encoding[
            "labels"
        ][0]

        return {

            "pixel_values": pixel_values,

            "labels": labels

        }

# =========================================================
# DATASET
# =========================================================

dataset = COCO_Person_Dataset(

    IMAGE_DIR,
    ANN_FILE

)

# =========================================================
# LIMIT TRAINING TO 5000 IMAGES
# =========================================================

dataset.image_ids = dataset.image_ids[:5000]

print(
    "FINAL TRAINING IMAGES:",
    len(dataset.image_ids)
)

# =========================================================
# COLLATE FUNCTION
# =========================================================

def collate_fn(batch):

    pixel_values = [

        item["pixel_values"]
        for item in batch

    ]

    labels = [

        item["labels"]
        for item in batch

    ]

    # =====================================
    # DETR PADDING
    # =====================================

    encoding = processor.pad(

        pixel_values,

        return_tensors="pt"

    )

    return {

        "pixel_values": encoding[
            "pixel_values"
        ],

        "pixel_mask": encoding[
            "pixel_mask"
        ],

        "labels": labels

    }

# =========================================================
# DATALOADER
# =========================================================

dataloader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=True,

    collate_fn=collate_fn

)

# =========================================================
# LOAD ORIGINAL DETR
# =========================================================

model = DetrForObjectDetection.from_pretrained(

    "facebook/detr-resnet-101"

)

model.to(device)

print("MODEL READY")

# =========================================================
# FREEZE BACKBONE
# =========================================================

for param in model.model.backbone.parameters():

    param.requires_grad = False

print("BACKBONE FROZEN")

# =========================================================
# FREEZE ENCODER
# =========================================================

for param in model.model.encoder.parameters():

    param.requires_grad = False

print("ENCODER FROZEN")

# =========================================================
# DECODER TRAINABLE
# =========================================================

for param in model.model.decoder.parameters():

    param.requires_grad = True

print("DECODER TRAINABLE")

# =========================================================
# DETECTION HEAD TRAINABLE
# =========================================================

for param in model.class_labels_classifier.parameters():

    param.requires_grad = True

for param in model.bbox_predictor.parameters():

    param.requires_grad = True

print("DETECTION HEAD TRAINABLE")

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=1e-5

)

# =========================================================
# MIXED PRECISION
# =========================================================

scaler = torch.cuda.amp.GradScaler()

# =========================================================
# TRAINING LOOP
# =========================================================

epochs = 3

model.train()

for epoch in range(epochs):

    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):

        # =====================================
        # INPUT
        # =====================================

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        pixel_mask = batch[
            "pixel_mask"
        ].to(device)

        labels = []

        for target in batch["labels"]:

            target = {

                k: v.to(device)
                for k, v in target.items()

            }

            labels.append(target)

        # =====================================
        # MIXED PRECISION
        # =====================================

        with torch.cuda.amp.autocast():

            outputs = model(

                pixel_values=pixel_values,

                pixel_mask=pixel_mask,

                labels=labels

            )

            loss = outputs.loss

        # =====================================
        # BACKPROP
        # =====================================

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        # =====================================
        # LOGGING
        # =====================================

        if batch_idx % 100 == 0:

            print(

                f"Epoch {epoch+1} | "
                f"Batch {batch_idx} | "
                f"Loss {loss.item():.4f}"

            )

        # =====================================
        # CLEAR CACHE
        # =====================================

        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)

    print(

        f"\nEpoch {epoch+1} COMPLETE | "
        f"Average Loss: {avg_loss:.4f}\n"

    )

print("TRAINING COMPLETE")

# =========================================================
# SAVE MODEL
# =========================================================

SAVE_PATH = "/content/adapted_detr_resnet101_5k.pth"

torch.save(

    model.state_dict(),

    SAVE_PATH

)

print("MODEL SAVED")

print("SAVE PATH:", SAVE_PATH)

# =========================================================
# FINAL SUMMARY
# =========================================================

"""


MODEL:
✅ DETR ResNet101

TRAINING:
✅ 5000 person images
✅ 3 epoch
✅ mixed precision
✅ batch size 1

FREEZE:
✅ backbone frozen
✅ encoder frozen

TRAINABLE:
✅ decoder
✅ detection head

FEATURES:
✅ Hungarian matching
✅ GIoU loss
✅ bbox regression
✅ transformer decoder
✅ object queries

THESIS POSITIONING:
✅ Adapted DETR
✅ Meta-DETR Inspired
✅ FSOD framing
✅ CCTV transfer learning
