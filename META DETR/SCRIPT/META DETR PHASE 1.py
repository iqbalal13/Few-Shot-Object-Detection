# =========================================================
# PHASE 2
# REAL COCO PERSON META-LEARNING
# ADAPTED META-DETR
# =========================================================

# =========================================================
# INSTALL
# =========================================================

!pip install transformers pycocotools timm accelerate -q

# =========================================================
# IMPORT
# =========================================================

import os
import random
import json

import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image

from torch.utils.data import Dataset

from pycocotools.coco import COCO

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

# =========================================================
# DOWNLOAD COCO
# =========================================================

!mkdir -p /content/coco

%cd /content/coco

if not os.path.exists(
    "/content/coco/train2017"
):

    !wget -q http://images.cocodataset.org/zips/train2017.zip
    !unzip -q train2017.zip

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
# TRANSFORM
# =========================================================

transform = T.Compose([

    T.Resize((800, 800)),

    T.ToTensor()

])

# =========================================================
# COCO PERSON DATASET
# =========================================================

class COCO_Person_FSOD(Dataset):

    def __init__(

        self,
        image_dir,
        ann_file,
        transform=None

    ):

        self.image_dir = image_dir

        self.transform = transform

        # =====================================
        # LOAD COCO
        # =====================================

        self.coco = COCO(ann_file)

        # =====================================
        # PERSON CATEGORY ID
        # COCO PERSON = 1
        # =====================================

        self.person_cat = 1

        # =====================================
        # GET PERSON IMAGE IDS
        # =====================================

        self.image_ids = self.coco.getImgIds(

            catIds=[self.person_cat]

        )

        print(
            "PERSON IMAGES:",
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

        width, height = image.size

        # =====================================
        # LOAD PERSON ANNOTATIONS
        # =====================================

        ann_ids = self.coco.getAnnIds(

            imgIds=image_id,
            catIds=[self.person_cat]

        )

        anns = self.coco.loadAnns(
            ann_ids
        )

        # =====================================
        # BBOX LIST
        # =====================================

        boxes = []

        for ann in anns:

            bbox = ann["bbox"]

            boxes.append(bbox)

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        # =====================================
        # TRANSFORM IMAGE
        # =====================================

        if self.transform:

            image = self.transform(image)

        return {

            "image": image,
            "boxes": boxes,
            "image_id": image_id

        }

# =========================================================
# CREATE DATASET
# =========================================================

dataset = COCO_Person_FSOD(

    IMAGE_DIR,
    ANN_FILE,
    transform=transform

)

print("DATASET READY")

# =========================================================
# EPISODIC SAMPLER
# =========================================================

class PersonEpisodeSampler:

    def __init__(

        self,
        dataset,
        k_shot=5

    ):

        self.dataset = dataset

        self.k_shot = k_shot

    def sample_episode(self):

        support_set = []

        # =====================================
        # SUPPORT SET
        # =====================================

        for _ in range(self.k_shot):

            idx = random.randint(

                0,
                len(self.dataset)-1

            )

            sample = self.dataset[idx]

            support_set.append(sample)

        # =====================================
        # QUERY IMAGE
        # =====================================

        idx = random.randint(

            0,
            len(self.dataset)-1

        )

        query_sample = self.dataset[idx]

        return support_set, query_sample

# =========================================================
# SUPPORT ENCODER
# =========================================================

class SupportEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):

        pooled = self.pool(features)

        pooled = pooled.flatten(1)

        return pooled

# =========================================================
# PROTOTYPE ATTENTION
# =========================================================

class PrototypeAttention(nn.Module):

    def __init__(

        self,
        hidden_dim=256

    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(

            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True

        )

    def forward(

        self,
        query_features,
        prototype

    ):

        prototype = prototype.unsqueeze(1)

        fused, _ = self.attn(

            query_features,
            prototype,
            prototype

        )

        return fused

# =========================================================
# BUILD PROTOTYPE
# =========================================================

def build_prototype(

    support_embeddings

):

    prototype = support_embeddings.mean(
        dim=0
    )

    return prototype

# =========================================================
# ADAPTED META DETR
# =========================================================

class AdaptedMetaDETR(nn.Module):

    def __init__(

        self,
        hidden_dim=256

    ):

        super().__init__()

        # =====================================
        # DETR RESNET-101
        # =====================================

        self.detr = DetrForObjectDetection.from_pretrained(

            "facebook/detr-resnet-101"

        )

        self.support_encoder = SupportEncoder()

        self.prototype_attention = PrototypeAttention()

        # PERSON CLASS ONLY
        self.class_head = nn.Linear(

            hidden_dim,
            2

        )

        self.bbox_head = nn.Linear(

            hidden_dim,
            4

        )

    def extract_query_features(

        self,
        query_image

    ):

        outputs = self.detr.model(

            pixel_values=query_image

        )

        return outputs.last_hidden_state

    def forward(

        self,
        query_image,
        support_embeddings

    ):

        # =====================================
        # QUERY FEATURES
        # =====================================

        query_features = self.extract_query_features(

            query_image

        )

        # =====================================
        # PROTOTYPE
        # =====================================

        prototype = build_prototype(

            support_embeddings

        )

        prototype = prototype.unsqueeze(0)

        # =====================================
        # SUPPORT QUERY ATTENTION
        # =====================================

        fused_features = self.prototype_attention(

            query_features,
            prototype

        )

        # =====================================
        # PREDICTION
        # =====================================

        logits = self.class_head(
            fused_features
        )

        boxes = self.bbox_head(
            fused_features
        )

        return logits, boxes

# =========================================================
# CREATE MODEL
# =========================================================

model = AdaptedMetaDETR()

model.to(device)

print("MODEL READY")

# =========================================================
# CREATE SAMPLER
# =========================================================

sampler = PersonEpisodeSampler(

    dataset,
    k_shot=5

)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),
    lr=1e-4

)

# =========================================================
# TRAINING LOOP
# =========================================================

epochs = 3

for epoch in range(epochs):

    # =====================================
    # SAMPLE EPISODE
    # =====================================

    support_set, query_sample = (

        sampler.sample_episode()

    )

    # =====================================
    # BUILD SUPPORT EMBEDDINGS
    # =====================================

    support_embeddings = []

    for sample in support_set:

        embedding = torch.randn(
            1,
            256
        )

        support_embeddings.append(
            embedding
        )

    support_embeddings = torch.cat(

        support_embeddings,
        dim=0

    ).to(device)

    # =====================================
    # QUERY IMAGE
    # =====================================

    query_image = query_sample["image"]

    query_image = query_image.unsqueeze(0)

    query_image = query_image.to(device)

    # =====================================
    # FORWARD
    # =====================================

    logits, pred_boxes = model(

        query_image,
        support_embeddings

    )

    # =====================================
    # REAL BBOX FROM COCO
    # =====================================

    gt_boxes = query_sample["boxes"]

    gt_boxes = gt_boxes.to(device)

    # =====================================
    # TEMPORARY LOSS
    # =====================================

    classification_loss = logits.mean()

    bbox_loss = pred_boxes.mean()

    loss = classification_loss + bbox_loss

    # =====================================
    # BACKPROP
    # =====================================

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(

        f"Epoch {epoch+1} | Loss {loss.item():.4f}"

    )

print("PHASE 2 COMPLETE")
