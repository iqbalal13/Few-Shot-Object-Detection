# ============================================================
# STEP 11 — COCO DATASET CLASS
# ============================================================

from torch.utils.data import Dataset
from PIL import Image, ImageOps
import random
import numpy as np
import torch
import os

print("=" * 70)
print("STEP 11 — BUILD COCO DATASET CLASS")
print("=" * 70)


class CenterNetCOCODataset(Dataset):

    def __init__(
        self,
        image_root,
        annotation_file,
        input_size=640,
        augment=False,
        flip_probability=0.0
    ):

        self.image_root = image_root
        self.coco = COCO(annotation_file)
        self.input_size = input_size
        self.augment = bool(augment)
        self.flip_probability = float(flip_probability)
        assert 0.0 <= self.flip_probability <= 1.0

        self.image_ids = sorted(
            self.coco.getImgIds()
        )

        self.cat_ids = sorted(
            self.coco.getCatIds()
        )

        self.cat_id_to_contiguous = {
            cat_id: idx
            for idx, cat_id in enumerate(self.cat_ids)
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(
            [image_id]
        )[0]

        image_path = os.path.join(
            self.image_root,
            image_info["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            iscrowd=False
        )

        annotations = self.coco.loadAnns(
            ann_ids
        )

        boxes = []
        labels = []

        for ann in annotations:

            x, y, w, h = ann["bbox"]

            if w <= 1 or h <= 1:
                continue

            cat_id = ann["category_id"]

            if cat_id not in self.cat_id_to_contiguous:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
