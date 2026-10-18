
# ============================================================
# STEP 11 — COCO DATASET CLASS
# BALANCED80K V1.3
# TRAIN-ONLY HORIZONTAL FLIP + BBOX TRANSFORM
# ============================================================

from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pycocotools.coco import COCO

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

        with Image.open(image_path) as img:
            image = img.convert("RGB")

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

            boxes.append([
                x,
                y,
                x + w,
                y + h
            ])

            labels.append(
                self.cat_id_to_contiguous[cat_id]
            )

        boxes = np.asarray(
            boxes,
            dtype=np.float32
        ).reshape(-1, 4)

        labels = np.asarray(
            labels,
            dtype=np.int64
        )

        # Augmentation exclusively for COCO training.
        # Mirror the original pixels and transform every bbox.
        if (
            self.augment
            and random.random() < self.flip_probability
        ):

            original_width = float(image.width)

            image = ImageOps.mirror(image)

            if len(boxes) > 0:

                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()

                boxes[:, 0] = original_width - old_x2
                boxes[:, 2] = original_width - old_x1

        # Step 10: resize + letterbox + normalize
        image_tensor, boxes, meta = (
            preprocess_image_and_boxes(
                image=image,
                boxes=boxes,
                input_size=self.input_size
            )
        )

        target = {
            "boxes": torch.as_tensor(
                boxes,
                dtype=torch.float32
            ),

            "labels": torch.as_tensor(
                labels,
                dtype=torch.long
            ),

            "image_id": torch.tensor(
                image_id,
                dtype=torch.long
            ),

            "original_size": torch.tensor(
                meta["original_size"],
                dtype=torch.long
            ),

            "scale": torch.tensor(
                meta["scale"],
                dtype=torch.float32
            ),

            "pad": torch.tensor(
                [
                    meta["pad_x"],
                    meta["pad_y"]
                ],
                dtype=torch.float32
            )
        }

        return image_tensor, target


# ============================================================
# CREATE TRAIN AND VALIDATION DATASETS
# ============================================================

train_dataset = CenterNetCOCODataset(
    image_root=CONFIG["train_images"],
    annotation_file=CONFIG["train_annotations"],
    input_size=CONFIG["input_size"],
    augment=True,
    flip_probability=CONFIG["train_horizontal_flip_p"]
)

val_dataset = CenterNetCOCODataset(
    image_root=CONFIG["val_images"],
    annotation_file=CONFIG["val_annotations"],
    input_size=CONFIG["input_size"],
    augment=False,
    flip_probability=0.0
)


# ============================================================
# SANITY CHECK
# ============================================================

print("Train dataset :", len(train_dataset))
print("Val dataset   :", len(val_dataset))

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert train_dataset.augment is True
assert val_dataset.augment is False

assert train_dataset.cat_ids == val_dataset.cat_ids

# Verify one dataset item
sample_image, sample_target = val_dataset[0]

assert sample_image.shape == (
    3,
    CONFIG["input_size"],
    CONFIG["input_size"]
)

assert sample_target["boxes"].ndim == 2
assert sample_target["boxes"].shape[1] == 4
assert sample_target["labels"].ndim == 1

assert (
    len(sample_target["boxes"])
    == len(sample_target["labels"])
)

assert "original_size" in sample_target
assert "scale" in sample_target
assert "pad" in sample_target

print("Sample tensor:", tuple(sample_image.shape))
print("Sample boxes :", tuple(sample_target["boxes"].shape))
print("Sample ID    :", sample_target["image_id"].item())

print("\nSTEP 11 PASSED")
