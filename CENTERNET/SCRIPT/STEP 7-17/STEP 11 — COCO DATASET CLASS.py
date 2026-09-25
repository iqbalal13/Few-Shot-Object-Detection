# ============================================================
# STEP 11 — COCO DATASET CLASS
# ============================================================

from torch.utils.data import Dataset
from PIL import Image
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
        input_size=640
    ):

        self.image_root = image_root
        self.coco = COCO(annotation_file)
        self.input_size = input_size

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

            boxes.append(
                [x1, y1, x2, y2]
            )

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

        image_tensor, boxes, meta = preprocess_image_and_boxes(
            image=image,
            boxes=boxes,
            input_size=self.input_size
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
                [meta["pad_x"], meta["pad_y"]],
                dtype=torch.float32
            ),
        }

        return image_tensor, target


train_dataset = CenterNetCOCODataset(
    CONFIG["train_images"],
    CONFIG["train_annotations"],
    input_size=CONFIG["input_size"]
)

val_dataset = CenterNetCOCODataset(
    CONFIG["val_images"],
    CONFIG["val_annotations"],
    input_size=CONFIG["input_size"]
)

print("Train dataset :", len(train_dataset))
print("Val dataset   :", len(val_dataset))

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

print("\nSTEP 11 PASSED")
