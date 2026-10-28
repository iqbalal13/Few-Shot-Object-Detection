
# ============================================================
# STEP 11 — AUGMENTED TRAIN / UNCHANGED COCO-VAL
# ============================================================

import os
import random
import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance
from pycocotools.coco import COCO
from torch.utils.data import Dataset


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

        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.flip_probability = float(
            flip_probability
        )

        assert 0 <= self.flip_probability <= 1

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
        image_id = int(
            self.image_ids[int(index)]
        )

        info = self.coco.loadImgs(
            [image_id]
        )[0]

        image_path = os.path.join(
            self.image_root,
            info["file_name"]
        )

        with Image.open(image_path) as fh:
            image = fh.convert("RGB")

        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            iscrowd=False
        )

        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for a in anns:
            x, y, w, h = a["bbox"]

            if w <= 1 or h <= 1:
                continue

            cid = int(a["category_id"])

            if cid not in self.cat_id_to_contiguous:
                continue

            boxes.append([
                x, y, x + w, y + h
            ])

            labels.append(
                self.cat_id_to_contiguous[cid]
            )

        boxes = np.asarray(
            boxes, dtype=np.float32
        ).reshape(-1, 4)

        labels = np.asarray(
            labels, dtype=np.int64
        )

        if self.augment:

            # 1. Random horizontal flip.
            if random.random() < self.flip_probability:
                old_w = float(image.width)

                image = ImageOps.mirror(image)

                if len(boxes):
                    x1 = boxes[:, 0].copy()
                    x2 = boxes[:, 2].copy()

                    boxes[:, 0] = old_w - x2
                    boxes[:, 2] = old_w - x1

            # 2. Color jitter.
            jitter = float(
                CONFIG["train_color_jitter"]
            )

            for cls in (
                ImageEnhance.Brightness,
                ImageEnhance.Contrast,
                ImageEnhance.Color
            ):
                image = cls(image).enhance(
                    random.uniform(
                        1 - jitter,
                        1 + jitter
                    )
                )

            # 3. Scale / translation.
            (
                image_tensor,
                new_boxes,
                meta,
                keep
            ) = preprocess_train_augmented(
                image,
                boxes,
                input_size=self.input_size
            )

            # 4. Apply same filtering to boxes/labels.
            new_boxes = new_boxes[keep]
            labels = labels[keep]

        else:
            (
                image_tensor,
                new_boxes,
                meta
            ) = preprocess_image_and_boxes(
                image,
                boxes,
                input_size=self.input_size
            )

        target = {
            "boxes": torch.as_tensor(
                new_boxes,
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
            )
        }

        assert (
            len(target["boxes"])
            == len(target["labels"])
        )

        return image_tensor, target


train_dataset = CenterNetCOCODataset(
    CONFIG["train_images"],
    CONFIG["train_annotations"],
    input_size=CONFIG["input_size"],
    augment=True,
    flip_probability=CONFIG["train_horizontal_flip_p"]
)

val_dataset = CenterNetCOCODataset(
    CONFIG["val_images"],
    CONFIG["val_annotations"],
    input_size=CONFIG["input_size"],
    augment=False,
    flip_probability=0.0
)

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert train_dataset.augment
assert not val_dataset.augment

assert (
    train_dataset.cat_ids
    == val_dataset.cat_ids
)

im, t = val_dataset[0]

assert im.shape == (3, 640, 640)
assert t["boxes"].shape[1] == 4

print(
    "STEP 11 PASS — train augmented / "
    "validation deterministic"
)
