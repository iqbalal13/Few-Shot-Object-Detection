# ==========================================================
# STEP 19 : COCO Base Dataset
# ==========================================================

from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class COCOBaseDataset(Dataset):

    def __init__(self, coco, image_dir, transform=None):

        self.coco = coco
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(coco.imgs.keys())

        # Mapping COCO category_id -> class index (0-79)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {
            cat_id: idx
            for idx, cat_id in enumerate(cat_ids)
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]

        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            cx = x + (w / 2)
            cy = y + (h / 2)

            boxes.append([cx, cy, w, h])

            # gunakan class index 0-79
            labels.append(
                self.cat2label[ann["category_id"]]
            )

        if self.transform:
            image = self.transform(image)

        if len(boxes) == 0:

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        else:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.long
            )

        target = {

            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                image_id,
                dtype=torch.long
            )

        }

        return image, target
