# ==========================================================
# STEP 19 : COCO Episode Dataset
# ==========================================================

from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import random

class COCOEpisodeDataset(Dataset):

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

        # --------------------------------------------------
        # Build Class -> Image Index
        # --------------------------------------------------
        self.class_to_images = {}

        for image_id in self.image_ids:

            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            classes = set()

            for ann in anns:
                cls = self.cat2label[ann["category_id"]]
                classes.add(cls)

            for cls in classes:

                if cls not in self.class_to_images:
                    self.class_to_images[cls] = []

                self.class_to_images[cls].append(image_id)

    def __len__(self):
        return len(self.image_ids)

    # ------------------------------------------------------
    # Load Single Image + Target
    # ------------------------------------------------------
    def load_single_sample(self, image_id):

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

    # ------------------------------------------------------
    # Episodic Sampling
    # ------------------------------------------------------
    def __getitem__(self, index):

        support_id = self.image_ids[index]

        ann_ids = self.coco.getAnnIds(imgIds=support_id)
        anns = self.coco.loadAnns(ann_ids)

        # Jika gambar tidak punya anotasi
        if len(anns) == 0:

            support_image, support_target = \
                self.load_single_sample(support_id)

            return {
                "support_image": support_image,
                "support_target": support_target,
                "query_image": support_image,
                "query_target": support_target
            }

        # Pilih salah satu class pada support image
        support_class = random.choice(anns)["category_id"]
        support_class = self.cat2label[support_class]

        candidate_images = self.class_to_images[support_class]

        # Cari query image yang berbeda
        if len(candidate_images) > 1:

            query_id = support_id

            while query_id == support_id:
                query_id = random.choice(candidate_images)

        else:

            query_id = support_id

        support_image, support_target = \
            self.load_single_sample(support_id)

        query_image, query_target = \
            self.load_single_sample(query_id)

        return {

            "support_image": support_image,
            "support_target": support_target,

            "query_image": query_image,
            "query_target": query_target

        }

        }

        return image, target
