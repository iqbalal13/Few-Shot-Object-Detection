# ==========================================================
# STEP 19 : COCO Base Dataset
# ==========================================================

from torch.utils.data import Dataset
from PIL import Image
import os

class COCOBaseDataset(Dataset):

    def __init__(self,
                 coco,
                 image_dir,
                 transform=None):

        self.coco = coco

        self.image_dir = image_dir

        self.transform = transform

        self.image_ids = list(coco.imgs.keys())

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

            boxes.append(ann["bbox"])

            labels.append(ann["category_id"])

        if self.transform:

            image = self.transform(image)

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": image_id

        }

        return image, target
