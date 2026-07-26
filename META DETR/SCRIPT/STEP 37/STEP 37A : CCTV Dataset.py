# ==========================================================
# STEP 37A : CCTV Dataset
# ==========================================================

from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class CCTVDataset(Dataset):

    def __init__(self, image_dir, label_dir, transform=None):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = []

        valid_ext = (".jpg", ".jpeg", ".png")

        for file in sorted(os.listdir(image_dir)):

            if not file.lower().endswith(valid_ext):
                continue

            base = os.path.splitext(file)[0]

            label_path = os.path.join(
                label_dir,
                base + ".txt"
            )

            # Skip image tanpa label
            if os.path.exists(label_path):
                self.images.append(file)

        print(f"Valid Images : {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(image_name)[0] + ".txt"
        )

        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []

        with open(label_path, "r") as f:

            for line in f.readlines():

                line = line.strip()

                if line == "":
                    continue

                cls, xc, yc, w, h = map(float, line.split())

                # --------------------------------------------------
                # YOLO format :
                # class cx cy w h
                #
                # Nilai sudah dalam format:
                # (cx, cy, w, h)
                # dan sudah ternormalisasi ke [0,1]
                #
                # Samakan dengan STEP 19 (COCO Dataset)
                # --------------------------------------------------

                boxes.append([xc, yc, w, h])
                labels.append(int(cls))

        if self.transform:
            image = self.transform(image)

        if len(boxes):

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.long
            )

            # ---------------------------------------
            # Debug (opsional)
            # Pastikan box tetap berada pada 0-1
            # ---------------------------------------
            if torch.any(boxes < 0) or torch.any(boxes > 1):
                print(f"[WARNING] Bounding box di luar rentang [0,1] pada image: {image_name}")

        else:

            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {

            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx)

        }

        return image, target

print("=" * 60)
print("CCTV Dataset Ready")
print("=" * 60)
