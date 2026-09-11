# ==========================================================
# STEP 16: Person-Only Episodic Dataset
#
# Support: crop person.
# Query: gambar penuh dengan seluruh person GT yang eligible.
# Support dan query berasal dari gambar berbeda.
#
# Crowd dan bbox sangat kecil tetap dikecualikan.
# ==========================================================

from PIL import Image
from torch.utils.data import Dataset


class COCOEpisodeDataset(Dataset):
    def __init__(
        self,
        coco,
        image_dir,
        support_transform,
        query_transform,
        num_episodes,
        seed,
        min_bbox_size=2.0,
    ):
        self.coco = coco
        self.image_dir = image_dir

        self.support_transform = support_transform
        self.query_transform = query_transform

        self.num_episodes = int(num_episodes)
        self.seed = int(seed)
        self.epoch = 0

        self.min_bbox_size = float(min_bbox_size)

        self.cat_ids = sorted(
            coco.getCatIds(
                catNms=["person"]
            )
        )

        assert len(self.cat_ids) == 1

        self.cat2label = {
            self.cat_ids[0]: 0
        }

        self.label2cat = {
            0: self.cat_ids[0]
        }

        self.valid_labels = [0]

        self.class_to_ann_ids = {
            0: []
        }

        self.image_to_ann_ids = {}
        self.valid_xyxy = {}

        for ann_id, ann in coco.anns.items():
            if ann.get("category_id") != self.cat_ids[0]:
                continue

            if ann.get("iscrowd", 0):
                continue

            if "bbox" not in ann:
                continue

            info = coco.imgs[ann["image_id"]]

            width = float(info["width"])
            height = float(info["height"])

            x, y, w, h = map(
                float,
                ann["bbox"]
            )

            if width <= 0 or height <= 0:
                continue

            if not all(
                map(
                    math.isfinite,
                    (x, y, w, h)
                )
            ):
                continue

            # Clip terhadap batas gambar.
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if (
                x2 - x1 < self.min_bbox_size
                or y2 - y1 < self.min_bbox_size
            ):
                continue

            ann_id = int(ann_id)
            image_id = int(ann["image_id"])

            self.valid_xyxy[ann_id] = (
                x1, y1, x2, y2
            )

            self.class_to_ann_ids[0].append(
                ann_id
            )

            self.image_to_ann_ids.setdefault(
                image_id,
                []
            ).append(
                ann_id
            )

        self.class_to_ann_ids[0].sort()

        self.class_to_img_ids = {
            0: sorted(self.image_to_ann_ids)
        }

        self.image_positions = {
            image_id: index
            for index, image_id in enumerate(
                self.class_to_img_ids[0]
            )
        }

        if len(self.image_positions) < 2:
            raise RuntimeError(
                "Need at least two images with valid person annotations."
            )

    def __len__(self):
        return self.num_episodes

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _get_rng(self, index):
        episode_seed = (
            self.seed
            + self.epoch * self.num_episodes
            + int(index)
        )

        return random.Random(episode_seed)

    def _load_image(self, image_id):
        info = self.coco.imgs[int(image_id)]

        image_path = os.path.join(
            self.image_dir,
            info["file_name"]
        )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        expected_size = (
            info["width"],
            info["height"]
        )

        if image.size != expected_size:
            raise RuntimeError(
                f"Image/annotation dimensions differ: {image_id}"
            )

        return image, info

    def _load_support(
        self,
        annotation_id,
        class_label=0,
    ):
        assert int(class_label) == 0

        annotation_id = int(annotation_id)

        ann = self.coco.anns[
            annotation_id
        ]

        image_id = int(
            ann["image_id"]
        )

        image, _ = self._load_image(
            image_id
        )

        x1, y1, x2, y2 = self.valid_xyxy[
            annotation_id
        ]

        crop = image.crop((
            math.floor(x1),
            math.floor(y1),
            math.ceil(x2),
            math.ceil(y2),
        ))

        if self.support_transform is not None:
            crop = self.support_transform(
                crop
            )

        support_target = {
            "boxes": torch.tensor(
                [[0.5, 0.5, 1.0, 1.0]],
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                [0],
                dtype=torch.long
            ),

            "image_id": torch.tensor(
                image_id,
                dtype=torch.long
            ),

            "annotation_id": torch.tensor(
                annotation_id,
                dtype=torch.long
            ),
        }

        return crop, support_target

    def _load_query(
        self,
        image_id,
        class_label=0,
    ):
        assert int(class_label) == 0

        image, info = self._load_image(
            image_id
        )

        width = float(info["width"])
        height = float(info["height"])

        boxes = []

        for ann_id in self.image_to_ann_ids.get(
            int(image_id),
            []
        ):
            x1, y1, x2, y2 = self.valid_xyxy[
                ann_id
            ]

            boxes.append([
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ])

        target = {
            "boxes": torch.tensor(
                boxes,
                dtype=torch.float32
            ).reshape(-1, 4),

            "labels": torch.zeros(
                len(boxes),
                dtype=torch.long
            ),

            "image_id": torch.tensor(
                int(image_id),
                dtype=torch.long
            ),
        }

        if self.query_transform is not None:
            image = self.query_transform(
                image
            )

        return image, target

    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)

        rng = self._get_rng(index)

        ann_id = rng.choice(
            self.class_to_ann_ids[0]
        )

        support_id = int(
            self.coco.anns[ann_id]["image_id"]
        )

        # Pilih query image berbeda tanpa membuat ulang
        # daftar seluruh kandidat pada setiap episode.
        support_position = self.image_positions[
            support_id
        ]

        query_position = rng.randrange(
            len(self.image_positions) - 1
        )

        if query_position >= support_position:
            query_position += 1

        query_id = self.class_to_img_ids[0][
            query_position
        ]

        support_image, support_target = self._load_support(
            ann_id
        )

        query_image, query_target = self._load_query(
            query_id
        )

        return {
            "episode_class": torch.tensor(
                0,
                dtype=torch.long
            ),

            "support_image": support_image,
            "support_target": support_target,

            "query_image": query_image,
            "query_target": query_target,
        }


train_dataset = COCOEpisodeDataset(
    coco=coco_train,
    image_dir=TRAIN_IMAGE_DIR,
    support_transform=support_transform,
    query_transform=query_transform,
    num_episodes=COCO_CONFIG["num_train_episodes"],
    seed=COCO_CONFIG["seed"],
    min_bbox_size=COCO_CONFIG["min_bbox_size"],
)

print("=" * 70)
print("STEP 16: PERSON DATASET READY")
print("=" * 70)

print(
    "Person images:",
    len(train_dataset.class_to_img_ids[0])
)

print(
    "Eligible person instances:",
    len(train_dataset.class_to_ann_ids[0])
)

print(
    "Training episodes:",
    len(train_dataset)
)

print("=" * 70)
