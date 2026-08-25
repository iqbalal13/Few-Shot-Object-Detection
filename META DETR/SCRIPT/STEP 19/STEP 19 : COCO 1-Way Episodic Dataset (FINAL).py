# ==========================================================
# STEP 19 : COCO 1-Way Episodic Dataset (FINAL)
# ==========================================================

import os
import math
import random

import torch

from torch.utils.data import Dataset

from PIL import Image


class COCOEpisodeDataset(Dataset):

    def __init__(
        self,
        coco,
        image_dir,
        support_transform,
        query_transform,
        num_episodes=8000,
        seed=42,
        min_bbox_size=2.0
    ):

        super().__init__()

        self.coco = coco

        self.image_dir = image_dir

        self.support_transform = (
            support_transform
        )

        self.query_transform = (
            query_transform
        )

        self.num_episodes = (
            num_episodes
        )

        self.seed = seed

        self.epoch = 0

        self.min_bbox_size = (
            min_bbox_size
        )


        # ==================================================
        # Category Mapping
        # ==================================================

        self.cat_ids = sorted(
            self.coco.getCatIds()
        )

        self.cat2label = {

            cat_id: label

            for label, cat_id
            in enumerate(self.cat_ids)
        }

        self.label2cat = {

            label: cat_id

            for cat_id, label
            in self.cat2label.items()
        }


        # ==================================================
        # Build:
        #
        # class -> valid annotations
        # class -> valid images
        # ==================================================

        self.class_to_ann_ids = {

            label: []

            for label in range(
                len(self.cat_ids)
            )
        }

        self.class_to_img_ids = {

            label: set()

            for label in range(
                len(self.cat_ids)
            )
        }


        for ann_id, ann in self.coco.anns.items():

            # Ignore crowd annotations
            if ann.get(
                "iscrowd",
                0
            ) == 1:

                continue

            if "bbox" not in ann:
                continue

            x, y, w, h = ann["bbox"]

            # Remove invalid / tiny boxes
            if (
                w < self.min_bbox_size
                or
                h < self.min_bbox_size
            ):

                continue

            cat_id = ann[
                "category_id"
            ]

            if (
                cat_id
                not in
                self.cat2label
            ):

                continue

            label = self.cat2label[
                cat_id
            ]

            self.class_to_ann_ids[
                label
            ].append(
                ann_id
            )

            self.class_to_img_ids[
                label
            ].add(
                ann["image_id"]
            )


        # Convert set -> sorted list
        for label in self.class_to_img_ids:

            self.class_to_img_ids[
                label
            ] = sorted(
                self.class_to_img_ids[
                    label
                ]
            )


        # Classes must have at least
        # two different images:
        #
        # support image != query image
        self.valid_labels = [

            label

            for label
            in range(
                len(self.cat_ids)
            )

            if (
                len(
                    self.class_to_ann_ids[
                        label
                    ]
                ) > 0
                and
                len(
                    self.class_to_img_ids[
                        label
                    ]
                ) >= 2
            )
        ]


        if (
            len(self.valid_labels)
            !=
            CONFIG["num_classes"]
        ):

            raise RuntimeError(

                "Tidak semua COCO classes "
                "mempunyai valid episodic samples. "

                f"Available = "
                f"{len(self.valid_labels)}"
            )


    # ======================================================
    # Epoch Control
    #
    # Training Step nanti memanggil:
    #
    # train_dataset.set_epoch(epoch)
    #
    # sehingga episode berubah setiap epoch
    # tetapi tetap reproducible.
    # ======================================================

    def set_epoch(
        self,
        epoch
    ):

        self.epoch = int(
            epoch
        )


    def __len__(self):

        return self.num_episodes


    # ======================================================
    # Deterministic RNG
    # ======================================================

    def _get_rng(
        self,
        index
    ):

        episode_seed = (

            self.seed

            +

            self.epoch
            *
            self.num_episodes

            +

            index
        )

        return random.Random(
            episode_seed
        )


    # ======================================================
    # Load RGB Image
    # ======================================================

    def _load_image(
        self,
        image_id
    ):

        info = self.coco.loadImgs(
            [image_id]
        )[0]

        image_path = os.path.join(
            self.image_dir,
            info["file_name"]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        return image, info


    # ======================================================
    # Support Object Crop
    #
    # This is what makes the support
    # representation CLASS-SPECIFIC.
    # ======================================================

    def _load_support(
        self,
        annotation_id,
        class_label
    ):

        ann = self.coco.anns[
            annotation_id
        ]

        image_id = ann[
            "image_id"
        ]

        image, image_info = (
            self._load_image(
                image_id
            )
        )

        x, y, w, h = ann[
            "bbox"
        ]


        # Clamp bounding box
        x1 = max(
            0,
            int(math.floor(x))
        )

        y1 = max(
            0,
            int(math.floor(y))
        )

        x2 = min(
            image.width,
            int(math.ceil(x + w))
        )

        y2 = min(
            image.height,
            int(math.ceil(y + h))
        )


        if (
            x2 <= x1
            or
            y2 <= y1
        ):

            raise RuntimeError(
                "Invalid support crop."
            )


        # Object-only support image
        support_image = image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        if self.support_transform:

            support_image = (
                self.support_transform(
                    support_image
                )
            )


        # Since support image itself
        # is the object crop, its box is
        # the entire support image.
        support_target = {

            "boxes":
                torch.tensor(
                    [
                        [
                            0.5,
                            0.5,
                            1.0,
                            1.0
                        ]
                    ],
                    dtype=torch.float32
                ),

            "labels":
                torch.tensor(
                    [class_label],
                    dtype=torch.long
                ),

            "image_id":
                torch.tensor(
                    image_id,
                    dtype=torch.long
                ),

            "annotation_id":
                torch.tensor(
                    annotation_id,
                    dtype=torch.long
                )
        }


        return (
            support_image,
            support_target
        )


    # ======================================================
    # Query Full Image
    #
    # IMPORTANT:
    # Only objects belonging to the
    # EPISODE CLASS are used as targets.
    # ======================================================

    def _load_query(
        self,
        image_id,
        class_label
    ):

        image, info = (
            self._load_image(
                image_id
            )
        )

        img_w = info[
            "width"
        ]

        img_h = info[
            "height"
        ]

        cat_id = self.label2cat[
            class_label
        ]


        ann_ids = self.coco.getAnnIds(

            imgIds=[image_id],

            catIds=[cat_id],

            iscrowd=False
        )


        anns = self.coco.loadAnns(
            ann_ids
        )


        boxes = []

        labels = []


        for ann in anns:

            x, y, w, h = ann[
                "bbox"
            ]

            if (
                w < self.min_bbox_size
                or
                h < self.min_bbox_size
            ):

                continue


            # ----------------------------------------------
            # COCO xywh
            # ->
            # normalized cxcywh
            # ----------------------------------------------

            cx = (
                x + w / 2.0
            ) / img_w

            cy = (
                y + h / 2.0
            ) / img_h

            nw = w / img_w

            nh = h / img_h


            boxes.append(
                [
                    cx,
                    cy,
                    nw,
                    nh
                ]
            )

            labels.append(
                class_label
            )


        if len(boxes) == 0:

            raise RuntimeError(
                "Selected query does not contain "
                "a valid target instance."
            )


        query_target = {

            "boxes":
                torch.tensor(
                    boxes,
                    dtype=torch.float32
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long
                ),

            "image_id":
                torch.tensor(
                    image_id,
                    dtype=torch.long
                )
        }


        if self.query_transform:

            image = (
                self.query_transform(
                    image
                )
            )


        return image, query_target


    # ======================================================
    # Generate One 1-Way Episode
    # ======================================================

    def __getitem__(
        self,
        index
    ):

        rng = self._get_rng(
            index
        )


        # ==================================================
        # Balanced class assignment
        #
        # Across 80 consecutive episodes,
        # every COCO class appears once.
        # ==================================================

        class_position = (

            index
            +
            self.epoch

        ) % len(
            self.valid_labels
        )


        class_label = (
            self.valid_labels[
                class_position
            ]
        )


        # ==================================================
        # Select Support Instance
        # ==================================================

        support_ann_id = rng.choice(

            self.class_to_ann_ids[
                class_label
            ]
        )


        support_ann = (
            self.coco.anns[
                support_ann_id
            ]
        )


        support_image_id = (
            support_ann[
                "image_id"
            ]
        )


        # ==================================================
        # Select Different Query Image
        # ==================================================

        query_candidates = [

            image_id

            for image_id
            in self.class_to_img_ids[
                class_label
            ]

            if (
                image_id
                !=
                support_image_id
            )
        ]


        if len(
            query_candidates
        ) == 0:

            raise RuntimeError(
                "No independent query image "
                "available for episode."
            )


        query_image_id = rng.choice(
            query_candidates
        )


        # ==================================================
        # Load Episode
        # ==================================================

        support_image, support_target = (
            self._load_support(
                support_ann_id,
                class_label
            )
        )


        query_image, query_target = (
            self._load_query(
                query_image_id,
                class_label
            )
        )


        return {

            "episode_class":
                torch.tensor(
                    class_label,
                    dtype=torch.long
                ),

            "support_image":
                support_image,

            "support_target":
                support_target,

            "query_image":
                query_image,

            "query_target":
                query_target
        }


# ==========================================================
# Initialize Base Meta-Training Dataset
# ==========================================================

TRAIN_IMAGE_DIR = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["train_images"]
)


train_dataset = COCOEpisodeDataset(

    coco=coco_train,

    image_dir=TRAIN_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        COCO_CONFIG[
            "num_train_episodes"
        ],

    seed=
        COCO_CONFIG["seed"],

    min_bbox_size=
        COCO_CONFIG[
            "min_bbox_size"
        ]
)


print("=" * 70)
print("STEP 19 : COCO EPISODIC DATASET READY")
print("=" * 70)

print(
    "Episodes / Epoch :",
    len(train_dataset)
)

print(
    "Valid Classes    :",
    len(
        train_dataset.valid_labels
    )
)

print(
    "Episode Way      :",
    CONFIG["episode_way"]
)

print(
    "Support Shot     :",
    CONFIG["base_support_shot"]
)

print("=" * 70)
