# ==========================================================
# STEP 16 : COCO 1-Way Episodic Dataset
# ==========================================================

import os
import math
import random

import torch

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
        min_bbox_size=2.0
    ):
        super().__init__()


        self.coco = coco

        self.image_dir = (
            image_dir
        )

        self.support_transform = (
            support_transform
        )

        self.query_transform = (
            query_transform
        )

        self.num_episodes = int(
            num_episodes
        )

        self.seed = int(seed)

        self.epoch = 0

        self.min_bbox_size = float(
            min_bbox_size
        )


        # --------------------------------------------------
        # COCO semantic mapping
        # --------------------------------------------------

        self.cat_ids = sorted(
            coco.getCatIds()
        )


        self.cat2label = {

            cat_id:
                label

            for label, cat_id
            in enumerate(
                self.cat_ids
            )
        }


        self.label2cat = {

            label:
                cat_id

            for cat_id, label
            in self.cat2label.items()
        }


        # --------------------------------------------------
        # Index valid object instances / images by class
        # --------------------------------------------------

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


        for (
            ann_id,
            ann
        ) in coco.anns.items():

            if ann.get(
                "iscrowd",
                0
            ) == 1:

                continue


            if "bbox" not in ann:
                continue


            x, y, w, h = (
                ann["bbox"]
            )


            if (
                w < self.min_bbox_size
                or
                h < self.min_bbox_size
            ):
                continue


            cat_id = ann.get(
                "category_id"
            )


            if cat_id not in (
                self.cat2label
            ):
                continue


            label = (
                self.cat2label[
                    cat_id
                ]
            )


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


        for label in (
            self.class_to_img_ids
        ):

            self.class_to_img_ids[
                label
            ] = sorted(

                self.class_to_img_ids[
                    label
                ]
            )


        # At least 2 images:
        # support image != query image.

        self.valid_labels = [

            label

            for label in range(
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
            COCO_CONFIG[
                "num_classes"
            ]
        ):

            raise RuntimeError(

                "Not all 80 COCO classes "
                "have valid episodic data. "
                f"Found {len(self.valid_labels)}."
            )


        # Cache used by absent-wrong-support sampling.

        self._present_label_cache = {}


    # ======================================================
    # EPOCH
    # ======================================================

    def set_epoch(
        self,
        epoch
    ):

        self.epoch = int(epoch)


    def __len__(self):

        return self.num_episodes


    # ======================================================
    # DETERMINISTIC RNG
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
            int(index)
        )


        return random.Random(
            episode_seed
        )


    # ======================================================
    # IMAGE LOAD
    # ======================================================

    def _load_image(
        self,
        image_id
    ):

        info = self.coco.loadImgs(
            [int(image_id)]
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
    # ALL SEMANTIC CLASSES PRESENT IN QUERY IMAGE
    #
    # Conservative:
    # a class appearing anywhere in annotations is treated
    # as present and cannot become a wrong support.
    # ======================================================

    def get_present_labels(
        self,
        image_id
    ):

        image_id = int(
            image_id
        )


        if image_id in (
            self._present_label_cache
        ):

            return set(

                self._present_label_cache[
                    image_id
                ]
            )


        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id]
        )


        anns = self.coco.loadAnns(
            ann_ids
        )


        present = set()


        for ann in anns:

            cat_id = ann.get(
                "category_id"
            )


            if cat_id in (
                self.cat2label
            ):

                present.add(

                    int(
                        self.cat2label[
                            cat_id
                        ]
                    )
                )


        self._present_label_cache[
            image_id
        ] = tuple(
            sorted(present)
        )


        return set(present)


    # ======================================================
    # SUPPORT OBJECT CROP
    # ======================================================

    def _load_support(
        self,
        annotation_id,
        class_label
    ):

        ann = self.coco.anns[
            int(annotation_id)
        ]


        image_id = int(
            ann["image_id"]
        )


        image, _ = self._load_image(
            image_id
        )


        x, y, w, h = (
            ann["bbox"]
        )


        x1 = max(
            0,
            int(
                math.floor(x)
            )
        )

        y1 = max(
            0,
            int(
                math.floor(y)
            )
        )

        x2 = min(

            image.width,

            int(
                math.ceil(
                    x + w
                )
            )
        )

        y2 = min(

            image.height,

            int(
                math.ceil(
                    y + h
                )
            )
        )


        if (
            x2 <= x1
            or
            y2 <= y1
        ):

            raise RuntimeError(
                "Invalid support crop."
            )


        support_image = (
            image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )
        )


        if (
            self.support_transform
            is not None
        ):

            support_image = (
                self.support_transform(
                    support_image
                )
            )


        support_target = {

            "boxes":
                torch.tensor(

                    [[
                        0.5,
                        0.5,
                        1.0,
                        1.0
                    ]],

                    dtype=torch.float32
                ),

            "labels":
                torch.tensor(

                    [int(class_label)],

                    dtype=torch.long
                ),

            "image_id":
                torch.tensor(

                    image_id,

                    dtype=torch.long
                ),

            "annotation_id":
                torch.tensor(

                    int(annotation_id),

                    dtype=torch.long
                )
        }


        return (
            support_image,
            support_target
        )


    # ======================================================
    # QUERY FULL IMAGE
    #
    # ONLY episode-class objects become detection targets.
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


        img_w = float(
            info["width"]
        )

        img_h = float(
            info["height"]
        )


        cat_id = (
            self.label2cat[
                int(class_label)
            ]
        )


        ann_ids = self.coco.getAnnIds(

            imgIds=[
                int(image_id)
            ],

            catIds=[
                cat_id
            ],

            iscrowd=False
        )


        anns = self.coco.loadAnns(
            ann_ids
        )


        boxes = []
        labels = []


        for ann in anns:

            x, y, w, h = (
                ann["bbox"]
            )


            if (
                w < self.min_bbox_size
                or
                h < self.min_bbox_size
            ):

                continue


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
                int(class_label)
            )


        if len(boxes) == 0:

            raise RuntimeError(

                "Selected query contains "
                "no valid episode-class object."
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
                    int(image_id),
                    dtype=torch.long
                )
        }


        if (
            self.query_transform
            is not None
        ):

            image = (
                self.query_transform(
                    image
                )
            )


        return (
            image,
            query_target
        )


    # ======================================================
    # ONE 1-WAY EPISODE
    # ======================================================

    def __getitem__(
        self,
        index
    ):

        rng = self._get_rng(
            index
        )


        # --------------------------------------------------
        # Balanced cyclic class assignment.
        #
        # Every consecutive 80 episodes covers
        # every COCO class exactly once.
        # --------------------------------------------------

        class_position = (

            int(index)
            +
            self.epoch

        ) % len(
            self.valid_labels
        )


        class_label = int(

            self.valid_labels[
                class_position
            ]
        )


        # --------------------------------------------------
        # Support
        # --------------------------------------------------

        support_ann_id = rng.choice(

            self.class_to_ann_ids[
                class_label
            ]
        )


        support_image_id = int(

            self.coco.anns[
                support_ann_id
            ][
                "image_id"
            ]
        )


        # --------------------------------------------------
        # Query must be a DIFFERENT image.
        # --------------------------------------------------

        query_candidates = [

            image_id

            for image_id
            in self.class_to_img_ids[
                class_label
            ]

            if int(image_id)
            !=
            support_image_id
        ]


        if not query_candidates:

            raise RuntimeError(
                "No independent query image."
            )


        query_image_id = int(

            rng.choice(
                query_candidates
            )
        )


        (
            support_image,
            support_target
        ) = self._load_support(

            support_ann_id,
            class_label
        )


        (
            query_image,
            query_target
        ) = self._load_query(

            query_image_id,
            class_label
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
# TRAIN DATASET
# ==========================================================

train_dataset = COCOEpisodeDataset(

    coco=
        coco_train,

    image_dir=
        TRAIN_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        COCO_CONFIG[
            "num_train_episodes"
        ],

    seed=
        COCO_CONFIG[
            "seed"
        ],

    min_bbox_size=
        COCO_CONFIG[
            "min_bbox_size"
        ]
)


print("=" * 70)
print("STEP 16 : COCO EPISODIC DATASET READY")
print("=" * 70)

print(
    "Train Episodes:",
    len(train_dataset)
)

print(
    "Valid Classes :",
    len(
        train_dataset.valid_labels
    )
)

print("=" * 70)
