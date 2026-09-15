# ==========================================================
# STEP 16 : COCO-80 Balanced Episodic Dataset
#
# 1-way / 1-shot episodic source meta-training.
#
# Example:
# support = dog crop
# query   = full image containing dog
# GT      = every eligible dog in query
#
# Next episode may use car, person, bus, ...
#
# IMPORTANT:
# detection labels remain binary-relative-to-support.
# Semantic class is stored separately as episode_class.
# ==========================================================

from PIL import Image

from torch.utils.data import Dataset


class COCOEpisodicDataset(Dataset):

    def __init__(
        self,
        coco,
        image_dir,
        support_transform,
        query_transform,
        num_episodes,
        seed,
        cat_id_to_label,
        category_names,
        min_bbox_size=2.0,
        allowed_labels=None,
    ):

        self.coco = coco

        self.image_dir = image_dir

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

        self.min_bbox_size = float(
            min_bbox_size
        )

        self.cat_id_to_label = dict(
            cat_id_to_label
        )

        self.label_to_cat_id = {
            label: cat_id
            for cat_id, label
            in self.cat_id_to_label.items()
        }

        self.category_names = dict(
            category_names
        )

        if allowed_labels is None:

            self.valid_labels = sorted(
                self.label_to_cat_id
            )

        else:

            self.valid_labels = sorted(
                int(label)
                for label
                in allowed_labels
            )

        if not self.valid_labels:
            raise ValueError(
                "allowed_labels menghasilkan "
                "dataset kosong."
            )

        for label in self.valid_labels:

            if label not in self.label_to_cat_id:
                raise ValueError(
                    f"Invalid internal label: {label}"
                )

        self.epoch = 0

        # --------------------------------------------------
        # Eligible annotation indexes
        # --------------------------------------------------

        self.valid_xyxy = {}

        self.class_to_ann_ids = {
            label: []
            for label in self.valid_labels
        }

        self.class_to_image_ann_ids = {
            label: {}
            for label in self.valid_labels
        }

        allowed_cat_ids = {
            self.label_to_cat_id[label]
            for label in self.valid_labels
        }

        for ann_id, ann in coco.anns.items():

            cat_id = int(
                ann.get(
                    "category_id",
                    -1
                )
            )

            if cat_id not in allowed_cat_ids:
                continue

            if ann.get(
                "iscrowd",
                0
            ):
                continue

            if "bbox" not in ann:
                continue

            image_id = int(
                ann["image_id"]
            )

            image_info = (
                coco.imgs[image_id]
            )

            image_width = float(
                image_info["width"]
            )

            image_height = float(
                image_info["height"]
            )

            if (
                image_width <= 0
                or
                image_height <= 0
            ):
                continue

            x, y, w, h = map(
                float,
                ann["bbox"]
            )

            if not all(
                math.isfinite(value)
                for value in (
                    x, y, w, h
                )
            ):
                continue

            x1 = max(
                0.0,
                x
            )

            y1 = max(
                0.0,
                y
            )

            x2 = min(
                image_width,
                x + w
            )

            y2 = min(
                image_height,
                y + h
            )

            if (
                x2 - x1
                <
                self.min_bbox_size
                or
                y2 - y1
                <
                self.min_bbox_size
            ):
                continue

            label = int(
                self.cat_id_to_label[
                    cat_id
                ]
            )

            ann_id = int(
                ann_id
            )

            self.valid_xyxy[
                ann_id
            ] = (
                x1,
                y1,
                x2,
                y2,
            )

            self.class_to_ann_ids[
                label
            ].append(
                ann_id
            )

            (
                self
                .class_to_image_ann_ids[
                    label
                ]
                .setdefault(
                    image_id,
                    []
                )
                .append(
                    ann_id
                )
            )

        # --------------------------------------------------
        # Sort indexes for reproducibility
        # --------------------------------------------------

        self.class_to_img_ids = {}

        self.class_image_positions = {}

        for label in self.valid_labels:

            self.class_to_ann_ids[
                label
            ].sort()

            for image_id in (
                self
                .class_to_image_ann_ids[
                    label
                ]
            ):

                (
                    self
                    .class_to_image_ann_ids[
                        label
                    ][image_id]
                    .sort()
                )

            image_ids = sorted(
                self
                .class_to_image_ann_ids[
                    label
                ]
            )

            if len(image_ids) < 2:

                raise RuntimeError(
                    "Category "
                    f"{label} "
                    "does not contain at least "
                    "two eligible images."
                )

            self.class_to_img_ids[
                label
            ] = image_ids

            self.class_image_positions[
                label
            ] = {

                image_id: index

                for index, image_id
                in enumerate(
                    image_ids
                )
            }

        # Balanced class schedule.
        self._build_episode_labels()

    def __len__(self):

        return self.num_episodes

    # ======================================================
    # BALANCED EPISODE SCHEDULE
    #
    # 8000 episodes / 80 classes
    # -> 100 episodes per class per epoch.
    #
    # 800 validation episodes
    # -> 10 per class.
    # ======================================================

    def _build_episode_labels(
        self
    ):

        repeats = math.ceil(
            self.num_episodes
            /
            len(
                self.valid_labels
            )
        )

        labels = (
            self.valid_labels
            *
            repeats
        )[
            :self.num_episodes
        ]

        rng = random.Random(
            self.seed
            +
            self.epoch
            *
            1_000_003
        )

        rng.shuffle(
            labels
        )

        self.episode_labels = (
            labels
        )

    def set_epoch(
        self,
        epoch
    ):

        self.epoch = int(
            epoch
        )

        self._build_episode_labels()

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
    # IMAGE LOADING
    # ======================================================

    def _load_image(
        self,
        image_id
    ):

        image_id = int(
            image_id
        )

        info = (
            self.coco.imgs[
                image_id
            ]
        )

        image_path = (
            os.path.join(
                self.image_dir,
                info["file_name"]
            )
        )

        with Image.open(
            image_path
        ) as source:

            image = source.convert(
                "RGB"
            )

        expected_size = (
            int(
                info["width"]
            ),
            int(
                info["height"]
            ),
        )

        if image.size != expected_size:

            raise RuntimeError(
                "Image/annotation "
                "dimensions differ for "
                f"image_id={image_id}"
            )

        return (
            image,
            info
        )

    # ======================================================
    # SUPPORT
    # ======================================================

    def _load_support(
        self,
        annotation_id,
        semantic_label
    ):

        annotation_id = int(
            annotation_id
        )

        semantic_label = int(
            semantic_label
        )

        ann = (
            self.coco.anns[
                annotation_id
            ]
        )

        cat_id = int(
            ann[
                "category_id"
            ]
        )

        expected_label = int(
            self.cat_id_to_label[
                cat_id
            ]
        )

        if (
            expected_label
            != semantic_label
        ):

            raise RuntimeError(
                "Support category/episode "
                "label mismatch."
            )

        image_id = int(
            ann[
                "image_id"
            ]
        )

        image, _ = (
            self._load_image(
                image_id
            )
        )

        (
            x1,
            y1,
            x2,
            y2

        ) = self.valid_xyxy[
            annotation_id
        ]

        crop = image.crop(
            (
                math.floor(
                    x1
                ),
                math.floor(
                    y1
                ),
                math.ceil(
                    x2
                ),
                math.ceil(
                    y2
                ),
            )
        )

        if (
            crop.width <= 0
            or
            crop.height <= 0
        ):
            raise RuntimeError(
                "Invalid support crop."
            )

        if (
            self.support_transform
            is not None
        ):

            crop = (
                self
                .support_transform(
                    crop
                )
            )

        support_target = {

            "image_id":
                torch.tensor(
                    image_id,
                    dtype=torch.long
                ),

            "annotation_id":
                torch.tensor(
                    annotation_id,
                    dtype=torch.long
                ),

            "semantic_label":
                torch.tensor(
                    semantic_label,
                    dtype=torch.long
                ),

            "category_id":
                torch.tensor(
                    cat_id,
                    dtype=torch.long
                ),
        }

        return (
            crop,
            support_target
        )

    # ======================================================
    # QUERY
    # ======================================================

    def _load_query(
        self,
        image_id,
        semantic_label
    ):

        image_id = int(
            image_id
        )

        semantic_label = int(
            semantic_label
        )

        image, info = (
            self._load_image(
                image_id
            )
        )

        width = float(
            info["width"]
        )

        height = float(
            info["height"]
        )

        ann_ids = (
            self
            .class_to_image_ann_ids[
                semantic_label
            ]
            .get(
                image_id,
                []
            )
        )

        if not ann_ids:

            raise RuntimeError(
                "Query image unexpectedly "
                "contains no target instances."
            )

        boxes = []

        for ann_id in ann_ids:

            (
                x1,
                y1,
                x2,
                y2

            ) = self.valid_xyxy[
                ann_id
            ]

            boxes.append(
                [
                    (
                        x1 + x2
                    )
                    /
                    (
                        2.0
                        *
                        width
                    ),

                    (
                        y1 + y2
                    )
                    /
                    (
                        2.0
                        *
                        height
                    ),

                    (
                        x2 - x1
                    )
                    /
                    width,

                    (
                        y2 - y1
                    )
                    /
                    height,
                ]
            )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(
            -1,
            4
        )

        # Binary episodic target:
        # every GT is foreground relative to support.
        labels = torch.zeros(
            len(boxes),
            dtype=torch.long
        )

        cat_id = int(
            self.label_to_cat_id[
                semantic_label
            ]
        )

        target = {

            "boxes":
                boxes,

            "labels":
                labels,

            "image_id":
                torch.tensor(
                    image_id,
                    dtype=torch.long
                ),

            "semantic_label":
                torch.tensor(
                    semantic_label,
                    dtype=torch.long
                ),

            "category_id":
                torch.tensor(
                    cat_id,
                    dtype=torch.long
                ),
        }

        if (
            self.query_transform
            is not None
        ):

            image = (
                self
                .query_transform(
                    image
                )
            )

        return (
            image,
            target
        )

    # ======================================================
    # EPISODE
    # ======================================================

    def __getitem__(
        self,
        index
    ):

        index = int(
            index
        )

        if not (
            0
            <= index
            <
            len(self)
        ):
            raise IndexError(
                index
            )

        rng = self._get_rng(
            index
        )

        semantic_label = int(
            self.episode_labels[
                index
            ]
        )

        # --------------------------------------------------
        # Pick support instance.
        # --------------------------------------------------

        support_ann_id = (
            rng.choice(
                self.class_to_ann_ids[
                    semantic_label
                ]
            )
        )

        support_image_id = int(
            self.coco.anns[
                support_ann_id
            ][
                "image_id"
            ]
        )

        # --------------------------------------------------
        # Pick query image from SAME semantic class
        # but DIFFERENT image.
        # --------------------------------------------------

        image_ids = (
            self.class_to_img_ids[
                semantic_label
            ]
        )

        support_position = (
            self
            .class_image_positions[
                semantic_label
            ][
                support_image_id
            ]
        )

        query_position = (
            rng.randrange(
                len(
                    image_ids
                )
                -
                1
            )
        )

        if (
            query_position
            >=
            support_position
        ):
            query_position += 1

        query_image_id = int(
            image_ids[
                query_position
            ]
        )

        if (
            support_image_id
            ==
            query_image_id
        ):
            raise RuntimeError(
                "Support/query leakage."
            )

        support_image, support_target = (
            self._load_support(
                support_ann_id,
                semantic_label
            )
        )

        query_image, query_target = (
            self._load_query(
                query_image_id,
                semantic_label
            )
        )

        return {

            "episode_class":
                torch.tensor(
                    semantic_label,
                    dtype=torch.long
                ),

            "support_image":
                support_image,

            "support_target":
                support_target,

            "query_image":
                query_image,

            "query_target":
                query_target,
        }


# ==========================================================
# BUILD SOURCE TRAIN / VALIDATION DATASETS
# ==========================================================

train_dataset = (
    COCOEpisodicDataset(

        coco=coco_train,

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

        cat_id_to_label=
            CAT_ID_TO_LABEL,

        category_names=
            CATEGORY_NAMES,

        min_bbox_size=
            COCO_CONFIG[
                "min_bbox_size"
            ],
    )
)


val_dataset = (
    COCOEpisodicDataset(

        coco=coco_val,

        image_dir=
            VAL_IMAGE_DIR,

        support_transform=
            support_transform,

        query_transform=
            query_transform,

        num_episodes=
            COCO_CONFIG[
                "num_val_episodes"
            ],

        # Different deterministic stream.
        seed=
            COCO_CONFIG[
                "seed"
            ]
            +
            100_000,

        cat_id_to_label=
            CAT_ID_TO_LABEL,

        category_names=
            CATEGORY_NAMES,

        min_bbox_size=
            COCO_CONFIG[
                "min_bbox_size"
            ],
    )
)


train_dataset.set_epoch(
    0
)

val_dataset.set_epoch(
    0
)


print("=" * 70)
print("STEP 16 : COCO-80 EPISODIC DATASETS READY")
print("=" * 70)

print(
    "Train episodes :",
    len(
        train_dataset
    )
)

print(
    "Val episodes   :",
    len(
        val_dataset
    )
)

print(
    "Source classes :",
    len(
        train_dataset
        .valid_labels
    )
)

print(
    "Person label   :",
    PERSON_LABEL
)

print("=" * 70)
