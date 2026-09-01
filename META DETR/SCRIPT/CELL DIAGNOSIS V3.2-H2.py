# ==========================================================
# CELL DIAGNOSIS V3.2-H2
# UNSEEN SUPPORT-QUERY REPRESENTATION DISCRIMINATION
#
# Model tested:
# v32_model = BEST V3.2 short-training epoch
#
# PURPOSE
# ----------------------------------------------------------
# Test whether the contrastive discrimination learned
# during V3.2 training GENERALIZES to unseen COCO-Val.
#
# Directly comparable to previous V3-H2:
#
#   correct support <-> target ROI
#   versus
#   wrong support   <-> same target ROI
#
# IMPORTANT:
# - NO training
# - NO gradients
# - 160 fixed COCO-Val episodes
# - Uses v32_model
# - Does NOT touch official V3 `model`
# ==========================================================

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS V3.2-H2 : UNSEEN REPRESENTATION")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert (
    "v32_model"
    in
    globals()
), (
    "v32_model does not exist. "
    "V3.2 short gate must exist in the current runtime."
)


v32_model.eval()


NUM_EPISODES = min(
    160,
    len(val_dataset)
)


# ==========================================================
# Find support from ANOTHER semantic class
#
# SAME protocol as previous V3-H2 so the results
# remain directly comparable.
# ==========================================================

def v32_h2_find_wrong_episode(
    dataset,
    current_index,
    current_class
):

    for offset in range(
        1,
        len(dataset)
    ):

        candidate_index = (

            current_index
            +
            offset

        ) % len(dataset)


        candidate = dataset[
            candidate_index
        ]


        candidate_class = int(

            candidate[
                "episode_class"
            ].item()
        )


        if (
            candidate_class
            !=
            current_class
        ):

            return candidate


    raise RuntimeError(
        "Could not find wrong-class support."
    )


# ==========================================================
# STORAGE
# ==========================================================

correct_similarities = []

wrong_similarities = []

support_pair_similarities = []

roi_cell_counts = []


# ==========================================================
# DIAGNOSIS
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(

        range(
            NUM_EPISODES
        ),

        desc=
            "V3.2 H2 Representation"
    ):

        # ==================================================
        # CURRENT VALIDATION EPISODE
        # ==================================================

        episode = val_dataset[
            episode_id
        ]


        episode_class = int(

            episode[
                "episode_class"
            ].item()
        )


        # ==================================================
        # CORRECT SUPPORT
        # ==================================================

        correct_support = (

            episode[
                "support_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # QUERY IMAGE
        # ==================================================

        query_image = (

            episode[
                "query_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # WRONG-CLASS SUPPORT
        #
        # Same protocol as old V3-H2.
        # ==================================================

        wrong_episode = (
            v32_h2_find_wrong_episode(

                dataset=
                    val_dataset,

                current_index=
                    episode_id,

                current_class=
                    episode_class
            )
        )


        wrong_support = (

            wrong_episode[
                "support_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # SHARED BACKBONE FEATURES
        # ==================================================

        correct_support_feature = (

            v32_model.backbone(
                correct_support
            )
        )


        wrong_support_feature = (

            v32_model.backbone(
                wrong_support
            )
        )


        query_feature = (

            v32_model.backbone(
                query_image
            )
        )


        # ==================================================
        # V3.2 SUPPORT PROTOTYPES
        #
        # Same V3SupportEncoder architecture.
        # ==================================================

        correct_prototype = (

            v32_model.support_encoder(
                correct_support_feature
            )
        )


        wrong_prototype = (

            v32_model.support_encoder(
                wrong_support_feature
            )
        )


        # ==================================================
        # CORRECT SUPPORT vs WRONG SUPPORT
        #
        # Measures how separated support prototypes are.
        # ==================================================

        support_pair_similarity = (

            F.cosine_similarity(

                correct_prototype,

                wrong_prototype,

                dim=-1
            )

            .item()
        )


        support_pair_similarities.append(
            support_pair_similarity
        )


        # ==================================================
        # QUERY GT BOXES
        #
        # normalized cxcywh
        # ==================================================

        gt_boxes = (

            episode[
                "query_target"
            ][
                "boxes"
            ]

            .detach()

            .cpu()
        )


        (
            _,
            _,
            feature_h,
            feature_w

        ) = query_feature.shape


        # ==================================================
        # EACH GT OBJECT
        # ==================================================

        for gt_box in gt_boxes:

            cx = float(
                gt_box[0].item()
            )

            cy = float(
                gt_box[1].item()
            )

            bw = float(
                gt_box[2].item()
            )

            bh = float(
                gt_box[3].item()
            )


            # ==================================================
            # normalized cxcywh
            # ->
            # query backbone feature-map xyxy
            # ==================================================

            x1_float = (

                cx
                -
                bw / 2.0

            ) * feature_w


            y1_float = (

                cy
                -
                bh / 2.0

            ) * feature_h


            x2_float = (

                cx
                +
                bw / 2.0

            ) * feature_w


            y2_float = (

                cy
                +
                bh / 2.0

            ) * feature_h


            x1 = max(

                0,

                min(

                    feature_w - 1,

                    int(
                        np.floor(
                            x1_float
                        )
                    )
                )
            )


            y1 = max(

                0,

                min(

                    feature_h - 1,

                    int(
                        np.floor(
                            y1_float
                        )
                    )
                )
            )


            x2 = max(

                x1 + 1,

                min(

                    feature_w,

                    int(
                        np.ceil(
                            x2_float
                        )
                    )
                )
            )


            y2 = max(

                y1 + 1,

                min(

                    feature_h,

                    int(
                        np.ceil(
                            y2_float
                        )
                    )
                )
            )


            # ==================================================
            # TARGET ROI FEATURE
            # ==================================================

            roi_feature = (

                query_feature[
                    :,
                    :,
                    y1:y2,
                    x1:x2
                ]
            )


            if (
                roi_feature.shape[-2]
                <=
                0

                or

                roi_feature.shape[-1]
                <=
                0
            ):

                raise RuntimeError(
                    "Empty ROI encountered."
                )


            roi_cells = (

                roi_feature.shape[-2]

                *

                roi_feature.shape[-1]
            )


            roi_cell_counts.append(
                int(
                    roi_cells
                )
            )


            # ==================================================
            # REPRESENT TARGET ROI WITH SAME SUPPORT ENCODER
            #
            # This matches V3.2 contrastive training mechanism.
            # ==================================================

            roi_prototype = (

                v32_model.support_encoder(
                    roi_feature
                )
            )


            # ==================================================
            # CORRECT SUPPORT vs TARGET ROI
            # ==================================================

            correct_similarity = (

                F.cosine_similarity(

                    correct_prototype,

                    roi_prototype,

                    dim=-1
                )

                .item()
            )


            # ==================================================
            # WRONG SUPPORT vs SAME TARGET ROI
            # ==================================================

            wrong_similarity = (

                F.cosine_similarity(

                    wrong_prototype,

                    roi_prototype,

                    dim=-1
                )

                .item()
            )


            correct_similarities.append(
                correct_similarity
            )


            wrong_similarities.append(
                wrong_similarity
            )


# ==========================================================
# NUMPY
# ==========================================================

correct_similarities = np.asarray(

    correct_similarities,

    dtype=np.float64
)


wrong_similarities = np.asarray(

    wrong_similarities,

    dtype=np.float64
)


support_pair_similarities = np.asarray(

    support_pair_similarities,

    dtype=np.float64
)


roi_cell_counts = np.asarray(

    roi_cell_counts,

    dtype=np.float64
)


assert (
    len(
        correct_similarities
    )
    >
    0
), (
    "No GT regions were evaluated."
)


# ==========================================================
# CORRECT-WRONG MARGINS
# ==========================================================

margins = (

    correct_similarities

    -

    wrong_similarities
)


# ==========================================================
# SUMMARY METRICS
# ==========================================================

mean_correct = float(
    correct_similarities.mean()
)


mean_wrong = float(
    wrong_similarities.mean()
)


mean_margin = float(
    margins.mean()
)


median_margin = float(
    np.median(
        margins
    )
)


correct_wins_fraction = float(

    (
        margins
        >
        0.0
    ).mean()
)


margin_005_fraction = float(

    (
        margins
        >=
        0.05
    ).mean()
)


margin_010_fraction = float(

    (
        margins
        >=
        0.10
    ).mean()
)


mean_support_pair_similarity = float(

    support_pair_similarities.mean()
)


median_roi_cells = float(

    np.median(
        roi_cell_counts
    )
)


q25_roi_cells = float(

    np.percentile(
        roi_cell_counts,
        25
    )
)


q75_roi_cells = float(

    np.percentile(
        roi_cell_counts,
        75
    )
)


# ==========================================================
# RESULT
# ==========================================================

print()
print("=" * 70)
print("V3.2-H2 UNSEEN REPRESENTATION RESULT")
print("=" * 70)


print(
    "Episodes tested             :",
    NUM_EPISODES
)


print(
    "GT regions compared         :",
    len(
        correct_similarities
    )
)


print("-" * 70)


print(
    f"Correct support ↔ ROI sim   : "
    f"{mean_correct:.6f}"
)


print(
    f"Wrong support ↔ ROI sim     : "
    f"{mean_wrong:.6f}"
)


print(
    f"Mean correct-wrong margin   : "
    f"{mean_margin:.6f}"
)


print(
    f"Median margin               : "
    f"{median_margin:.6f}"
)


print(
    f"Correct > wrong fraction    : "
    f"{correct_wins_fraction:.6f}"
)


print("-" * 70)


print(
    f"Margin >= 0.05 fraction     : "
    f"{margin_005_fraction:.6f}"
)


print(
    f"Margin >= 0.10 fraction     : "
    f"{margin_010_fraction:.6f}"
)


print("-" * 70)


print(
    f"Correct ↔ wrong support sim : "
    f"{mean_support_pair_similarity:.6f}"
)


print("-" * 70)


print(
    f"Median query ROI cells      : "
    f"{median_roi_cells:.2f}"
)


print(
    f"Q25 query ROI cells         : "
    f"{q25_roi_cells:.2f}"
)


print(
    f"Q75 query ROI cells         : "
    f"{q75_roi_cells:.2f}"
)


print("=" * 70)


# ==========================================================
# DIRECT REFERENCE TO PREVIOUS V3-H2
# ==========================================================

print()
print("REFERENCE — PREVIOUS V3-H2")
print("-" * 70)


print(
    "Correct support ↔ ROI sim   : 0.297627"
)


print(
    "Wrong support ↔ ROI sim     : 0.269295"
)


print(
    "Mean correct-wrong margin   : 0.028332"
)


print(
    "Median margin               : 0.011966"
)


print(
    "Correct > wrong fraction    : 0.569405"
)


print(
    "Correct ↔ wrong support sim : 0.529899"
)


print("-" * 70)


print(
    "V3.2 training target margin : "
    f"{V32_CONFIG['contrastive_margin']:.6f}"
)


print(
    "Interpretation must use unseen COCO-Val results above, "
    "not the training margin."
)


print("=" * 70)
