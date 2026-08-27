# ==========================================================
# CELL DIAGNOSIS H2
# SUPPORT ↔ QUERY REPRESENTATION ALIGNMENT
#
# TEST A:
# correct support prototype
# vs target-object region inside full query
#
# TEST B:
# wrong-class support prototype
# vs same target-object region
#
# TEST C:
# support prototype stability under scale changes
# ==========================================================

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS H2 : SUPPORT-QUERY REPRESENTATION")
print("=" * 70)

model.eval()

NUM_EPISODES = min(
    160,
    len(val_dataset)
)

TEST_SCALES = [
    0.75,
    0.50
]


# ==========================================================
# Helper: wrong support
# ==========================================================

def find_wrong_support_index(
    dataset,
    current_index,
    current_class
):

    for offset in range(
        1,
        len(dataset)
    ):

        idx = (
            current_index + offset
        ) % len(dataset)

        candidate = dataset[idx]

        candidate_class = int(
            candidate[
                "episode_class"
            ].item()
        )

        if candidate_class != current_class:
            return idx

    raise RuntimeError(
        "Different-class support not found."
    )


# ==========================================================
# Helper: extract support prototype
# ==========================================================

def extract_support_prototype(
    support_image
):

    feature = model.backbone(
        support_image
    )

    _, prototype = (
        model.support_encoder(
            feature
        )
    )

    return prototype


# ==========================================================
# Helper:
# extract prototype from GT region
# inside FULL QUERY feature map
#
# GT format:
# normalized cxcywh
# ==========================================================

def extract_query_roi_prototype(
    query_feature,
    gt_box
):

    # query_feature:
    # [1, 256, H, W]

    _, C, H, W = (
        query_feature.shape
    )

    cx, cy, bw, bh = (
        gt_box.tolist()
    )


    # ----------------------------------------------
    # normalized bbox boundaries
    # ----------------------------------------------

    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0

    x1 = cx + bw / 2.0
    y1 = cy + bh / 2.0


    # ----------------------------------------------
    # map to feature-map coordinates
    # ----------------------------------------------

    ix0 = int(
        np.floor(
            x0 * W
        )
    )

    iy0 = int(
        np.floor(
            y0 * H
        )
    )

    ix1 = int(
        np.ceil(
            x1 * W
        )
    )

    iy1 = int(
        np.ceil(
            y1 * H
        )
    )


    # ----------------------------------------------
    # Clamp
    # At least one feature cell
    # ----------------------------------------------

    ix0 = max(
        0,
        min(
            ix0,
            W - 1
        )
    )

    iy0 = max(
        0,
        min(
            iy0,
            H - 1
        )
    )

    ix1 = max(
        ix0 + 1,
        min(
            ix1,
            W
        )
    )

    iy1 = max(
        iy0 + 1,
        min(
            iy1,
            H
        )
    )


    # ----------------------------------------------
    # ROI features
    #
    # [1,C,h,w]
    # ----------------------------------------------

    roi = query_feature[
        :,
        :,
        iy0:iy1,
        ix0:ix1
    ]


    # ----------------------------------------------
    # Match SupportEncoder representation:
    #
    # spatial tokens
    # -> LayerNorm
    # -> mean
    # -> L2 normalize
    # ----------------------------------------------

    tokens = (
        roi
        .flatten(2)
        .transpose(1, 2)
    )

    norm_layer = (
        model
        .support_encoder
        .prototype_extractor
        .norm
    )

    tokens = norm_layer(
        tokens
    )

    prototype = (
        tokens.mean(
            dim=1
        )
    )

    prototype = F.normalize(
        prototype,
        p=2,
        dim=-1
    )

    return prototype


# ==========================================================
# Helper:
# synthetically shrink support inside same 640x640 canvas
#
# Diagnostic only.
# This does NOT change training pipeline.
# ==========================================================

def scale_support(
    support,
    scale
):

    B, C, H, W = (
        support.shape
    )

    new_h = max(
        1,
        int(
            H * scale
        )
    )

    new_w = max(
        1,
        int(
            W * scale
        )
    )

    resized = F.interpolate(
        support,
        size=(
            new_h,
            new_w
        ),
        mode="bilinear",
        align_corners=False
    )


    # Zero in normalized space ≈ ImageNet mean.
    # Controlled neutral canvas for diagnosis.
    canvas = torch.zeros_like(
        support
    )

    top = (
        H - new_h
    ) // 2

    left = (
        W - new_w
    ) // 2

    canvas[
        :,
        :,
        top:top + new_h,
        left:left + new_w
    ] = resized

    return canvas


# ==========================================================
# Containers
# ==========================================================

correct_roi_sim = []
wrong_roi_sim = []

representation_margin = []

correct_beats_wrong = []

scale_similarity = {
    scale: []
    for scale in TEST_SCALES
}

roi_cell_counts = []


# ==========================================================
# Diagnostic
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(
        range(NUM_EPISODES),
        desc="H2 Representation"
    ):

        episode = (
            val_dataset[
                episode_id
            ]
        )

        episode_class = int(
            episode[
                "episode_class"
            ].item()
        )


        # ==============================================
        # Correct support
        # ==============================================

        support = (
            episode[
                "support_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )


        # ==============================================
        # Query
        # ==============================================

        query = (
            episode[
                "query_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )

        gt_boxes = (
            episode[
                "query_target"
            ][
                "boxes"
            ]
            .to(
                CONFIG["device"]
            )
        )

        if gt_boxes.shape[0] == 0:
            continue


        # ==============================================
        # Wrong-class support
        # ==============================================

        wrong_idx = (
            find_wrong_support_index(
                val_dataset,
                episode_id,
                episode_class
            )
        )

        wrong_episode = (
            val_dataset[
                wrong_idx
            ]
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


        # ==============================================
        # Support prototypes
        # ==============================================

        correct_proto = (
            extract_support_prototype(
                support
            )
        )

        wrong_proto = (
            extract_support_prototype(
                wrong_support
            )
        )


        # ==============================================
        # Full query feature map
        #
        # expected:
        # [1,256,20,20]
        # ==============================================

        query_feature = (
            model.backbone(
                query
            )
        )


        _, _, feat_h, feat_w = (
            query_feature.shape
        )


        # ==============================================
        # Compare EACH target GT object
        # ==============================================

        for gt_box in gt_boxes:

            query_roi_proto = (
                extract_query_roi_prototype(
                    query_feature,
                    gt_box
                )
            )


            # ------------------------------------------
            # Correct support vs target ROI
            # ------------------------------------------

            sim_correct = (
                F.cosine_similarity(
                    correct_proto,
                    query_roi_proto,
                    dim=-1
                )
                .item()
            )


            # ------------------------------------------
            # Wrong support vs SAME target ROI
            # ------------------------------------------

            sim_wrong = (
                F.cosine_similarity(
                    wrong_proto,
                    query_roi_proto,
                    dim=-1
                )
                .item()
            )


            correct_roi_sim.append(
                sim_correct
            )

            wrong_roi_sim.append(
                sim_wrong
            )

            margin = (
                sim_correct
                -
                sim_wrong
            )

            representation_margin.append(
                margin
            )

            correct_beats_wrong.append(
                float(
                    sim_correct
                    >
                    sim_wrong
                )
            )


            # ------------------------------------------
            # Approx ROI size in 20x20 feature grid
            # ------------------------------------------

            cx, cy, bw, bh = (
                gt_box.tolist()
            )

            roi_cells = (
                max(
                    1,
                    int(
                        np.ceil(
                            bw * feat_w
                        )
                    )
                )
                *
                max(
                    1,
                    int(
                        np.ceil(
                            bh * feat_h
                        )
                    )
                )
            )

            roi_cell_counts.append(
                roi_cells
            )


        # ==============================================
        # Support scale stability
        # ==============================================

        for scale in TEST_SCALES:

            scaled_support = (
                scale_support(
                    support,
                    scale
                )
            )

            scaled_proto = (
                extract_support_prototype(
                    scaled_support
                )
            )

            sim_scale = (
                F.cosine_similarity(
                    correct_proto,
                    scaled_proto,
                    dim=-1
                )
                .item()
            )

            scale_similarity[
                scale
            ].append(
                sim_scale
            )


# ==========================================================
# Convert arrays
# ==========================================================

correct_roi_sim = np.asarray(
    correct_roi_sim,
    dtype=np.float64
)

wrong_roi_sim = np.asarray(
    wrong_roi_sim,
    dtype=np.float64
)

representation_margin = np.asarray(
    representation_margin,
    dtype=np.float64
)

correct_beats_wrong = np.asarray(
    correct_beats_wrong,
    dtype=np.float64
)

roi_cell_counts = np.asarray(
    roi_cell_counts,
    dtype=np.float64
)


# ==========================================================
# Result
# ==========================================================

print()
print("=" * 70)
print("H2 REPRESENTATION RESULT")
print("=" * 70)

print(
    "Episodes tested             :",
    NUM_EPISODES
)

print(
    "GT regions compared         :",
    len(
        correct_roi_sim
    )
)

print("-" * 70)

print(
    f"Correct support ↔ ROI sim   : "
    f"{correct_roi_sim.mean():.6f}"
)

print(
    f"Wrong support ↔ ROI sim     : "
    f"{wrong_roi_sim.mean():.6f}"
)

print(
    f"Mean correct-wrong margin   : "
    f"{representation_margin.mean():.6f}"
)

print(
    f"Correct > wrong fraction    : "
    f"{correct_beats_wrong.mean():.6f}"
)

print("-" * 70)

print(
    f"Median query ROI cells      : "
    f"{np.median(roi_cell_counts):.2f}"
)

print(
    f"Q25 query ROI cells         : "
    f"{np.quantile(roi_cell_counts, 0.25):.2f}"
)

print(
    f"Q75 query ROI cells         : "
    f"{np.quantile(roi_cell_counts, 0.75):.2f}"
)

print("-" * 70)

for scale in TEST_SCALES:

    values = np.asarray(
        scale_similarity[
            scale
        ],
        dtype=np.float64
    )

    print(
        f"Prototype sim original vs "
        f"{scale:.2f} scale : "
        f"{values.mean():.6f}"
    )

print("=" * 70)
