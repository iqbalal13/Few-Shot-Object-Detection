# ==========================================================
# STEP 23 : Combined Loss & Matcher Sanity
#
# Tests:
# 1. Synthetic Hungarian matching
# 2. Real COCO episode
# 3. Detection loss
# 4. Absent wrong support
# 5. Matched-query support ranking
# 6. Combined objective finite
# ==========================================================


# ==========================================================
# TARGET DEVICE HELPER
# Kept globally for later training/evaluation.
# ==========================================================

def move_targets_to_device(
    targets,
    device
):

    moved = []


    for target in targets:

        moved_target = {}


        for key, value in (
            target.items()
        ):

            if torch.is_tensor(
                value
            ):

                moved_target[key] = (
                    value.to(
                        device
                    )
                )

            else:

                moved_target[key] = (
                    value
                )


        moved.append(
            moved_target
        )


    return moved


print("=" * 70)
print("STEP 23 : COMBINED LOSS & MATCHER SANITY")
print("=" * 70)


device = CONFIG["device"]


# ==========================================================
# PART A — SYNTHETIC MATCHER
# ==========================================================

synthetic_outputs = {

    "pred_logits":
        torch.tensor(

            [[
                [6.0],
                [6.0],
                [-6.0],
                [-6.0]
            ]],

            dtype=torch.float32,

            device=device
        ),


    "pred_boxes":
        torch.tensor(

            [[

                [
                    0.20,
                    0.20,
                    0.10,
                    0.10
                ],

                [
                    0.80,
                    0.80,
                    0.10,
                    0.10
                ],

                [
                    0.50,
                    0.50,
                    0.30,
                    0.30
                ],

                [
                    0.10,
                    0.90,
                    0.20,
                    0.20
                ]

            ]],

            dtype=torch.float32,

            device=device
        )
}


synthetic_targets = [{

    "boxes":
        torch.tensor(

            [
                [
                    0.20,
                    0.20,
                    0.10,
                    0.10
                ],

                [
                    0.80,
                    0.80,
                    0.10,
                    0.10
                ]
            ],

            dtype=torch.float32,

            device=device
        ),

    "labels":
        torch.tensor(

            [0, 0],

            dtype=torch.long,

            device=device
        )
}]


synthetic_indices = matcher(

    synthetic_outputs,

    synthetic_targets
)


src_idx, tgt_idx = (
    synthetic_indices[0]
)


matching_pairs = {

    int(src):
        int(tgt)

    for src, tgt
    in zip(

        src_idx.tolist(),

        tgt_idx.tolist()
    )
}


assert matching_pairs == {
    0: 0,
    1: 1
}


print(
    "Synthetic Matcher : ✓ PASS"
)


# ==========================================================
# PART B — REAL COCO EPISODE
# ==========================================================

batch = next(
    iter(train_loader)
)


support_images = (

    batch[
        "support_images"
    ]

    .to(
        device,
        non_blocking=True
    )
)


query_images = (

    batch[
        "query_images"
    ]

    .to(
        device,
        non_blocking=True
    )
)


query_targets = (

    move_targets_to_device(

        batch[
            "query_targets"
        ],

        device
    )
)


current_class = int(

    batch[
        "episode_classes"
    ][0]
    .item()
)


query_image_id = int(

    batch[
        "query_targets"
    ][0][
        "image_id"
    ]
    .item()
)


# ==========================================================
# CORRECT SUPPORT FORWARD
# ==========================================================

model.eval()


with torch.inference_mode():

    (
        outputs,
        extras
    ) = model.forward_with_features(

        support_image=
            support_images,

        query_image=
            query_images
    )


    # ------------------------------------------------------
    # Hungarian matching ON CORRECT SUPPORT predictions
    # ------------------------------------------------------

    indices = matcher(

        outputs,

        query_targets
    )


    # ------------------------------------------------------
    # Detection objective uses SAME indices
    # ------------------------------------------------------

    detection_losses = criterion(

        outputs,

        query_targets,

        indices=indices
    )


    # ------------------------------------------------------
    # Select absent wrong support
    # ------------------------------------------------------

    wrong_episode = (
        sample_absent_wrong_support(

            dataset=
                train_dataset,

            query_image_id=
                query_image_id,

            current_class=
                current_class,

            epoch=0,

            step=0
        )
    )


    wrong_support_images = (

        wrong_episode[
            "image"
        ]

        .unsqueeze(0)

        .to(
            device,
            non_blocking=True
        )
    )


    # ------------------------------------------------------
    # Matched-query ranking objective
    # ------------------------------------------------------

    (
        support_rank_loss,
        support_rank_stats
    ) = support_rank_criterion(

        model=
            model,

        decoder_objects=
            extras[
                "decoder_objects"
            ],

        correct_similarity=
            extras[
                "support_similarity"
            ],

        wrong_support_images=
            wrong_support_images,

        indices=
            indices
    )


    combined_loss = (

        detection_losses[
            "loss_total"
        ]

        +

        CONFIG[
            "support_rank_weight"
        ]

        *
        support_rank_loss
    )


# ==========================================================
# SAFETY
# ==========================================================

for name, value in {

    **detection_losses,

    "loss_support_rank":
        support_rank_loss,

    "loss_combined":
        combined_loss

}.items():

    assert torch.isfinite(
        value
    ), (
        f"{name} is NaN/Inf."
    )


assert (
    wrong_episode[
        "class"
    ]
    not in
    wrong_episode[
        "present_labels"
    ]
)


assert (
    len(
        indices[0][0]
    )
    ==
    len(
        query_targets[0][
            "boxes"
        ]
    )
)


# ==========================================================
# REPORT
# ==========================================================

print("-" * 70)

print(
    "Episode Class :",
    current_class,
    CATEGORY_NAMES[
        current_class
    ]
)

print(
    "Wrong Class   :",
    wrong_episode[
        "class"
    ],
    CATEGORY_NAMES[
        wrong_episode[
            "class"
        ]
    ]
)

print(
    "GT Objects    :",
    len(
        query_targets[0][
            "boxes"
        ]
    )
)

print(
    "Matched Query :",
    len(
        indices[0][0]
    )
)

print("-" * 70)

print(
    f"Focal Loss    : "
    f"{detection_losses['loss_cls'].item():.6f}"
)

print(
    f"BBox Loss     : "
    f"{detection_losses['loss_bbox'].item():.6f}"
)

print(
    f"GIoU Loss     : "
    f"{detection_losses['loss_giou'].item():.6f}"
)

print(
    f"Detection Loss: "
    f"{detection_losses['loss_total'].item():.6f}"
)

print(
    f"Support Rank  : "
    f"{support_rank_loss.item():.6f}"
)

print(
    f"Combined Loss : "
    f"{combined_loss.item():.6f}"
)

print("-" * 70)

print(
    f"Correct Sim   : "
    f"{support_rank_stats['correct_similarity']:.6f}"
)

print(
    f"Wrong Sim     : "
    f"{support_rank_stats['wrong_similarity']:.6f}"
)

print(
    f"Observed Margin: "
    f"{support_rank_stats['observed_margin']:.6f}"
)

print(
    f"Required Margin: "
    f"{CONFIG['support_rank_margin']:.6f}"
)

print("-" * 70)

print(
    "✓ Wrong support is absent from query."
)

print(
    "✓ Detection and ranking use the SAME matched queries."
)

print(
    "✓ Ranking operates in actual detector-query space."
)

print("=" * 70)
print("✓ STEP 23 PASSED")
print("=" * 70)
