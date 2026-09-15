# ==========================================================
# STEP 22 : Generic Support-Conditioned Episodic Objective
#           + Padding-Mask Validation
#
# Episode semantic class can be any of 80 COCO classes.
# Detection target remains:
# matched support category = foreground.
# ==========================================================

# Remove obsolete definitions if a notebook runtime
# accidentally contains cells from the old notebook.
for old_name in (
    "sample_absent_wrong_support",
    "MatchedQuerySupportRankingLoss",
    "support_rank_criterion",
    "compute_combined_training_loss",
):
    globals().pop(
        old_name,
        None
    )


def validate_episodic_batch(
    batch
):

    required_batch_keys = (
        "support_images",
        "support_padding_masks",
        "query_images",
        "query_padding_masks",
        "episode_classes",
        "support_targets",
        "query_targets",
    )

    for key in required_batch_keys:
        if key not in batch:
            raise KeyError(
                f"Missing episodic batch key: {key}"
            )

    episode_classes = (
        batch[
            "episode_classes"
        ]
    )

    if episode_classes.ndim != 1:
        raise ValueError(
            "episode_classes must be 1-D."
        )

    batch_size = len(
        episode_classes
    )

    if (
        batch[
            "support_images"
        ].shape[0]
        !=
        batch_size
        or
        batch[
            "query_images"
        ].shape[0]
        !=
        batch_size
    ):
        raise ValueError(
            "Image batch size mismatch."
        )

    if (
        batch[
            "support_padding_masks"
        ].shape[0]
        !=
        batch_size
        or
        batch[
            "query_padding_masks"
        ].shape[0]
        !=
        batch_size
    ):
        raise ValueError(
            "Padding-mask batch size mismatch."
        )

    if (
        batch[
            "support_padding_masks"
        ].dtype
        !=
        torch.bool
        or
        batch[
            "query_padding_masks"
        ].dtype
        !=
        torch.bool
    ):
        raise TypeError(
            "Padding masks must be bool tensors."
        )

    for batch_index in range(
        batch_size
    ):

        semantic_label = int(
            episode_classes[
                batch_index
            ].item()
        )

        if not (
            0
            <= semantic_label
            <
            CONFIG[
                "source_num_categories"
            ]
        ):
            raise ValueError(
                "Invalid source semantic "
                f"label: {semantic_label}"
            )

        support_target = (
            batch[
                "support_targets"
            ][
                batch_index
            ]
        )

        query_target = (
            batch[
                "query_targets"
            ][
                batch_index
            ]
        )

        if (
            int(
                support_target[
                    "semantic_label"
                ].item()
            )
            !=
            semantic_label
        ):
            raise ValueError(
                "Support semantic label mismatch."
            )

        if (
            int(
                query_target[
                    "semantic_label"
                ].item()
            )
            !=
            semantic_label
        ):
            raise ValueError(
                "Query semantic label mismatch."
            )

        if not bool(
            (
                query_target[
                    "labels"
                ]
                ==
                0
            ).all()
        ):
            raise ValueError(
                "Episodic query detection labels must be "
                "support-relative foreground encoding."
            )

    return True


print("=" * 70)
print("STEP 22 : GENERIC EPISODIC OBJECTIVE READY")
print("=" * 70)
print("Source semantic tasks : 80 COCO categories")
print("Classifier            : support-match binary")
print("Padding masks         : validated")
print("Ranking auxiliary loss: not used")
print("=" * 70)
