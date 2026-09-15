# ==========================================================
# STEP 22 : Generic Support-Conditioned Episodic Objective
#
# Episode semantic class can be any of 80 COCO classes.
#
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

    episode_classes = (
        batch[
            "episode_classes"
        ]
    )

    if episode_classes.ndim != 1:

        raise ValueError(
            "episode_classes must "
            "be 1-D."
        )

    for batch_index in range(
        len(
            episode_classes
        )
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
                "Support semantic label "
                "mismatch."
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
                "Query semantic label "
                "mismatch."
            )

        # Detection labels are always
        # support-relative foreground encoding.
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
                "Episodic query detection "
                "labels must be binary-relative "
                "foreground encoding."
            )

    return True


print("=" * 70)
print("STEP 22 : GENERIC EPISODIC OBJECTIVE READY")
print("=" * 70)

print(
    "Source semantic tasks : 80 COCO categories"
)

print(
    "Classifier            : support-match binary"
)

print(
    "Ranking auxiliary loss: not used"
)

print("=" * 70)
