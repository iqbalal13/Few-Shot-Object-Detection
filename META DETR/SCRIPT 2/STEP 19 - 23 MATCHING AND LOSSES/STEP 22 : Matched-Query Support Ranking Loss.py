# ==========================================================
# STEP 22 : Matched-Query Support Ranking Loss
#
# For Hungarian-matched detector queries:
#
# sim(query, correct support)
# >
# sim(query, absent wrong support) + margin
#
# Same decoder objects are used under both supports.
# ==========================================================


# ==========================================================
# ABSENT WRONG SUPPORT SAMPLER
# ==========================================================

def sample_absent_wrong_support(
    dataset,
    query_image_id,
    current_class,
    epoch,
    step
):

    present_labels = (
        dataset.get_present_labels(
            query_image_id
        )
    )


    # Positive episode class is never eligible.

    present_labels.add(
        int(current_class)
    )


    candidate_labels = [

        int(label)

        for label
        in dataset.valid_labels

        if int(label)
        not in present_labels
    ]


    if not candidate_labels:

        raise RuntimeError(

            "No absent semantic class "
            "available for wrong support."
        )


    rng = random.Random(

        int(CONFIG["seed"])

        +
        2_000_000

        +
        int(epoch)
        *
        100_000

        +
        int(step)
    )


    wrong_class = int(

        rng.choice(
            candidate_labels
        )
    )


    wrong_ann_id = rng.choice(

        dataset.class_to_ann_ids[
            wrong_class
        ]
    )


    (
        wrong_support_image,
        wrong_support_target
    ) = dataset._load_support(

        wrong_ann_id,

        wrong_class
    )


    assert (
        wrong_class
        not in
        present_labels
    )


    return {

        "image":
            wrong_support_image,

        "target":
            wrong_support_target,

        "class":
            wrong_class,

        "present_labels":
            present_labels
    }


# ==========================================================
# RANKING LOSS
# ==========================================================

class MatchedQuerySupportRankingLoss(
    nn.Module
):

    def __init__(
        self,
        margin=
            CONFIG[
                "support_rank_margin"
            ]
    ):
        super().__init__()


        self.margin = float(
            margin
        )


    def forward(
        self,
        model,
        decoder_objects,
        correct_similarity,
        wrong_support_images,
        indices
    ):

        # --------------------------------------------------
        # Wrong support prototype
        # --------------------------------------------------

        wrong_prototype = (
            model.encode_support(
                wrong_support_images
            )
        )


        # --------------------------------------------------
        # SAME decoder objects,
        # but conditioned on WRONG support.
        # --------------------------------------------------

        (
            _,
            wrong_extras
        ) = model.condition_and_predict(

            decoder_objects,

            wrong_prototype
        )


        wrong_similarity = (
            wrong_extras[
                "support_similarity"
            ]
        )


        correct_values = []
        wrong_values = []


        for b, (
            src_idx,
            _
        ) in enumerate(
            indices
        ):

            if len(src_idx) == 0:
                continue


            correct_values.append(

                correct_similarity[
                    b,
                    src_idx,
                    0
                ]
            )


            wrong_values.append(

                wrong_similarity[
                    b,
                    src_idx,
                    0
                ]
            )


        if not correct_values:

            zero = (
                correct_similarity.sum()
                *
                0.0
            )

            return (
                zero,
                {
                    "correct_similarity":
                        0.0,

                    "wrong_similarity":
                        0.0,

                    "observed_margin":
                        0.0,

                    "margin_satisfied_fraction":
                        0.0
                }
            )


        correct_values = torch.cat(
            correct_values,
            dim=0
        )


        wrong_values = torch.cat(
            wrong_values,
            dim=0
        )


        observed_margin = (

            correct_values
            -
            wrong_values
        )


        ranking_loss = F.relu(

            self.margin
            -
            observed_margin
        ).mean()


        stats = {

            "correct_similarity":
                float(
                    correct_values
                    .detach()
                    .mean()
                    .item()
                ),

            "wrong_similarity":
                float(
                    wrong_values
                    .detach()
                    .mean()
                    .item()
                ),

            "observed_margin":
                float(
                    observed_margin
                    .detach()
                    .mean()
                    .item()
                ),

            "margin_satisfied_fraction":
                float(
                    (
                        observed_margin
                        >=
                        self.margin
                    )
                    .float()
                    .detach()
                    .mean()
                    .item()
                )
        }


        return (
            ranking_loss,
            stats
        )


support_rank_criterion = (
    MatchedQuerySupportRankingLoss()
    .to(
        CONFIG["device"]
    )
)


print("=" * 70)
print("STEP 22 : MATCHED-QUERY SUPPORT RANKING READY")
print("=" * 70)

print(
    "Margin:",
    support_rank_criterion.margin
)

print(
    "Weight:",
    CONFIG[
        "support_rank_weight"
    ]
)

print("=" * 70)
