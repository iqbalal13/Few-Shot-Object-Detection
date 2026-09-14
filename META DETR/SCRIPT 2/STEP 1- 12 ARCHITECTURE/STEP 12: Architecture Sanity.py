# ==========================================================
# STEP 12 : Architecture Sanity
#
# Checks:
# 1. Final architecture = 3E / 6D / 100 queries
# 2. Output shapes
# 3. Finite values
# 4. Bounding boxes in [0,1]
# 5. Support changes prototype/relation/logits
#
# NOTE:
# Support sensitivity only proves computational dependency.
# It does NOT yet prove useful meta-learning.
# ==========================================================

model.eval()

with torch.inference_mode():

    query = torch.randn(
        1,
        3,
        256,
        256,
        device=CONFIG["device"]
    )

    support_a = torch.randn_like(
        query
    )

    support_b = torch.randn_like(
        query
    )

    (
        decoder_objects,
        _
    ) = model.encode_query(
        query
    )

    prototype_a = (
        model.encode_support(
            support_a
        )
    )

    prototype_b = (
        model.encode_support(
            support_b
        )
    )

    (
        output_a,
        extra_a

    ) = model.condition_and_predict(

        decoder_objects,
        prototype_a
    )

    (
        output_b,
        extra_b

    ) = model.condition_and_predict(

        decoder_objects,
        prototype_b
    )

    # ------------------------------------------------------
    # Architecture
    # ------------------------------------------------------

    assert (
        len(
            model
            .transformer_encoder
            .layers
        )
        == 3
    )

    assert (
        len(
            model
            .transformer_decoder
            .layers
        )
        == 6
    )

    assert (
        model
        .transformer_decoder
        .num_queries
        == 100
    )

    # ------------------------------------------------------
    # Output shapes
    # ------------------------------------------------------

    assert (
        output_a[
            "pred_logits"
        ].shape
        ==
        (
            1,
            CONFIG[
                "num_queries"
            ],
            1,
        )
    )

    assert (
        output_a[
            "pred_boxes"
        ].shape
        ==
        (
            1,
            CONFIG[
                "num_queries"
            ],
            4,
        )
    )

    # Important:
    # final classifier is binary support match,
    # NOT an 80-dimensional classifier.
    assert (
        output_a[
            "pred_logits"
        ].shape[-1]
        == 1
    )

    # ------------------------------------------------------
    # Finite / valid boxes
    # ------------------------------------------------------

    for output in (
        output_a,
        output_b
    ):

        for value in output.values():

            assert torch.isfinite(
                value
            ).all()

        boxes = output[
            "pred_boxes"
        ]

        assert (
            (boxes >= 0.0)
            &
            (boxes <= 1.0)
        ).all()

    # ------------------------------------------------------
    # Support dependency
    # ------------------------------------------------------

    deltas = {}

    for name, a, b in (

        (
            "prototype",
            prototype_a,
            prototype_b,
        ),

        (
            "relation",
            extra_a[
                "relation_features"
            ],
            extra_b[
                "relation_features"
            ],
        ),

        (
            "logits",
            output_a[
                "pred_logits"
            ],
            output_b[
                "pred_logits"
            ],
        ),

        (
            "bbox",
            output_a[
                "pred_boxes"
            ],
            output_b[
                "pred_boxes"
            ],
        ),
    ):

        delta = (
            (a - b)
            .abs()
            .mean()
            .item()
        )

        deltas[name] = delta

        print(
            f"{name:10s}: "
            f"mean absolute change "
            f"= {delta:.8f}"
        )

    assert (
        deltas["prototype"]
        > 0.0
    )

    assert (
        deltas["relation"]
        > 0.0
    )

    assert (
        deltas["logits"]
        > 0.0
    )


print("=" * 70)
print("STEP 12 PASS")
print("=" * 70)

print(
    "Architecture       : 3 encoder / "
    "6 decoder / 100 queries"
)

print(
    "Classifier output  : binary "
    "support-match"
)

print(
    "Support dependency : present"
)

print(
    "Reminder           : dependency != "
    "useful meta-learning"
)

print("=" * 70)


del query
del support_a
del support_b

del decoder_objects

del prototype_a
del prototype_b

del output_a
del output_b

del extra_a
del extra_b

del deltas

if torch.cuda.is_available():
    torch.cuda.empty_cache()
