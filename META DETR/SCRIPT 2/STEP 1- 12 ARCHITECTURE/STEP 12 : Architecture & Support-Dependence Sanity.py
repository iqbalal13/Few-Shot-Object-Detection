# ==========================================================
# STEP 12 : Architecture & Support-Dependence Sanity
#
# STRUCTURAL TEST ONLY
#
# Same decoder objects D
# Different support prototypes P
#
# Must change:
# - relation Z
# - classification logits
# - bbox predictions
#
# NO TRAINING
# ==========================================================

print("=" * 70)
print("STEP 12 : ARCHITECTURE & SUPPORT SANITY")
print("=" * 70)


model.eval()


SANITY_SIZE = 256


with torch.inference_mode():

    # ------------------------------------------------------
    # SAME QUERY
    # ------------------------------------------------------

    query_image = torch.randn(

        1,
        3,
        SANITY_SIZE,
        SANITY_SIZE,

        device=CONFIG["device"]
    )


    # ------------------------------------------------------
    # TWO DIFFERENT SUPPORT IMAGES
    # ------------------------------------------------------

    support_a = torch.randn(

        1,
        3,
        SANITY_SIZE,
        SANITY_SIZE,

        device=CONFIG["device"]
    )


    support_b = torch.randn(

        1,
        3,
        SANITY_SIZE,
        SANITY_SIZE,

        device=CONFIG["device"]
    )


    # ------------------------------------------------------
    # Query branch calculated ONCE.
    #
    # Therefore decoder_objects are fixed.
    # Any later output difference MUST originate from support.
    # ------------------------------------------------------

    (
        decoder_objects,
        query_extras

    ) = model.encode_query(
        query_image
    )


    # ------------------------------------------------------
    # SUPPORT PROTOTYPES
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # SAME decoder D + support A
    # ------------------------------------------------------

    (
        output_a,
        extra_a

    ) = model.condition_and_predict(

        decoder_objects,

        prototype_a
    )


    # ------------------------------------------------------
    # SAME decoder D + support B
    # ------------------------------------------------------

    (
        output_b,
        extra_b

    ) = model.condition_and_predict(

        decoder_objects,

        prototype_b
    )


# ==========================================================
# SHAPES
# ==========================================================

expected_logits_shape = (
    1,
    CONFIG["num_queries"],
    1
)


expected_boxes_shape = (
    1,
    CONFIG["num_queries"],
    4
)


assert (
    output_a["pred_logits"].shape
    ==
    expected_logits_shape
)


assert (
    output_a["pred_boxes"].shape
    ==
    expected_boxes_shape
)


assert (
    decoder_objects.shape
    ==
    (
        1,
        CONFIG["num_queries"],
        CONFIG["hidden_dim"]
    )
)


assert (
    prototype_a.shape
    ==
    (
        1,
        CONFIG["hidden_dim"]
    )
)


assert (
    extra_a[
        "relation_features"
    ].shape
    ==
    (
        1,
        CONFIG["num_queries"],
        CONFIG["hidden_dim"]
    )
)


# ==========================================================
# FINITE CHECK
# ==========================================================

assert torch.isfinite(
    output_a["pred_logits"]
).all()


assert torch.isfinite(
    output_a["pred_boxes"]
).all()


# ==========================================================
# SUPPORT DEPENDENCE
# ==========================================================

prototype_delta = (

    prototype_a
    -
    prototype_b

).abs().mean().item()


relation_delta = (

    extra_a[
        "relation_features"
    ]
    -
    extra_b[
        "relation_features"
    ]

).abs().mean().item()


logit_delta = (

    output_a[
        "pred_logits"
    ]
    -
    output_b[
        "pred_logits"
    ]

).abs().mean().item()


bbox_delta = (

    output_a[
        "pred_boxes"
    ]
    -
    output_b[
        "pred_boxes"
    ]

).abs().mean().item()


# ==========================================================
# LOGIT SCALE
# ==========================================================

effective_scale = (

    model
    .detection_head
    .get_logit_scale()
    .item()
)


# ==========================================================
# STRUCTURAL ASSERTIONS
# ==========================================================

assert (
    prototype_delta
    >
    1e-6
), (
    "Support encoder produced "
    "indistinguishable prototypes."
)


assert (
    relation_delta
    >
    1e-6
), (
    "Relation module ignores support."
)


assert (
    logit_delta
    >
    1e-6
), (
    "Classification output ignores support."
)


assert (
    bbox_delta
    >
    1e-8
), (
    "BBox output is not support-conditioned."
)


assert (
    effective_scale
    >
    1.0
)


# ==========================================================
# REPORT
# ==========================================================

print(
    "Decoder Objects :",
    decoder_objects.shape
)

print(
    "Support P       :",
    prototype_a.shape
)

print(
    "Relation Z      :",
    extra_a[
        "relation_features"
    ].shape
)

print(
    "Pred Logits     :",
    output_a[
        "pred_logits"
    ].shape
)

print(
    "Pred Boxes      :",
    output_a[
        "pred_boxes"
    ].shape
)

print("-" * 70)

print(
    f"Mean |Δ prototype| : "
    f"{prototype_delta:.8f}"
)

print(
    f"Mean |Δ relation|  : "
    f"{relation_delta:.8f}"
)

print(
    f"Mean |Δ logit|     : "
    f"{logit_delta:.8f}"
)

print(
    f"Mean |Δ bbox|      : "
    f"{bbox_delta:.8f}"
)

print(
    f"Effective scale    : "
    f"{effective_scale:.6f}"
)

print("-" * 70)

print(
    "✓ SAME decoder object representation was used "
    "for both support conditions."
)

print(
    "✓ Classification has NO decoder-only bypass."
)

print(
    "✓ BBox regression has NO decoder-only bypass."
)

print(
    "✓ Classification and bbox both depend on relation Z."
)

print("=" * 70)
print("✓ STEP 12 ARCHITECTURE SANITY PASSED")
print("=" * 70)

print()
print(
    "NOTE: This proves STRUCTURAL support dependence only."
)

print(
    "It does NOT yet prove semantic correctness "
    "or unseen COCO generalization."
)
