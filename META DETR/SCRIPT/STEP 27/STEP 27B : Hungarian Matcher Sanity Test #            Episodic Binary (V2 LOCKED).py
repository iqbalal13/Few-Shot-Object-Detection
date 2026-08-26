# ==========================================================
# STEP 27B : Hungarian Matcher Sanity Test
#            Episodic Binary (V2 LOCKED)
# ==========================================================

import torch


print("=" * 70)

print(
    "STEP 27B : HUNGARIAN MATCHER "
    "SANITY TEST"
)

print("=" * 70)


device = (
    CONFIG["device"]
)


# ==========================================================
# Synthetic Predictions
#
# Query 0:
# high foreground confidence
# perfect match GT 0
#
# Query 1:
# high foreground confidence
# perfect match GT 1
#
# Query 2-3:
# low foreground confidence
# wrong boxes
# ==========================================================

synthetic_outputs = {

    "pred_logits":

        torch.tensor(

            [
                [
                    [6.0],
                    [6.0],
                    [-6.0],
                    [-6.0]
                ]
            ],

            dtype=
                torch.float32,

            device=
                device
        ),


    "pred_boxes":

        torch.tensor(

            [
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

                ]
            ],

            dtype=
                torch.float32,

            device=
                device
        )
}


# ==========================================================
# Synthetic Ground Truth
#
# Dua object GT.
#
# labels tetap semantic metadata.
# Matcher V2 tidak menggunakan label semantic
# sebagai classification output.
# ==========================================================

synthetic_targets = [

    {

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

                dtype=
                    torch.float32,

                device=
                    device
            ),


        "labels":

            torch.tensor(

                [
                    0,
                    0
                ],

                dtype=
                    torch.long,

                device=
                    device
            )
    }
]


# ==========================================================
# Sanity Check:
# Utility Functions Must Already Exist
# ==========================================================

assert callable(
    box_cxcywh_to_xyxy
), (
    "box_cxcywh_to_xyxy "
    "not available. "
    "Run STEP 25 first."
)


assert callable(
    box_iou
), (
    "box_iou "
    "not available. "
    "Run STEP 26 first."
)


assert callable(
    generalized_box_iou
), (
    "generalized_box_iou "
    "not available. "
    "Run STEP 27 first."
)


# ==========================================================
# Run Matcher
# ==========================================================

matched_indices = matcher(

    synthetic_outputs,

    synthetic_targets
)


assert (
    len(
        matched_indices
    )
    ==
    1
)


src_idx, tgt_idx = (
    matched_indices[
        0
    ]
)


# ==========================================================
# Shape / Count Check
# ==========================================================

assert (
    len(
        src_idx
    )
    ==
    2
), (
    "Expected exactly "
    "2 matched queries."
)


assert (
    len(
        tgt_idx
    )
    ==
    2
), (
    "Expected exactly "
    "2 matched targets."
)


# ==========================================================
# Expected Matching
#
# query 0 -> GT 0
# query 1 -> GT 1
# ==========================================================

assert (
    set(
        src_idx.tolist()
    )
    ==
    {
        0,
        1
    }
), (
    "Unexpected matched "
    "query indices."
)


assert (
    set(
        tgt_idx.tolist()
    )
    ==
    {
        0,
        1
    }
), (
    "Unexpected matched "
    "target indices."
)


# ==========================================================
# Explicit Pair Verification
#
# Hungarian order secara teoritis dapat ditampilkan
# dalam urutan tertentu, jadi dibuat dictionary.
# ==========================================================

matching_pairs = {

    int(
        src
    ):
        int(
            tgt
        )

    for src, tgt
    in zip(

        src_idx.tolist(),

        tgt_idx.tolist()
    )
}


assert (
    matching_pairs[
        0
    ]
    ==
    0
), (
    "Query 0 should "
    "match GT 0."
)


assert (
    matching_pairs[
        1
    ]
    ==
    1
), (
    "Query 1 should "
    "match GT 1."
)


# ==========================================================
# Output
# ==========================================================

print(
    "Matched Query Indices :",
    src_idx.tolist()
)


print(
    "Matched Target Indices:",
    tgt_idx.tolist()
)


print(
    "Matching Pairs        :",
    matching_pairs
)


print("-" * 70)


print(
    "Expected:"
)

print(
    "Query 0 -> Target 0"
)

print(
    "Query 1 -> Target 1"
)


print("=" * 70)

print(
    "✓ STEP 27B PASSED"
)

print(
    "✓ Hungarian Matcher "
    "can now be used by STEP 28"
)

print("=" * 70)
