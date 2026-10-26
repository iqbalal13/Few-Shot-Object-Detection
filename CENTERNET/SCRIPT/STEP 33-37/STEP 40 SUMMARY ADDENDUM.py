
# ============================================================
# STEP 40 SUMMARY ADDENDUM — A3 VERSION
# A2 CONTROL VS A3 NEIGHBOR R1
# NO INFERENCE / NO TRAINING
# ============================================================

import json
import os

assert "summary40" in globals(), (
    "Run STEP 40 before this addendum."
)

assert "A2_control" in summary40
assert "A3_neighbor_R1" in summary40

GROUPS40 = (
    "all",
    "small",
    "medium",
    "large",
    "person"
)

RADII40 = (1, 2, 4)
LOCATIONS40 = ("all", "inside")

addendum40 = {
    "experiment": "A3_R1_PILOT",
    "comparison": "A3_neighbor_R1_minus_A2_control",
    "groups": {}
}

print("=" * 76)
print("STEP 40 ADDENDUM — A2 VS A3")
print("WH SPATIAL ROBUSTNESS")
print("=" * 76)

for group in GROUPS40:

    a2 = summary40["A2_control"][group]
    a3 = summary40["A3_neighbor_R1"][group]

    assert a2["gt"] == a3["gt"], (
        f"GT mismatch: {group}"
    )

    result = {
        "gt": int(a2["gt"]),
        "center": {},
        "neighbor_eligible": {},
        "radii": {}
    }

    print("\n" + "-" * 76)
    print(
        f"GROUP: {group.upper()} | "
        f"GT={a2['gt']}"
    )

    # ========================================================
    # CENTER WH IOU
    # ========================================================

    a2_center = a2["center_iou_all_gt"]
    a3_center = a3["center_iou_all_gt"]

    result["center"] = {
        "A2": a2_center,
        "A3": a3_center,
        "delta": a3_center - a2_center
    }

    print(
        f"WH @ GT center: "
        f"A2={a2_center:.6f} | "
        f"A3={a3_center:.6f} | "
        f"Delta={a3_center-a2_center:+.6f}"
    )

    # ========================================================
    # RADIUS 1, 2, 4
    # ========================================================

    for radius in RADII40:

        result["radii"][str(radius)] = {}

        for location in LOCATIONS40:

            prefix = f"r{radius}_{location}"

            count_key = f"{prefix}_n"
            mean_key = f"{prefix}_mean_iou"
            gap_key = f"{prefix}_gap_from_center"

            assert a2[count_key] == a3[count_key], (
                f"Sample mismatch: {group}, {prefix}"
            )

            a2_mean = a2[mean_key]
            a3_mean = a3[mean_key]

            if a2_mean is None or a3_mean is None:
                result["radii"][str(radius)][location] = {
                    "n": int(a2[count_key]),
                    "A2": None,
                    "A3": None,
                    "delta": None
                }
                continue

            delta = a3_mean - a2_mean

            a2_gap = a2[gap_key]
            a3_gap = a3[gap_key]

            result["radii"][str(radius)][location] = {
                "n": int(a2[count_key]),
                "A2": a2_mean,
                "A3": a3_mean,
                "delta": delta,
                "A2_gap_from_center": a2_gap,
                "A3_gap_from_center": a3_gap,
                "delta_gap_from_center": (
                    a3_gap - a2_gap
                    if a2_gap is not None
                    and a3_gap is not None
                    else None
                )
            }

            print(
                f"R{radius:<2} {location:<6} | "
                f"N={a2[count_key]:5d} | "
                f"A2={a2_mean:.6f} | "
                f"A3={a3_mean:.6f} | "
                f"Delta={delta:+.6f}"
            )

    # ========================================================
    # NEIGHBOR ELIGIBLE OBJECTS
    # ========================================================

    assert (
        a2["eligible_GT"]
        == a3["eligible_GT"]
    )

    eligible_n = int(a2["eligible_GT"])

    neighbor_a2 = a2["eligible_neighbor_iou"]
    neighbor_a3 = a3["eligible_neighbor_iou"]

    eligible_center_a2 = a2["eligible_center_iou"]
    eligible_center_a3 = a3["eligible_center_iou"]

    result["neighbor_eligible"] = {
        "gt": eligible_n,
        "neighbor_A2": neighbor_a2,
        "neighbor_A3": neighbor_a3,
        "neighbor_delta": (
            neighbor_a3 - neighbor_a2
            if neighbor_a2 is not None
            and neighbor_a3 is not None
            else None
        ),
        "center_A2": eligible_center_a2,
        "center_A3": eligible_center_a3,
        "center_delta": (
            eligible_center_a3 - eligible_center_a2
            if eligible_center_a2 is not None
            and eligible_center_a3 is not None
            else None
        )
    }

    print(
        f"\nEligible GT: {eligible_n}"
    )

    if neighbor_a2 is not None and neighbor_a3 is not None:
        print(
            f"Eligible-neighbor WH IoU: "
            f"A2={neighbor_a2:.6f} | "
            f"A3={neighbor_a3:.6f} | "
            f"Delta={neighbor_a3-neighbor_a2:+.6f}"
        )

    addendum40["groups"][group] = result


# ============================================================
# FINAL INTEGRITY CHECK
# ============================================================

assert (
    addendum40["groups"]["all"]["gt"]
    == 7117
), "Unexpected 1000-image GT population"

assert (
    addendum40["groups"]["all"]["neighbor_eligible"]["gt"]
    == 4159
), "Unexpected eligible-neighbor GT population"

# ============================================================
# SAVE
# ============================================================

OUTPUT40_ADDENDUM = os.path.join(
    A3_QC_DIR,
    "qc40_addendum_A2_vs_A3.json"
)

with open(
    OUTPUT40_ADDENDUM,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        addendum40,
        f,
        indent=2,
        allow_nan=False
    )

print("\n" + "=" * 76)
print("STEP 40 ADDENDUM PASS")
print("NO INFERENCE / NO TRAINING")
print("Saved:", OUTPUT40_ADDENDUM)
print("=" * 76)
