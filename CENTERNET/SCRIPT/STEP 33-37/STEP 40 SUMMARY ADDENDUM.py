
# ============================================================
# STEP 40 SUMMARY ADDENDUM — B2 VERSION (CORRECTED)
# A2 CONTROL VS B2 C3/C4 GATED FUSION
# NO TRAINING / NO GPU INFERENCE
# ============================================================

import os
import json
import math

# B2 Step 40 menghasilkan report40, bukan summary40.
assert "report40" in globals(), (
    "Step 40 B2 belum dijalankan: report40 tidak tersedia."
)
assert "B2_ARM" in globals(), (
    "Jalankan setup B2 / Step 30."
)
assert "QC_DIR" in globals(), (
    "Jalankan QC setup / Step 32."
)
assert callable(qc_save), (
    "Jalankan Step 32 untuk mendefinisikan qc_save()."
)

ARM_A2 = "A2_control"
ARM_B2 = B2_ARM

GROUPS40 = (
    "all",
    "small",
    "medium",
    "large",
    "person"
)

RADII40 = (1, 2, 4)
ZONES40 = ("all", "inside")

assert ARM_A2 in report40
assert ARM_B2 in report40


# ============================================================
# HELPER — PAIRED COMPARISON
# ============================================================

def make_pair40(a, b):
    if a is None or b is None:
        return {
            "A2": a,
            "B2": b,
            "delta_B2_minus_A2": None
        }

    a = float(a)
    b = float(b)

    assert math.isfinite(a)
    assert math.isfinite(b)

    return {
        "A2": a,
        "B2": b,
        "delta_B2_minus_A2": b - a
    }


# ============================================================
# SUMMARY STRUCTURE
# ============================================================

addendum40 = {
    "experiment": "B2_C3C4_GATED_PILOT_FROM_A2_160K",
    "comparison": "B2_minus_A2_control",
    "metric": "mean WH size-only IoU",
    "data": "1000 COCO-Val images, paired GT population",
    "groups": {},
    "notes": [
        "These are WH readout diagnostics, not official COCO AP.",
        "The B2 Step 40 contains group means, not separate "
        "inside-zone sample counts."
    ]
}

print("=" * 76)
print("STEP 40 SUMMARY ADDENDUM — B2 VS A2 CONTROL")
print("=" * 76)


# ============================================================
# GROUP ANALYSIS
# ============================================================

for group in GROUPS40:

    assert group in report40[ARM_A2], (
        f"A2 group missing: {group}"
    )

    assert group in report40[ARM_B2], (
        f"B2 group missing: {group}"
    )

    a2 = report40[ARM_A2][group]
    b2 = report40[ARM_B2][group]

    assert int(a2["GT"]) == int(b2["GT"]), (
        f"GT mismatch: {group}"
    )

    row = {
        "GT": int(a2["GT"]),
        "center": make_pair40(
            a2["center"],
            b2["center"]
        ),
        "radius": {}
    }

    print(f"\n{group.upper():<12} GT={row['GT']}")

    # --------------------------------------------------------
    # GT CENTER
    # --------------------------------------------------------

    center = row["center"]

    print(
        "  CENTER      "
        f"A2={center['A2']:.6f} | "
        f"B2={center['B2']:.6f} | "
        f"Delta={center['delta_B2_minus_A2']:+.6f}"
    )

    # --------------------------------------------------------
    # SPATIAL WH — RADIUS 1, 2, 4
    # --------------------------------------------------------

    for rad in RADII40:

        row["radius"][f"R{rad}"] = {}

        for zone in ZONES40:

            key = f"R{rad}_{zone}"

            assert key in a2, (
                f"Missing A2 field: {group}/{key}"
            )

            assert key in b2, (
                f"Missing B2 field: {group}/{key}"
            )

            pair = make_pair40(
                a2[key],
                b2[key]
            )

            row["radius"][f"R{rad}"][zone] = pair

            if pair["delta_B2_minus_A2"] is None:
                print(
                    f"  {key:<12} "
                    "no valid paired aggregate"
                )
            else:
                print(
                    f"  {key:<12} "
                    f"A2={pair['A2']:.6f} | "
                    f"B2={pair['B2']:.6f} | "
                    f"Delta="
                    f"{pair['delta_B2_minus_A2']:+.6f}"
                )

    addendum40["groups"][group] = row


# ============================================================
# INTEGRITY CHECK
# ============================================================

assert addendum40["groups"]["all"]["GT"] == 7117, (
    "Unexpected GT count. Check QC dataset and target filtering."
)


# ============================================================
# SAVE B2 REPORT
# ============================================================

OUTPUT40_ADDENDUM = qc_save(
    "qc40_addendum_A2_vs_B2.json",
    addendum40
)

print("\n" + "=" * 76)
print("STEP 40 ADDENDUM B2 PASS")
print("NO TRAINING / NO INFERENCE")
print("Saved:", OUTPUT40_ADDENDUM)
print("=" * 76)
