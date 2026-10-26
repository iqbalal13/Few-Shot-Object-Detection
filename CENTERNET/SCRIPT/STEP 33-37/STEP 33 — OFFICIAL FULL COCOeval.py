# ============================================================
# STEP 33-A3 — PILOT SUMMARY
# NO TRAINING / NO RE-EVALUATION
# Replaces incompatible B1 historical Step 33–41
# ============================================================

import os
import json

# ============================================================
# 1. VERIFY EXPERIMENT CONFIGURATION
# ============================================================

assert set(results31) == {
    "A2_control",
    "A3_neighbor_R1"
}

assert set(qc32) == {
    "A2_control",
    "A3_neighbor_R1"
}

assert (
    CONFIG["gaussian_radius_mode"]
    == "v1_2_original"
)

assert CONFIG["a3_neighbor_radius"] == 1
assert CONFIG["a3_neighbor_lambda"] == 0.25

for model_name in (
    "A2_control",
    "A3_neighbor_R1"
):
    path = paths[model_name]

    assert os.path.isfile(path), (
        f"Checkpoint missing: {path}"
    )

    assert results31[model_name]["images"] == 5000


# ============================================================
# 2. OFFICIAL COCOeval COMPARISON
# ============================================================

metric_keys = (
    "mAP_50_95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR100"
)

a = results31["A2_control"]["metrics"]
b = results31["A3_neighbor_R1"]["metrics"]

print("=" * 70)
print("A3 PILOT — OFFICIAL COCOeval")
print("A3 minus A2 continuation")
print("=" * 70)

deltas = {}

for key in metric_keys:
    deltas[key] = 100 * (b[key] - a[key])

    print(
        f"{key:<18}: "
        f"A2={100*a[key]:8.4f}% | "
        f"A3={100*b[key]:8.4f}% | "
        f"Delta={deltas[key]:+8.4f} pp"
    )


# ============================================================
# 3. WH NEIGHBORHOOD ROBUSTNESS
# ============================================================

qa = qc32["A2_control"]["all"]
qb = qc32["A3_neighbor_R1"]["all"]

assert (
    qa["eligible_GT"]
    == qb["eligible_GT"]
)

center_delta = (
    qb["center_iou"]
    - qa["center_iou"]
)

neighbor_delta = (
    qb["neighbor_iou"]
    - qa["neighbor_iou"]
)

print("\n" + "=" * 70)
print("A3 PILOT — WH ROBUSTNESS")
print("=" * 70)

print("Eligible GT:", qb["eligible_GT"])

print(
    "Center WH IoU:"
    f"\n  A2 = {qa['center_iou']:.6f}"
    f"\n  A3 = {qb['center_iou']:.6f}"
    f"\n  Delta = {center_delta:+.6f}"
)

print(
    "Neighbor WH IoU:"
    f"\n  A2 = {qa['neighbor_iou']:.6f}"
    f"\n  A3 = {qb['neighbor_iou']:.6f}"
    f"\n  Delta = {neighbor_delta:+.6f}"
)


# ============================================================
# 4. SAVE FINAL PILOT REPORT
# ============================================================

summary = {
    "experiment": "A3-R1 pilot warm-start from A2 160K",
    "additional_exposures_per_arm": 16000,
    "comparison": "A3_neighbor_R1 vs A2_control",

    "COCO_mAP_delta_pp": deltas["mAP_50_95"],

    "official_metric_deltas_pp": deltas,

    "eligible_gt_wh_qc": qb["eligible_GT"],

    "center_iou_delta": center_delta,

    "neighbor_iou_delta": neighbor_delta,

    "note": (
        "Pilot from A2 checkpoint; "
        "not a fresh 160K A3 experiment "
        "or multi-seed result."
    )
}

output_path = os.path.join(
    PILOT_ROOT,
    "A3_pilot_summary_verified.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        allow_nan=False
    )

print("\n" + "=" * 70)
print("STEP 33-A3 COMPLETE")
print("Saved:", output_path)
print("=" * 70)
