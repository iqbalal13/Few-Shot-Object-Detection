
# ============================================================
# STEP 37 — QC4 / FULL OFFICIAL COCOeval
# BASELINE, A1, A2, B1
# ALL 5000 COCO-VAL IMAGES / NO TRAINING
# ============================================================

import os
import json

assert callable(evaluate_official_checkpoint)

q4_paths = {
    "Baseline_V12_160K": QC_BASE160,
    "Experiment_A1_160K": QC_A1_160,

    "Experiment_A2_40K": os.path.join(
        QC_A2_DIR, "epoch5_40k.pth"
    ),
    "Experiment_A2_80K": os.path.join(
        QC_A2_DIR, "epoch10_80k.pth"
    ),
    "Experiment_A2_120K": os.path.join(
        QC_A2_DIR, "epoch15_120k.pth"
    ),
    "Experiment_A2_160K": QC_A2_160,
    "Experiment_A2_best": os.path.join(
        QC_A2_DIR, "centernet_coco80_best.pth"
    ),

    "Experiment_B1_160K": QC_B1_160
}

baseline200 = os.path.join(
    "/content/drive/MyDrive",
    "CenterNet_COCO80_V12_FRESH_200K_BS8",
    "epoch25_200k.pth"
)

if os.path.isfile(baseline200):
    q4_paths[
        "Historical_V12_200K_reference"
    ] = baseline200

for name, path in q4_paths.items():
    assert os.path.isfile(path), f"Missing {name}: {path}"

assert (
    os.path.abspath(q4_paths["Experiment_A2_160K"])
    != os.path.abspath(q4_paths["Experiment_B1_160K"])
)

assert (
    os.path.abspath(q4_paths["Experiment_A2_best"])
    != os.path.abspath(BEST_CHECKPOINT)
)

# Prevent a previous run's cached results from being reused.
assert "OFFICIAL_COCO_CACHE" in globals()
OFFICIAL_COCO_CACHE.clear()

q4 = {}

for name, path in q4_paths.items():

    print("\nEvaluating:", name)

    q4[name] = evaluate_official_checkpoint(
        path, name
    )

    value = q4[name]["metrics"]["mAP_50_95"]

    print(
        f"COCOeval {name}: "
        f"mAP={100 * value:.4f}%"
    )


# ============================================================
# METRIC TABLE
# ============================================================

metric_keys = (
    "mAP_50_95",
    "AP50",
    "AP75",
    "AP95",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large"
)

print("\nOFFICIAL COCO METRICS — PERCENT")

print(
    f'{"Metric":<19}'
    + "".join(
        f"{name:>23}"
        for name in q4
    )
)

for key in metric_keys:
    print(
        f"{key:<19}"
        + "".join(
            f'{100 * q4[name]["metrics"][key]:23.4f}'
            for name in q4
        )
    )


# ============================================================
# AP BY IOU THRESHOLD
# ============================================================

print("\nAP PER IoU THRESHOLD — PERCENT")

for threshold in range(50, 100, 5):
    key = f"{threshold / 100:.2f}"

    print(
        f"{key:<19}"
        + "".join(
            f'{100 * q4[name]["AP_by_IoU"][key]:23.4f}'
            for name in q4
        )
    )


# ============================================================
# PAIRWISE COMPARISONS
# ============================================================

baseline_metrics = q4[
    "Baseline_V12_160K"
]["metrics"]

a1_metrics = q4[
    "Experiment_A1_160K"
]["metrics"]

a2_metrics = q4[
    "Experiment_A2_160K"
]["metrics"]

b1_metrics = q4[
    "Experiment_B1_160K"
]["metrics"]

for title, left, right in (
    ("A1 - BASELINE", baseline_metrics, a1_metrics),
    ("A2 - BASELINE", baseline_metrics, a2_metrics),
    ("A2 - A1", a1_metrics, a2_metrics),
    ("B1 - A2", a2_metrics, b1_metrics)
):

    print(f"\n{title} (percentage points)")

    for key in metric_keys:
        difference = 100 * (
            right[key] - left[key]
        )

        print(
            f"{key:<19}: {difference:+.4f} pp"
        )


print(
    "\nBest A2 checkpoint epoch:",
    q4["Experiment_A2_best"]["epoch"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

result37 = {
    "models": q4,
    "primary_comparison": [
        "Baseline_V12_160K",
        "Experiment_A1_160K",
        "Experiment_A2_160K",
        "Experiment_B1_160K"
    ],
    "description": (
        "Official COCOeval bbox. "
        "Primary comparison uses fixed "
        "epoch-20 checkpoints."
    )
}

with open(
    os.path.join(
        QC_OUT,
        "qc4_coco_baseline_A1_A2_B1_160k.json"
    ),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        result37,
        file,
        indent=2,
        allow_nan=False
    )

print("STEP 37 COMPLETE — NO TRAINING")
print("Saved:", QC_OUT)
