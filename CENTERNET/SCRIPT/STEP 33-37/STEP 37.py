# ============================================================
# STEP 37 / QUALITY CHECK 4
# OFFICIAL COCOeval CHECKPOINT COMPARISON
# NO TRAINING / NO F1
# ============================================================

import os
import json

q4_paths = {
    "V1.2_80K": BASE_BEST30,
    "V1.2_120K": MILESTONE15_30,
    "V1.2_160K": MILESTONE20_30,
}

q4_v13 = os.path.join(
    ROOT30,
    "CenterNet_COCO80_Source_BALANCED80K_V1_3_AUG_GAUSS_BS8",
    "centernet_coco80_best.pth"
)

if os.path.isfile(q4_v13):
    q4_paths["V1.3_80K"] = q4_v13

q4_results = {}

for name, path in q4_paths.items():

    if not os.path.isfile(path):
        print("NOT AVAILABLE:", name)
        continue

    print("COCOeval:", name)

    q4_results[name] = evaluate_official_checkpoint(
        path, name
    )

assert "V1.2_80K" in q4_results
assert "V1.2_160K" in q4_results


# ============================================================
# 1. COMPLETE AP / AR TABLE
# ============================================================

metric_names = [
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
]

print("\nOFFICIAL COCOeval ALL 80 CLASSES — %")

print(
    f'{"metric":<16}'
    + "".join(
        f"{name:>17}"
        for name in q4_results
    )
)

for m in metric_names:
    print(
        f"{m:<16}"
        + "".join(
            f'{100*r["metrics"][m]:17.4f}'
            for r in q4_results.values()
        )
    )


# ============================================================
# 2. AP FOR EACH IoU THRESHOLD
# ============================================================

print("\nAP PER IoU — %")

for threshold in [
    f"{x/100:.2f}"
    for x in range(50, 100, 5)
]:
    print(
        f"{threshold:<16}"
        + "".join(
            f'{100*r["AP_by_IoU"][threshold]:17.4f}'
            for r in q4_results.values()
        )
    )


# ============================================================
# 3. SELECT HIGHEST VALIDATION mAP
# ============================================================

winner = max(
    q4_results,
    key=lambda n: (
        q4_results[n]["metrics"]["mAP_50_95"]
    )
)

print("\nBEST VALIDATION mAP:", winner)

print(
    "mAP:",
    100 * q4_results[winner][
        "metrics"
    ]["mAP_50_95"],
    "%"
)


# ============================================================
# 4. SAVE RESULTS
# ============================================================

save_dir = os.path.join(
    ROOT30,
    "CenterNet_Quality_Diagnostics"
)

os.makedirs(save_dir, exist_ok=True)

with open(
    os.path.join(
        save_dir,
        "qc4_all_AP_AR.json"
    ),
    "w"
) as f:
    json.dump(
        {
            "winner_by_map": winner,
            "models": q4_results
        },
        f,
        indent=2,
        allow_nan=False
    )

print("QC4 COMPLETE — NO TRAINING")
