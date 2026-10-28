
# ============================================================
# STEP 32 — A3-v2 FINAL MODEL CARD
# ============================================================

import os
import json

print("=" * 70)
print("STEP 32 — A3-v2 FINAL MODEL SUMMARY")
print("=" * 70)

assert "results31" in globals(), (
    "Run STEP 31 before STEP 32."
)

base = results31[
    "A3_176K_reference"
]

final = results31.get(
    "A3_v2_selected",
    base
)

source_is_selected = (
    "A3_v2_selected" not in results31
)

gain_map = 100 * (
    final["metrics"]["mAP_50_95"]
    - base["metrics"]["mAP_50_95"]
)

report = {
    "method": (
        "ResNet101 CenterNet "
        "A2-hybrid + A3-neighborhood R1"
    ),

    "change": (
        "Train augmentation + "
        "50/50 uniform/class-balanced sampling "
        "+ longer COCO continuation"
    ),

    "source_model": "A3 176K",

    "reference_checkpoint": A3_CKPT,

    "selected_checkpoint": SELECTED_A3_V2,

    "selected_is_reference": source_is_selected,

    "reference_metrics": base["metrics"],

    "selected_metrics": final["metrics"],

    "reference_person": base["person_metrics"],

    "selected_person": final["person_metrics"],

    "gain_mAP_percentage_points": gain_map,

    "train_history": V2_HISTORY,

    "limit": (
        "Results are COCO source-stage, "
        "not few-shot CCTV metrics."
    )
}

output_path = os.path.join(
    V2_ROOT,
    "A3_v2_model_card.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        allow_nan=False
    )

print(
    "Chosen checkpoint:",
    SELECTED_A3_V2
)

print(
    "Reference A3 mAP50:95:",
    100 * base["metrics"]["mAP_50_95"],
    "%"
)

print(
    "Selected mAP50:95:",
    100 * final["metrics"]["mAP_50_95"],
    "%"
)

print(
    "Gain over original A3:",
    gain_map,
    "percentage points"
)

print("Saved model card:", output_path)
print("STEP 32 COMPLETE")
