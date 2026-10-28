
# ============================================================
# STEP 35 — A3-v2 TRAINING HISTORY AUDIT
# NO INFERENCE / NO TRAINING
# ============================================================

import os
import json

assert os.path.isfile(V2_HISTORY)

with open(
    V2_HISTORY,
    "r",
    encoding="utf-8"
) as f:
    v2_hist35 = json.load(f)

assert isinstance(v2_hist35, list)
assert len(v2_hist35) > 0

assert [
    row["epoch"]
    for row in v2_hist35
] == list(range(1, len(v2_hist35) + 1))

report35 = {
    "base_checkpoint": A3_CKPT,
    "selected_checkpoint": SELECTED_A3_V2,

    "completed_epochs": len(v2_hist35),

    "additional_exposures_completed": int(
        v2_hist35[-1]["additional_exposures"]
    ),

    "best_epoch": int(
        report33[QC_CANDIDATE]["epoch_recorded"]
    ),

    "best_candidate_map_pct": (
        100 * results31[
            "A3_v2_selected"
        ]["metrics"]["mAP_50_95"]
    ),

    "epochs": []
}

for row in v2_hist35:

    entry = {
        "epoch": int(row["epoch"]),

        "additional_exposures": int(
            row["additional_exposures"]
        ),

        "lr_backbone": float(
            row["lr_backbone"]
        ),

        "lr_neck_head": float(
            row["lr_neck_head"]
        ),

        "official_map_pct": (
            100 * float(row["official_map"])
            if row.get("official_map") is not None
            else None
        )
    }

    report35["epochs"].append(entry)

    print(entry)

assert (
    report35["best_epoch"]
    <= report35["completed_epochs"]
)

qc_save(
    "qc35_training_history.json",
    report35
)

print("STEP 35 PASS — training history inspected")
