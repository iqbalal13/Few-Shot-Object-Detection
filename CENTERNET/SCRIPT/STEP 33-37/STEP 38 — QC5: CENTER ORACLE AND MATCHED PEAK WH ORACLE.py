
# ============================================================
# STEP 38 — MATCHED PEAK WH / BOX IoU
# A3 REFERENCE VS A3-v2
# ============================================================

import numpy as np

report38 = {}

for group in GROUPS:

    a = {
        r["id"]: r
        for r in qc_group(
            QC_RECORDS[QC_REFERENCE],
            group
        )
    }

    b = {
        r["id"]: r
        for r in qc_group(
            QC_RECORDS[QC_CANDIDATE],
            group
        )
    }

    assert set(a) == set(b)

    common = [
        (a[k], b[k])
        for k in a
        if a[k]["matched"] and b[k]["matched"]
    ]

    report38[group] = {
        "common_GT": len(common),

        "delta_peak_WH_size_IoU": (
            float(np.mean([
                y["peak_wh_iou"]
                - x["peak_wh_iou"]
                for x, y in common
            ]))
            if common
            else None
        ),

        "delta_peak_box_IoU": (
            float(np.mean([
                y["peak_box_iou"]
                - x["peak_box_iou"]
                for x, y in common
            ]))
            if common
            else None
        )
    }

    print(group, report38[group])


qc_save(
    "qc38_matched_peak_box_WH.json",
    report38
)

print("STEP 38 PASS")
