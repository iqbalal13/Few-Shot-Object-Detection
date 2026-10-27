
# STEP 38 — One-to-one matched peak WH/box IoU

report38 = {}

for g in GROUPS:
    a = {
        r["id"]: r
        for r in qc_group(
            QC_RECORDS["A2_control"], g
        )
    }

    b = {
        r["id"]: r
        for r in qc_group(
            QC_RECORDS[B2_ARM], g
        )
    }

    assert set(a) == set(b)

    common = [
        (a[k], b[k])
        for k in a
        if a[k]["matched"] and b[k]["matched"]
    ]

    report38[g] = {
        "common_GT": len(common),

        "delta_peak_WH_size_IoU": (
            float(np.mean([
                y["peak_wh_iou"] - x["peak_wh_iou"]
                for x, y in common
            ]))
            if common else None
        ),

        "delta_peak_box_IoU": (
            float(np.mean([
                y["peak_box_iou"] - x["peak_box_iou"]
                for x, y in common
            ]))
            if common else None
        )
    }

    print(g, report38[g])

qc_save(
    "qc38_matched_peak_box_WH.json",
    report38
)

print("STEP 38 PASS — paired matched-GT comparison")
