
# ============================================================
# STEP 39 — WH CENTER / BIAS DIAGNOSTIC
# ============================================================

report39 = {}

for arm, rows in QC_RECORDS.items():

    report39[arm] = {}

    for group in GROUPS + (
        "radius_zero",
        "radius_positive"
    ):

        if group in GROUPS:
            rr = qc_group(rows, group)

        else:
            rr = [
                r for r in rows
                if (
                    (r["radius_v12"] == 0)
                    == (group == "radius_zero")
                )
            ]

        report39[arm][group] = {
            "GT": len(rr),

            "center_WH_iou": mean_field(
                rr, "center_wh_iou"
            ),

            "center_WH_mae": mean_field(
                rr, "center_wh_mae"
            ),

            "bias_width": mean_field(
                rr, "wh_bias_w"
            ),

            "bias_height": mean_field(
                rr, "wh_bias_h"
            ),

            "center_offset_mae": mean_field(
                rr, "center_offset_mae"
            ),

            "center_box_iou": mean_field(
                rr, "center_box_iou"
            )
        }


for group in report39[QC_REFERENCE]:

    a = report39[QC_REFERENCE][group]
    b = report39[QC_CANDIDATE][group]

    assert a["GT"] == b["GT"]

    x = a["center_WH_iou"]
    y = b["center_WH_iou"]

    delta = (
        y - x
        if x is not None and y is not None
        else None
    )

    print(
        group,
        "GT:", a["GT"],
        "WH IoU delta:", delta
    )


qc_save(
    "qc39_WH_center_bias.json",
    report39
)

print("STEP 39 PASS")
