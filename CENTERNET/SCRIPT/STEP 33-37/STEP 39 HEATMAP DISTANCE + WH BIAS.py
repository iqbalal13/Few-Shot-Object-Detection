
# STEP 39 — WH bias by size and fixed V1.2 radius

report39 = {}

for arm, rows in QC_RECORDS.items():
    report39[arm] = {}

    for g in GROUPS + (
        "radius_zero",
        "radius_positive"
    ):
        rr = (
            qc_group(rows, g)
            if g in GROUPS
            else [
                r for r in rows
                if (
                    (r["radius_v12"] == 0)
                    == (g == "radius_zero")
                )
            ]
        )

        report39[arm][g] = {
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
            )
        }

for g in report39["A2_control"]:
    a = report39["A2_control"][g]
    b = report39[B2_ARM][g]

    assert a["GT"] == b["GT"]

    print(
        g,
        "GT", a["GT"],
        "Δ center WH IoU",
        b["center_WH_iou"] - a["center_WH_iou"]
    )

qc_save(
    "qc39_WH_bias_fixed_V12.json",
    report39
)

print("STEP 39 PASS")
