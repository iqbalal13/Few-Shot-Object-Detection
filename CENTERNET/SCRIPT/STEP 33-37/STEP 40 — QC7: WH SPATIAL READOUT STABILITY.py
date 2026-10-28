
# ============================================================
# STEP 40 — WH SPATIAL ROBUSTNESS
# A3 VS A3-v2
# ============================================================

report40 = {}

for arm, rows in QC_RECORDS.items():

    report40[arm] = {}

    for group in GROUPS:

        rr = qc_group(rows, group)

        item = {
            "GT": len(rr),

            "center": mean_field(
                rr, "center_wh_iou"
            )
        }

        for radius in (1, 2, 4):

            for zone in ("all", "inside"):

                key = f"R{radius}_{zone}"

                item[key] = mean_field(
                    rr,
                    f"r{radius}_{zone}"
                )

        report40[arm][group] = item


for group in GROUPS:

    a = report40[QC_REFERENCE][group]
    b = report40[QC_CANDIDATE][group]

    assert a["GT"] == b["GT"]

    print("\nGROUP:", group)

    keys = [
        "center",
        "R1_all",
        "R1_inside",
        "R2_all",
        "R2_inside",
        "R4_all",
        "R4_inside"
    ]

    for key in keys:

        x = a[key]
        y = b[key]

        delta = (
            y - x
            if x is not None and y is not None
            else None
        )

        print(
            key,
            "A3:", x,
            "A3-v2:", y,
            "delta:", delta
        )


qc_save(
    "qc40_WH_spatial_robustness.json",
    report40
)

print("STEP 40 PASS")
