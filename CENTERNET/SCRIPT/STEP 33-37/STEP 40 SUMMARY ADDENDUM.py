
# STEP 40 — WH robustness around GT center

report40 = {}

for arm, rows in QC_RECORDS.items():
    report40[arm] = {}

    for g in GROUPS:
        rr = qc_group(rows, g)

        item = {
            "GT": len(rr),
            "center": mean_field(
                rr, "center_wh_iou"
            )
        }

        for r in (1, 2, 4):
            for zone in ("all", "inside"):
                item[f"R{r}_{zone}"] = mean_field(
                    rr,
                    f"r{r}_{zone}"
                )

        report40[arm][g] = item

for g in GROUPS:
    a = report40["A2_control"][g]
    b = report40[B2_ARM][g]

    assert a["GT"] == b["GT"]

    print(
        g,
        "Δ center",
        b["center"] - a["center"]
    )

    for r in (1, 2, 4):
        key = f"R{r}_all"

        print(
            " ", key,
            "Δ", b[key] - a[key]
        )

qc_save(
    "qc40_WH_spatial_robustness.json",
    report40
)

print("STEP 40 PASS")
