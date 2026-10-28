
# ============================================================
# STEP 36 — ONE-TO-ONE TOP100 PEAK COVERAGE
# A3 vs A3-v2
# ============================================================

import numpy as np

GROUPS = (
    "all",
    "small",
    "medium",
    "large",
    "person"
)


def qc_group(rows, group):
    return [
        r for r in rows
        if group == "all"
        or (
            r["person"]
            if group == "person"
            else r["group"] == group
        )
    ]


def mean_field(rows, key):
    values = [
        r[key]
        for r in rows
        if r.get(key) is not None
    ]

    return (
        float(np.mean(values))
        if values
        else None
    )


report36 = {}

for arm, rows in QC_RECORDS.items():

    report36[arm] = {}

    for group in GROUPS:

        rr = qc_group(rows, group)

        report36[arm][group] = {
            "GT": len(rr),

            "top100_one_to_one_R1_ge005_pct": (
                100 * (mean_field(rr, "peak_r1") or 0.)
            ),

            "top100_one_to_one_R2_ge005_pct": (
                100 * (mean_field(rr, "peak_r2") or 0.)
            ),

            "top100_one_to_one_R4_ge005_pct": (
                100 * (mean_field(rr, "peak_r4") or 0.)
            )
        }


for group in GROUPS:

    a = report36[QC_REFERENCE][group]
    b = report36[QC_CANDIDATE][group]

    assert a["GT"] == b["GT"]

    print(
        group,
        "GT:", a["GT"],

        "R1 delta pp:",
        b["top100_one_to_one_R1_ge005_pct"]
        - a["top100_one_to_one_R1_ge005_pct"],

        "R2 delta pp:",
        b["top100_one_to_one_R2_ge005_pct"]
        - a["top100_one_to_one_R2_ge005_pct"]
    )


qc_save(
    "qc36_heatmap_peak_coverage.json",
    report36
)

print("STEP 36 PASS")
