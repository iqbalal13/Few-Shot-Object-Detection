
# STEP 36 — One-to-one class-correct top100 coverage
import numpy as np

GROUPS = (
    "all",
    "small",
    "medium",
    "large",
    "person"
)


def qc_group(rows, g):
    return [
        r for r in rows
        if g == "all"
        or (
            r["person"] if g == "person"
            else r["group"] == g
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
        if values else None
    )


report36 = {}

for arm, rows in QC_RECORDS.items():
    report36[arm] = {}

    for g in GROUPS:
        rr = qc_group(rows, g)

        report36[arm][g] = {
            "GT": len(rr),

            "top100_one_to_one_R1_ge005_pct": (
                100 * mean_field(rr, "peak_r1")
            ),

            "top100_one_to_one_R2_ge005_pct": (
                100 * mean_field(rr, "peak_r2")
            ),

            "top100_one_to_one_R4_ge005_pct": (
                100 * mean_field(rr, "peak_r4")
            )
        }

for g in GROUPS:
    a = report36["A2_control"][g]
    b = report36[B2_ARM][g]

    assert a["GT"] == b["GT"]

    print(
        g,
        "GT", a["GT"],
        "R1 Δpp",
        b["top100_one_to_one_R1_ge005_pct"]
        - a["top100_one_to_one_R1_ge005_pct"],
        "R2 Δpp",
        b["top100_one_to_one_R2_ge005_pct"]
        - a["top100_one_to_one_R2_ge005_pct"]
    )

qc_save(
    "qc36_heatmap_peak_coverage.json",
    report36
)

print("STEP 36 PASS — diagnostic, not official recall")
