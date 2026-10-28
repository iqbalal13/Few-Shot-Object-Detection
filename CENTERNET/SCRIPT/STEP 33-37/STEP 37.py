
# ============================================================
# STEP 37 — OFFICIAL COCO AP SUMMARY
# FROM STEP 31 / FULL 5000 COCO-VAL
# ============================================================

assert "A3_176K_reference" in results31
assert "A3_v2_selected" in results31

report37 = {
    "reference": "A3_176K_reference",
    "candidate": "A3_v2_selected",
    "delta_percentage_points": {}
}

for domain in (
    "metrics",
    "person_metrics"
):

    old = results31[
        "A3_176K_reference"
    ][domain]

    new = results31[
        "A3_v2_selected"
    ][domain]

    for key in old:

        a = old[key]
        b = new[key]

        if a is None or b is None:
            continue

        delta = 100 * (float(b) - float(a))

        report37[
            "delta_percentage_points"
        ][key] = delta

        print(
            f"{key:<20}: "
            f"A3={100*a:.4f}%  "
            f"A3-v2={100*b:.4f}%  "
            f"DELTA={delta:+.4f} pp"
        )


qc_save(
    "qc37_official_COCO_AP.json",
    report37
)

print("STEP 37 PASS")
