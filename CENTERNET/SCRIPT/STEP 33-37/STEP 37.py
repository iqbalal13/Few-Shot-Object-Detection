
# STEP 37 — Official COCO AP summary from Step 31

assert set(results31) == {
    "A2_control", B2_ARM
}

report37 = {
    "control": "A2_control",
    "experiment": B2_ARM,
    "delta_percentage_points": {}
}

for domain in ("metrics", "person_metrics"):
    for k in results31["A2_control"][domain]:
        a = results31["A2_control"][domain][k]
        b = results31[B2_ARM][domain][k]

        if a is None or b is None:
            continue

        report37["delta_percentage_points"][k] = (
            100 * (b - a)
        )

        print(
            f"{k:<20}: "
            f"A2={100*a:.4f}% "
            f"B2={100*b:.4f}% "
            f"Δ={100*(b-a):+.4f} pp"
        )

qc_save(
    "qc37_official_COCO_AP.json",
    report37
)

print("STEP 37 PASS — 5000-image official COCOeval")
