
# STEP 41 — B2 pilot decision summary
# No training, no new inference.

for name in (
    "report35",
    "report36",
    "report37",
    "report38",
    "report39",
    "report40"
):
    assert name in globals(), name

AP = report37["delta_percentage_points"]

summary41 = {
    "experiment": (
        "B2 C3/C4 zero-initialized "
        "gated residual fusion"
    ),

    "control": (
        "A2 original 160K + "
        "16K continuation"
    ),

    "B2": (
        "A2 original 160K + "
        "16K B2 continuation"
    ),

    "official_COCO_AP_delta_pp": AP,
    "gates": report35,

    "one_to_one_peak_R1_delta_pp": (
        report36[B2_ARM]["all"][
            "top100_one_to_one_R1_ge005_pct"
        ]
        - report36["A2_control"]["all"][
            "top100_one_to_one_R1_ge005_pct"
        ]
    ),

    "matched_peak_common_GT": (
        report38["all"]["common_GT"]
    ),

    "matched_peak_WH_IoU_delta": (
        report38["all"]["delta_peak_WH_size_IoU"]
    ),

    "matched_peak_box_IoU_delta": (
        report38["all"]["delta_peak_box_IoU"]
    ),

    "GT_in_1000_image_QC": (
        report39["A2_control"]["all"]["GT"]
    ),

    "interpretation": (
        "One-seed pilot; do not infer "
        "statistical significance or "
        "CCTV FSOD success"
    )
}

qc_save(
    "qc41_final_B2_pilot_summary.json",
    summary41
)

print(
    "B2 mAP delta pp:",
    round(AP["mAP_50_95"], 4)
)

print(
    "B2 AP75 delta pp:",
    round(AP["AP75"], 4)
)

print(
    "B2 person AP delta pp:",
    round(AP["person_AP_50_95"], 4)
)

print(
    "B2 matched peak WH delta:",
    summary41["matched_peak_WH_IoU_delta"]
)

print("STEP 41 PASS — no new training")
