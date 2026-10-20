# STEP 40 SUMMARY ADDENDUM — NO INFERENCE / NO TRAINING

for model_name, report in results40.items():

    print("\n" + "=" * 65)
    print("MODEL:", model_name)

    for group in (
        "all",
        "small",
        "medium",
        "large",
        "person",
        "isolated",
        "small_isolated",
        "person_isolated"
    ):

        s = report["groups"].get(group)

        if s is None:
            continue

        print(f"\nGROUP: {group} | GT={s['gt_count']}")

        for radius in (1, 2, 4):

            r = s[f"ring_{radius}"]

            inside = r["inside_bbox_only_iou"]
            overall = r["size_iou"]

            print(
                f"R{radius}: "
                f"all={overall['mean']:.4f} | "
                f"inside_bbox="
                f"{inside['mean'] if inside['mean'] is not None else 'N/A'} | "
                f"inside_direction_pct="
                f"{r['directions_inside_bbox_pct']:.2f}%"
            )

        peak = s["peak_matching"]

        print(
            "Peak matched:",
            f"{peak['matched_within_8_pct']:.2f}%"
        )

        print(
            "Peak WH delta IoU:",
            peak[
                "delta_size_iou_peak_minus_center_paired"
            ]["mean"]
        )

print("\nADDITIONAL SUMMARY COMPLETE — NO TRAINING")
