
# ============================================================
# STEP 38 — QC5: CENTER ORACLE AND MATCHED PEAK WH ORACLE
# Top100 nearest within R4 is NON-EXCLUSIVE diagnostic.
# ============================================================

summary38 = {}
common_by_group = {}

for group in (
    'all', 'small', 'medium', 'large', 'person'
):
    per_arm = {}

    for arm, rows in QC34_RECORDS.items():
        rr = [
            r for r in rows
            if group == 'all'
            or (
                r['person'] if group == 'person'
                else r['group'] == group
            )
        ]

        matched = [
            r for r in rr
            if r['matched_peak_box_iou'] is not None
        ]

        per_arm[arm] = {
            'GT': len(rr),
            'matched_top100_within_r4_n': len(matched),

            'center_box_iou': a3_mean(
                rr, 'center_box_iou'
            ),

            'center_perfect_WH_box_iou': a3_mean(
                rr, 'center_wh_oracle_box_iou'
            ),

            'center_perfect_offset_box_iou': a3_mean(
                rr, 'center_offset_oracle_box_iou'
            ),

            'center_perfect_WH_gain': (
                float(np.mean([
                    r['center_wh_oracle_box_iou']
                    - r['center_box_iou']
                    for r in rr
                ]))
                if rr else None
            ),

            'center_perfect_offset_gain': (
                float(np.mean([
                    r['center_offset_oracle_box_iou']
                    - r['center_box_iou']
                    for r in rr
                ]))
                if rr else None
            ),

            'matched_peak_box_iou': a3_mean(
                matched, 'matched_peak_box_iou'
            ),

            'matched_peak_WH_size_iou': a3_mean(
                matched, 'matched_peak_wh_iou'
            ),

            'matched_peak_perfect_WH_box_gain': (
                float(np.mean([
                    r['matched_peak_perfect_wh_box_iou']
                    - r['matched_peak_box_iou']
                    for r in matched
                ]))
                if matched else None
            ),
        }

    summary38[group] = per_arm

    # Strict paired subset:
    # Both arms must find a candidate for the SAME GT.
    aa = {
        r['id']: r
        for r in QC34_RECORDS['A2_control']
    }

    bb = {
        r['id']: r
        for r in QC34_RECORDS['A3_neighbor_R1']
    }

    common = [
        (aa[i], bb[i])
        for i in aa
        if (
            aa[i]['matched_peak_wh_iou'] is not None
            and
            bb[i]['matched_peak_wh_iou'] is not None
            and
            (
                group == 'all'
                or (
                    aa[i]['person']
                    if group == 'person'
                    else aa[i]['group'] == group
                )
            )
        )
    ]

    common_by_group[group] = {
        'n': len(common),

        'delta_matched_WH_size_iou': (
            float(np.mean([
                b['matched_peak_wh_iou']
                - a['matched_peak_wh_iou']
                for a, b in common
            ]))
            if common else None
        ),

        'delta_matched_box_iou': (
            float(np.mean([
                b['matched_peak_box_iou']
                - a['matched_peak_box_iou']
                for a, b in common
            ]))
            if common else None
        ),
    }

    print(
        group,
        'center A2/A3:',
        round(
            per_arm['A2_control']['center_box_iou'], 5
        ),
        round(
            per_arm['A3_neighbor_R1']['center_box_iou'], 5
        ),
        'paired matched',
        common_by_group[group]
    )

summary38[
    'common_gt_paired_at_peak'
] = common_by_group

summary38['method_note'] = (
    'Matched peak is nearest class-correct top100 '
    'score>=0.05 within Chebyshev R4; '
    'may be reused by multiple GT. '
    'Not official TP matching.'
)

a3_save_json(
    'qc38_wh_oracle_matched_peaks.json',
    summary38
)

print(
    'STEP 38 PASS — center and '
    'detected-peak WH oracle'
)
