
# ============================================================
# STEP 41 — A3 PILOT EVIDENCE SUMMARY / GO-NO-GO
# No inference, no training. All metrics from Steps 31–40.
# ============================================================

for key in (
    'summary35',
    'summary36',
    'summary37',
    'summary38',
    'summary39',
    'summary40'
):
    assert key in globals(), (
        f'Run prior QC step first: {key}'
    )

AP = summary37[
    'delta_A3_minus_A2_percentage_points'
]

robustness = (
    summary40['A3_neighbor_R1']['all'][
        'eligible_neighbor_iou'
    ]
    -
    summary40['A2_control']['all'][
        'eligible_neighbor_iou'
    ]
)

center_delta = (
    summary40['A3_neighbor_R1']['all'][
        'eligible_center_iou'
    ]
    -
    summary40['A2_control']['all'][
        'eligible_center_iou'
    ]
)

peak_shared = summary38[
    'common_gt_paired_at_peak'
]['all']

final41 = {
    'experiment': (
        'A3-R1 pilot warm-start, '
        'matched A2-continuation control'
    ),

    'training': {
        'base_checkpoint': A2_CKPT,
        'base_exposures': 160000,
        'additional_exposures_per_arm': 16000,
        'total_exposures_per_arm': 176000,
        'lambda_control': 0.0,
        'lambda_A3': 0.25,
        'gaussian_mode': 'v1_2_original',
        'WH_loss': (
            'hybrid_absolute_0.75_relative_0.25'
        )
    },

    'official_AP_delta_percentage_points': AP,

    'neighbor_WH_size_iou_delta': robustness,
    'eligible_center_WH_size_iou_delta': center_delta,

    'shared_matched_top100_GT_count': peak_shared['n'],

    'shared_matched_peak_WH_size_iou_delta': (
        peak_shared['delta_matched_WH_size_iou']
    ),

    'shared_matched_peak_box_iou_delta': (
        peak_shared['delta_matched_box_iou']
    ),

    'quality_checks': {
        'GT_total': (
            QC34_COUNTS['A2_control']['gt']
        ),

        'GT_with_valid_neighbors': (
            QC34_COUNTS['A2_control'][
                'eligible_neighbors_gt'
            ]
        ),

        'lambda_gradient_sanity_passed': (
            summary35['gradient_sanity']
        ),
    },

    'limitations': [
        'One seed and one paired pilot; no uncertainty interval',
        'Matched-peak diagnostic is non-exclusive; not COCOeval matching',
        'QC uses 1000 val images, official AP uses 5000',
        'Warm-start from A2; not a fresh 160K A3 training',
        'Results are source-stage COCO, not CCTV few-shot adaptation',
    ]
}

a3_save_json(
    'qc41_A3_pilot_final_summary.json',
    final41
)

print('=' * 74)
print('A3 PILOT FINAL SUMMARY')

print(
    'mAP Δ (percentage points):',
    round(AP['mAP_50_95'], 5)
)

print(
    'AP75 Δ (percentage points):',
    round(AP['AP75'], 5)
)

print(
    'AP small/medium/large Δ:',
    *[
        round(AP[k], 5)
        for k in (
            'AP_small',
            'AP_medium',
            'AP_large'
        )
    ]
)

print(
    'Person AP Δ:',
    round(
        AP['person_AP_50_95'],
        5
    )
)

print(
    'WH size IoU Δ, eligible neighbor:',
    round(robustness, 6)
)

print(
    'WH size IoU Δ, eligible GT center:',
    round(center_delta, 6)
)

print(
    'Matched top100 common-GT WH IoU Δ:',
    peak_shared[
        'delta_matched_WH_size_iou'
    ]
)

print(
    'Interpretation:',
    (
        'positive pilot, needs replication'
        if AP['mAP_50_95'] > 0
        and robustness > 0
        else 'not clearly beneficial on paired pilot'
    )
)

print(
    'Do not claim large improvement '
    'or FSOD CCTV success.'
)

print(
    'STEP 41 COMPLETE — NO TRAINING'
)
