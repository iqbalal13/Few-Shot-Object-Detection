
# ============================================================
# STEP 39 — QC6: WH BIAS AND FIXED V1.2 RADIUS GROUPS
# A3 uses Gaussian V1.2. No old B1 assertions.
# ============================================================

assert (
    CONFIG['gaussian_radius_mode']
    == 'v1_2_original'
)

summary39 = {}

for arm, rows in QC34_RECORDS.items():
    groups = {
        'all': rows,

        'small': [
            r for r in rows
            if r['group'] == 'small'
        ],

        'medium': [
            r for r in rows
            if r['group'] == 'medium'
        ],

        'large': [
            r for r in rows
            if r['group'] == 'large'
        ],

        'person': [
            r for r in rows
            if r['person']
        ],

        'radius_0_v12': [
            r for r in rows
            if r['radius_zero']
        ],

        'radius_positive_v12': [
            r for r in rows
            if not r['radius_zero']
        ],
    }

    assert (
        len(groups['radius_0_v12'])
        + len(groups['radius_positive_v12'])
        == len(rows)
    )

    summary39[arm] = {}

    for group, rr in groups.items():
        if not rr:
            continue

        sw = np.array([
            r['center_wh_bias_w']
            for r in rr
        ])

        sh = np.array([
            r['center_wh_bias_h']
            for r in rr
        ])

        summary39[arm][group] = {
            'gt': len(rr),

            'WH_center_mae_cells': a3_mean(
                rr, 'center_wh_mae_cells'
            ),

            'WH_size_iou_center': a3_mean(
                rr, 'center_iou'
            ),

            'WH_size_iou_neighbor_eligible': a3_mean(
                rr, 'neighbor_iou'
            ),

            'relative_bias_w_pct': (
                100 * float(np.mean(sw))
            ),

            'relative_bias_h_pct': (
                100 * float(np.mean(sh))
            ),

            'WH_w_underestimated_pct': (
                100 * float(np.mean(sw < 0))
            ),

            'WH_h_underestimated_pct': (
                100 * float(np.mean(sh < 0))
            ),

            'absolute_relative_error_w_pct': (
                100 * float(np.mean(np.abs(sw)))
            ),

            'absolute_relative_error_h_pct': (
                100 * float(np.mean(np.abs(sh)))
            ),
        }

for group in (
    'all',
    'small',
    'medium',
    'large',
    'person',
    'radius_0_v12',
    'radius_positive_v12'
):
    a = summary39['A2_control'][group]
    b = summary39['A3_neighbor_R1'][group]

    assert a['gt'] == b['gt'], (
        f'Mismatched group: {group}'
    )

    print(
        f'{group:<20} '
        f'N={a["gt"]:5d} '
        f'Δ WH_center_IoU='
        f'{b["WH_size_iou_center"]-a["WH_size_iou_center"]:+.5f} '
        f'Δ MAE_cell='
        f'{b["WH_center_mae_cells"]-a["WH_center_mae_cells"]:+.5f}'
    )

a3_save_json(
    'qc39_WH_bias_radius_v12.json',
    summary39
)

print(
    'STEP 39 PASS — same V1.2 population '
    'for both arms'
)
