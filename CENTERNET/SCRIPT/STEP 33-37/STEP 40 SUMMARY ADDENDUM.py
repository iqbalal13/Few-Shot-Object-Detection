
# ============================================================
# STEP 40 — QC7: WH SPATIAL READOUT STABILITY
# Eight directions at R1/R2/R4, fixed GT subset.
# ============================================================

summary40 = {}

for arm, rows in QC34_RECORDS.items():
    summary40[arm] = {}

    for group in (
        'all', 'small', 'medium', 'large', 'person'
    ):
        rr = [
            r for r in rows
            if group == 'all'
            or (
                r['person'] if group == 'person'
                else r['group'] == group
            )
        ]

        summary40[arm][group] = {
            'gt': len(rr),

            'center_iou_all_gt': a3_mean(
                rr, 'center_iou'
            )
        }

        for rad in (1, 2, 4):
            for location in ('all', 'inside'):
                f = f'r{rad}_{location}'

                subset = [
                    r for r in rr
                    if r[f] is not None
                ]

                summary40[arm][group][
                    f'{f}_n'
                ] = len(subset)

                summary40[arm][group][
                    f'{f}_mean_iou'
                ] = a3_mean(subset, f)

                summary40[arm][group][
                    f'{f}_center_mean_same_gt'
                ] = a3_mean(
                    subset, 'center_iou'
                )

                summary40[arm][group][
                    f'{f}_gap_from_center'
                ] = (
                    float(np.mean([
                        r[f] - r['center_iou']
                        for r in subset
                    ]))
                    if subset else None
                )

        eligible = [
            r for r in rr
            if r['neighbor_n'] > 0
        ]

        summary40[arm][group][
            'eligible_GT'
        ] = len(eligible)

        summary40[arm][group][
            'eligible_center_iou'
        ] = a3_mean(
            eligible, 'center_iou'
        )

        summary40[arm][group][
            'eligible_neighbor_iou'
        ] = a3_mean(
            eligible, 'neighbor_iou'
        )

for group in (
    'all', 'small', 'medium', 'large', 'person'
):
    a = summary40['A2_control'][group]
    b = summary40['A3_neighbor_R1'][group]

    assert a['gt'] == b['gt']

    print(
        '\n', group,
        'GT=', a['gt']
    )

    print(
        'center Δ',
        round(
            b['center_iou_all_gt']
            - a['center_iou_all_gt'],
            6
        )
    )

    for rad in (1, 2, 4):
        for where in ('all', 'inside'):
            prefix = f'r{rad}_{where}'

            ak = a[f'{prefix}_mean_iou']
            bk = b[f'{prefix}_mean_iou']

            if ak is None or bk is None:
                continue

            assert (
                a[f'{prefix}_n']
                == b[f'{prefix}_n']
            )

            print(
                prefix,
                'N', a[f'{prefix}_n'],
                'A2', round(ak, 5),
                'A3', round(bk, 5),
                'delta', round(bk - ak, 6)
            )

    print(
        'A3 neighbor-eligible gain',
        round(
            b['eligible_neighbor_iou']
            - a['eligible_neighbor_iou'],
            6
        )
    )

a3_save_json(
    'qc40_spatial_robustness_r1_r2_r4.json',
    summary40
)

print(
    'STEP 40 PASS — directions are diagnostic only, '
    'not actual detector matching'
)
