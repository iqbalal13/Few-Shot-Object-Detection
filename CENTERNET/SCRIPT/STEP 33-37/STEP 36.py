
# ============================================================
# STEP 36 — QC3: LOCALIZATION + TOP100 RECALL PROXIES
# Non-exclusive peak proximity, NOT COCO recall.
# ============================================================

summary36 = {}

for arm, rows in QC34_RECORDS.items():
    summary36[arm] = {}

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

        if not rr:
            continue

        summary36[arm][group] = {
            'gt': len(rr),

            'center_score_mean': a3_mean(
                rr, 'gt_center_score'
            ),

            'local_peak_within_r1_score_ge005_pct': (
                100 * np.mean([
                    r['local_peak_r1_ge005']
                    for r in rr
                ])
            ),

            'top100_class_correct_within_r1_ge005_pct': (
                100 * np.mean([
                    r['top100_r1_ge005']
                    for r in rr
                ])
            ),

            'top100_class_correct_within_r2_ge005_pct': (
                100 * np.mean([
                    r['top100_r2_ge005']
                    for r in rr
                ])
            ),

            'top100_class_correct_within_r4_ge005_pct': (
                100 * np.mean([
                    r['top100_r4_ge005']
                    for r in rr
                ])
            ),
        }

for group in (
    'all', 'small', 'medium', 'large', 'person'
):
    a = summary36['A2_control'][group]
    b = summary36['A3_neighbor_R1'][group]

    assert a['gt'] == b['gt']

    print('\n', group, 'N=', a['gt'])

    for field in (
        'local_peak_within_r1_score_ge005_pct',
        'top100_class_correct_within_r1_ge005_pct',
        'top100_class_correct_within_r2_ge005_pct',
        'top100_class_correct_within_r4_ge005_pct'
    ):
        print(
            field,
            'A2', round(a[field], 3),
            'A3', round(b[field], 3),
            'delta_pp',
            round(b[field] - a[field], 3)
        )

a3_save_json(
    'qc36_heatmap_top100_proximity.json',
    summary36
)

print(
    'STEP 36 PASS — proximity is a diagnostic, '
    'not detection recall'
)
