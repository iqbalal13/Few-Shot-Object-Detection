
# ============================================================
# STEP 37 — QC4: OFFICIAL COCOeval FINAL PAIR COMPARISON
# Reuses Step 31 official 5000-image evaluation.
# ============================================================

metric_names = (
    'mAP_50_95',
    'AP50',
    'AP75',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR1',
    'AR10',
    'AR100',
    'AR_small',
    'AR_medium',
    'AR_large'
)

person_names = (
    'person_AP_50_95',
    'person_AP50',
    'person_AP75'
)

summary37 = {
    'protocol': (
        'COCO bbox official 5000-image, K100, '
        'A2 and A3 each +16K from same A2 160K checkpoint'
    ),
    'models': {},
    'delta_A3_minus_A2_percentage_points': {}
}

for arm in A3_ARMS:
    item = results31[arm]

    assert int(item['images']) == 5000

    assert (
        os.path.abspath(item['checkpoint'])
        ==
        os.path.abspath(A3_ARMS[arm])
    )

    summary37['models'][arm] = {
        'images': item['images'],
        'checkpoint': item['checkpoint'],
        'metrics': item['metrics'],
        'person_metrics': item['person_metrics'],
    }

print(
    'OFFICIAL COCOeval — both arms '
    'after +16K exposures:'
)

for key in metric_names + person_names:
    group = (
        'person_metrics'
        if key in person_names
        else 'metrics'
    )

    a = results31[
        'A2_control'
    ][group][key]

    b = results31[
        'A3_neighbor_R1'
    ][group][key]

    assert a is not None and b is not None

    delta = 100 * (b - a)

    summary37[
        'delta_A3_minus_A2_percentage_points'
    ][key] = delta

    print(
        f'{key:<24} '
        f'A2={100*a:8.4f}% '
        f'A3={100*b:8.4f}% '
        f'Δ={delta:+8.4f} pp'
    )

summary37['historical_note'] = (
    'Historical A2 160K ~14.3732% and '
    'B1 160K ~13.7688% are separate runs. '
    'Do not use them as same-exposure '
    'controls for A3 +16K.'
)

a3_save_json(
    'qc37_official_cocoeval_A2_vs_A3.json',
    summary37
)

print(
    'STEP 37 PASS — authoritative '
    'end-to-end metrics'
)
