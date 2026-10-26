
# ============================================================
# STEP 35 — A3 NEIGHBOR COVERAGE + GRADIENT ISOLATION
# No training. Checks lambda=0 identity and neighbor gradient.
# ============================================================
from collections import Counter

summary35 = {}

for arm, rows in QC34_RECORDS.items():
    assert len(rows) == QC34_COUNTS[arm]['gt']

    dist = Counter(
        r['neighbor_n'] for r in rows
    )

    group_info = {}

    for g in (
        'all', 'small', 'medium', 'large', 'person'
    ):
        rr = [
            r for r in rows
            if g == 'all'
            or (
                r['person'] if g == 'person'
                else r['group'] == g
            )
        ]

        n = len(rr)
        eligible = sum(
            r['neighbor_n'] > 0 for r in rr
        )

        group_info[g] = {
            'gt': n,
            'eligible_gt': eligible,
            'eligible_pct': (
                100 * eligible / n if n else None
            ),
            'mean_valid_neighbors': (
                float(np.mean([
                    r['neighbor_n'] for r in rr
                ]))
                if n else None
            )
        }

    summary35[arm] = {
        'neighbor_distribution': dict(
            sorted(dist.items())
        ),
        'groups': group_info
    }

    print(
        '\n', arm,
        'neighbor counts',
        dict(sorted(dist.items()))
    )

    for name, item in group_info.items():
        print(name, item)

assert (
    summary35['A2_control']['neighbor_distribution']
    ==
    summary35['A3_neighbor_R1']['neighbor_distribution']
)

# ------------------------------------------------------------
# Isolated gradient test.
# One unambiguous GT with 8 valid neighbors.
# ------------------------------------------------------------

probe = {
    'boxes': torch.tensor([
        [160., 160., 208., 208.]
    ]),
    'labels': torch.tensor([0])
}

t = build_centernet_target(
    probe,
    gaussian_radius_mode='v1_2_original'
)

assert t['neighbor_mask'][0].sum().item() == 8

keys = (
    'heatmap', 'wh', 'offset',
    'indices', 'mask',
    'neighbor_indices', 'neighbor_mask'
)

targets = {
    k: t[k].unsqueeze(0).to(device)
    for k in keys
}

ci = int(t['indices'][0])
cy, cx = divmod(ci, OUTPUT_WIDTH)

ni = int(
    t['neighbor_indices'][0][
        t['neighbor_mask'][0]
    ][0]
)

ny, nx = divmod(ni, OUTPUT_WIDTH)

old_lambda = criterion.neighbor_lambda
prev_training = criterion.training

criterion.eval()

try:
    gradients = {}
    center_losses = {}

    for label, lam in (
        ('A2', 0.0),
        ('A3', 0.25)
    ):
        criterion.neighbor_lambda = lam

        pred_wh = torch.ones(
            1, 2,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            device=device,
            requires_grad=True
        )

        preds = {
            'heatmap': torch.zeros(
                1, 80,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
                device=device,
                requires_grad=True
            ),
            'wh': pred_wh,
            'offset': torch.zeros(
                1, 2,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
                device=device,
                requires_grad=True
            )
        }

        loss = criterion(preds, targets)

        assert torch.isfinite(
            loss['loss_total']
        )

        loss['loss_total'].backward()

        gradients[label] = {
            'center': (
                pred_wh.grad[0, :, cy, cx]
                .detach().cpu().numpy().copy()
            ),
            'neighbor': (
                pred_wh.grad[0, :, ny, nx]
                .detach().cpu().numpy().copy()
            )
        }

        center_losses[label] = float(
            loss['loss_wh_center'].detach()
        )

    assert np.allclose(
        center_losses['A2'],
        center_losses['A3'],
        atol=1e-7
    )

    assert np.allclose(
        gradients['A2']['neighbor'],
        0,
        atol=1e-10
    )

    assert np.any(
        np.abs(
            gradients['A3']['neighbor']
        ) > 1e-9
    )

    assert np.allclose(
        gradients['A2']['center'],
        gradients['A3']['center'],
        atol=1e-7
    )

finally:
    criterion.neighbor_lambda = old_lambda
    criterion.train(prev_training)

summary35['gradient_sanity'] = {
    'lambda0_no_neighbor_gradient': True,
    'lambda025_neighbor_gradient': True,
    'center_loss_identical': True,
    'center_gradient_preserved': True
}

a3_save_json(
    'qc35_neighbor_coverage_and_gradient.json',
    summary35
)

print(
    'STEP 35 PASS — A3 assignment '
    'and gradient sanity'
)
