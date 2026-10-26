
# ============================================================
# STEP 33 — A3 QC SETUP & CHECKPOINT INTEGRITY
# Compatible with Steps 1–32-A3. No training.
# ============================================================
import os, json, gc, math, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

assert CONFIG['gaussian_radius_mode'] == 'v1_2_original'
assert CONFIG['wh_loss_type'] == 'hybrid_abs_relative'
assert CONFIG['a3_neighbor_radius'] == 1
assert abs(CONFIG['a3_neighbor_lambda'] - 0.25) < 1e-10
assert len(val_dataset) == 5000 and not val_dataset.augment
assert OUTPUT_WIDTH == OUTPUT_HEIGHT == 160
assert val_dataset.cat_id_to_contiguous[1] == 0

A3_QC_DIR = os.path.join(
    PILOT_ROOT, 'quality_checks_A3_pilot'
)
os.makedirs(A3_QC_DIR, exist_ok=True)

A3_ARMS = {
    'A2_control': os.path.join(
        PILOT_ROOT, 'A2_control', 'epoch2_add16k.pth'
    ),
    'A3_neighbor_R1': os.path.join(
        PILOT_ROOT, 'A3_neighbor_R1', 'epoch2_add16k.pth'
    ),
}

assert set(A3_ARMS) == {
    'A2_control', 'A3_neighbor_R1'
}

assert len(set(
    map(os.path.abspath, A3_ARMS.values())
)) == 2

A3_CK_META = {}

for arm, path in A3_ARMS.items():
    assert os.path.isfile(path), f'Missing: {path}'

    ck = torch.load(
        path, map_location='cpu', weights_only=False
    )

    assert ck['arm'] == arm
    assert ck['additional_exposures'] == 16000
    assert ck['gaussian_radius_mode'] == 'v1_2_original'
    assert ck['neighbor_lambda'] == (
        0.0 if arm == 'A2_control' else 0.25
    )
    assert ck['base_checkpoint'] == A2_CKPT

    A3_CK_META[arm] = {
        'checkpoint': path,
        'base_checkpoint': ck['base_checkpoint'],
        'source_manifest_sha256': ck['base_source_sha256'],
        'additional_exposures': ck['additional_exposures'],
        'neighbor_lambda': ck['neighbor_lambda'],
    }

    del ck

assert (
    A3_CK_META['A2_control']['source_manifest_sha256']
    == A3_CK_META['A3_neighbor_R1']['source_manifest_sha256']
)

# Reuse official COCOeval output from Step 31.
if 'results31' not in globals():
    with open(
        os.path.join(
            PILOT_ROOT, 'pilot_A2_vs_A3_COCOeval.json'
        )
    ) as f:
        results31 = json.load(f)

if 'qc32' not in globals():
    with open(
        os.path.join(
            PILOT_ROOT, 'pilot_A3_WH_neighborhood_QC.json'
        )
    ) as f:
        qc32 = json.load(f)

assert set(results31) == set(A3_ARMS)
assert set(qc32) == set(A3_ARMS)

A3_QC_IMAGE_COUNT = 1000

def a3_qc_loader():
    return DataLoader(
        Subset(
            val_dataset, range(A3_QC_IMAGE_COUNT)
        ),
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=centernet_collate_fn
    )

def a3_load_net(arm):
    ck = torch.load(
        A3_ARMS[arm],
        map_location='cpu',
        weights_only=False
    )

    net = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    net.load_state_dict(
        ck['model_state_dict'], strict=True
    )
    net.eval()

    del ck
    return net

def a3_wh_iou(p, t):
    p = np.maximum(
        np.asarray(p, dtype=np.float64), 0.
    )
    t = np.asarray(t, dtype=np.float64)

    intersection = np.minimum(p, t).prod()
    union = p.prod() + t.prod() - intersection

    return float(
        intersection / max(union, 1e-9)
    )

def a3_xyxy_iou(p, t):
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    inter = np.maximum(
        0.,
        np.minimum(p[2:], t[2:])
        - np.maximum(p[:2], t[:2])
    ).prod()

    a = np.maximum(
        0., p[2:] - p[:2]
    ).prod()

    b = np.maximum(
        0., t[2:] - t[:2]
    ).prod()

    return float(inter / max(a + b - inter, 1e-9))

def a3_pred_box(grid_xy, offset_xy, wh):
    c = (
        np.asarray(grid_xy, dtype=float)
        + np.asarray(offset_xy, dtype=float)
    ) * OUTPUT_STRIDE

    h = (
        np.maximum(
            np.asarray(wh, dtype=float), 0
        ) * OUTPUT_STRIDE / 2
    )

    return np.r_[c - h, c + h]

def a3_mean(rows, field):
    v = [
        r[field]
        for r in rows
        if r.get(field) is not None
        and np.isfinite(r[field])
    ]

    return float(np.mean(v)) if v else None

def a3_save_json(name, data):
    path = os.path.join(A3_QC_DIR, name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            data, f,
            indent=2,
            allow_nan=False
        )

    print('Saved:', path)
    return path

print(
    'STEP 33 PASS — checkpoints independent, '
    'pilot A2/A3 ready'
)

for arm, meta in A3_CK_META.items():
    print(arm, meta)
