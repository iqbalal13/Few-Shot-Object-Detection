
# ================================================================
# STEP 33 — OFFICIAL COCO BBOX EVALUATION (ALL 80 CLASSES)
# Balanced80K V1.3 — 5000 val images, zero extra training
# AP50 / AP75 / AP95 / AP@0.50:0.95 / AR + saved JSON
# ================================================================

import os
import json
import math
import time
import contextlib
import io
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

print('=' * 72)
print('STEP 33 — OFFICIAL COCO BBOX mAP / IoU — ALL 80 CLASSES')
print('=' * 72)

# ================================================================
# 1. EXPERIMENT AND CHECKPOINT IDENTITY
# ================================================================

assert CONFIG['experiment_name'] == (
    'CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8'
)

assert CONFIG['num_classes'] == 80
assert len(val_dataset) == 5000

assert not val_dataset.augment, (
    'Validation must have NO random augmentation.'
)

assert os.path.isfile(BEST_CHECKPOINT), (
    'Step 30 best checkpoint missing.'
)

ckpt33 = torch.load(
    BEST_CHECKPOINT,
    map_location='cpu',
    weights_only=False
)

assert ckpt33['experiment_name'] == CONFIG['experiment_name']
assert ckpt33['source_images_total'] == 80000
assert 1 <= int(ckpt33['epoch']) <= 10

if 'source_manifest_sha256' in globals():
    assert (
        ckpt33['source_manifest_sha256']
        == source_manifest_sha256
    )

stored_stats33 = ckpt33.get('val_stats', {})

print('Best checkpoint epoch:', int(ckpt33['epoch']))
print('Stored official mAP  :', stored_stats33.get('official_map'))
print('Stored official AP50 :', stored_stats33.get('official_ap50'))
print('Stored official AP75 :', stored_stats33.get('official_ap75'))

# ================================================================
# 2. RELOAD BEST CENTERNET MODEL
# ================================================================

model33 = CenterNetResNet101(
    num_classes=CONFIG['num_classes'],
    pretrained_backbone=False
).to(device)

model33.load_state_dict(
    ckpt33['model_state_dict'],
    strict=True
)

model33.eval()

ckpt33_epoch = int(ckpt33['epoch'])

del ckpt33

# ================================================================
# 3. FULL COCO VALIDATION CONFIGURATION
# ================================================================

img_ids33 = set(
    map(int, val_dataset.image_ids)
)

cat_ids33 = list(
    map(int, val_dataset.cat_ids)
)

assert len(img_ids33) == 5000
assert len(cat_ids33) == 80

COCO33_TOPK = 100
COCO33_MIN_SCORE = 0.0

preds33 = []
seen33 = set()

t_start33 = time.perf_counter()

# ================================================================
# 4. INFERENCE ON ALL 5000 COCO-VAL IMAGES
# ================================================================

with torch.inference_mode():

    for batch_num33, batch33 in enumerate(
        source_val_loader,
        start=1
    ):

        # Encoded DataLoader:
        # (images, raw_targets, encoded_targets)

        images33, targets33 = batch33[0], batch33[1]

        images33 = images33.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16
        ):
            outputs33 = model33(images33)

        # Decoder receives FP32 tensors.
        decoded33 = decode_centernet(
            {
                key: value.float()
                for key, value in outputs33.items()
            },
            K=COCO33_TOPK,
            score_threshold=COCO33_MIN_SCORE
        )

        assert len(decoded33) == len(targets33)

        # ========================================================
        # 5. RESTORE ORIGINAL COCO IMAGE COORDINATES
        # ========================================================

        for pred33, target33 in zip(
            decoded33,
            targets33
        ):

            image_id33 = int(
                target33['image_id'].item()
            )

            assert image_id33 in img_ids33

            assert image_id33 not in seen33, (
                f'Duplicate image ID {image_id33}'
            )

            seen33.add(image_id33)

            original_h33, original_w33 = map(
                int,
                target33['original_size'].tolist()
            )

            pad_x33, pad_y33 = map(
                float,
                target33['pad'].tolist()
            )

            scale33 = float(
                target33['scale'].item()
            )

            assert math.isfinite(scale33)
            assert scale33 > 0

            boxes33 = (
                pred33['boxes']
                .detach()
                .cpu()
                .float()
                .clone()
            )

            scores33 = (
                pred33['scores']
                .detach()
                .cpu()
                .float()
            )

            labels33 = (
                pred33['labels']
                .detach()
                .cpu()
                .long()
            )

            # Undo letterbox padding and resize scale.
            boxes33[:, [0, 2]] = (
                (
                    boxes33[:, [0, 2]] - pad_x33
                ) / scale33
            ).clamp(0, original_w33)

            boxes33[:, [1, 3]] = (
                (
                    boxes33[:, [1, 3]] - pad_y33
                ) / scale33
            ).clamp(0, original_h33)

            # ====================================================
            # 6. CONVERT XYXY TO COCO XYWH
            # ====================================================

            for bbox33, score33, label33 in zip(
                boxes33.tolist(),
                scores33.tolist(),
                labels33.tolist()
            ):

                x1, y1, x2, y2 = map(
                    float,
                    bbox33
                )

                score33 = float(score33)
                label33 = int(label33)

                if not (
                    0 <= label33 < 80
                    and math.isfinite(score33)
                    and all(
                        math.isfinite(x)
                        for x in (x1, y1, x2, y2)
                    )
                ):
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                preds33.append({
                    'image_id': image_id33,
                    'category_id': cat_ids33[label33],
                    'bbox': [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    ],
                    'score': score33
                })

        if (
            batch_num33 == 1
            or batch_num33 % 100 == 0
        ):
            print(
                f'Processed {len(seen33)}/5000 images; '
                f'{len(preds33)} valid detections'
            )

# ================================================================
# 7. INFERENCE VALIDATION
# ================================================================

assert seen33 == img_ids33, (
    'Validation image coverage incomplete.'
)

assert preds33, (
    'No detections generated.'
)

torch.cuda.synchronize()

t_seconds33 = (
    time.perf_counter() - t_start33
)

print(
    f'Inference and decoding completed in '
    f'{t_seconds33 / 60:.2f} minutes'
)

print(
    'Total valid detections:',
    len(preds33)
)

# ================================================================
# 8. OFFICIAL COCO EVALUATION
# ================================================================

gt33 = val_dataset.coco

dt33 = gt33.loadRes(
    preds33
)

eval33 = COCOeval(
    gt33,
    dt33,
    iouType='bbox'
)

eval33.params.imgIds = sorted(
    img_ids33
)

eval33.params.catIds = sorted(
    cat_ids33
)

eval33.params.iouThrs = np.linspace(
    0.50,
    0.95,
    10
)

eval33.params.maxDets = [
    1,
    10,
    100
]

print('\nRunning official COCOeval...')

eval33.evaluate()
eval33.accumulate()
eval33.summarize()

# ================================================================
# 9. AP AT EACH IoU THRESHOLD
# ================================================================

# Precision tensor:
# [IoU, Recall, Category, Area, MaxDets]

precision33 = eval33.eval['precision']

assert precision33.shape[2] == 80

area_idx33 = (
    eval33.params.areaRngLbl.index('all')
)

maxdet_idx33 = (
    eval33.params.maxDets.index(100)
)

AP_iou33 = {}

for i33, threshold33 in enumerate(
    eval33.params.iouThrs
):

    vals33 = precision33[
        i33,
        :,
        :,
        area_idx33,
        maxdet_idx33
    ]

    # -1 means unavailable in COCOeval.
    available33 = vals33[
        vals33 > -1
    ]

    assert available33.size > 0

    AP_iou33[
        f'{threshold33:.2f}'
    ] = float(
        available33.mean()
    )

# ================================================================
# 10. COLLECT ALL OFFICIAL METRICS
# ================================================================

metrics33 = {
    'mAP_50_95': float(eval33.stats[0]),
    'AP50': float(eval33.stats[1]),
    'AP75': float(eval33.stats[2]),

    # COCOeval does not report AP95 directly
    # in its standard summary.
    'AP95': AP_iou33['0.95'],

    'AP_small': float(eval33.stats[3]),
    'AP_medium': float(eval33.stats[4]),
    'AP_large': float(eval33.stats[5]),

    'AR1': float(eval33.stats[6]),
    'AR10': float(eval33.stats[7]),
    'AR100': float(eval33.stats[8])
}

# ================================================================
# 11. CROSS-CHECK AP50 AGAINST STEP 30
# ================================================================

expected33 = stored_stats33.get(
    'official_ap50'
)

if expected33 is not None:

    absolute_difference33 = abs(
        metrics33['AP50'] - float(expected33)
    )

    print(
        '\nAP50 vs Step 30 absolute difference:',
        absolute_difference33
    )

    assert absolute_difference33 < 0.001, (
        'AP50 disagrees with Step 30; '
        'verify checkpoint, preprocessing and decoding.'
    )

# ================================================================
# 12. PRINT FINAL RESULTS — ALL 80 CLASSES
# ================================================================

print('\n' + '=' * 72)
print('OFFICIAL COCO RESULTS — BALANCED80K V1.3')
print('ALL 80 CLASSES — NO PER-CLASS BREAKDOWN')
print('=' * 72)

for k33 in (
    'mAP_50_95',
    'AP50',
    'AP75',
    'AP95',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR1',
    'AR10',
    'AR100'
):

    print(
        f'{k33:<16}: '
        f'{100 * metrics33[k33]:.4f}%'
    )

print('\nAP for each IoU:')

for iou33, ap33 in AP_iou33.items():

    print(
        f'IoU {iou33} : '
        f'{100 * ap33:.4f}%'
    )

# ================================================================
# 13. SAVE FINAL EVALUATION RESULTS
# ================================================================

output_dir33 = os.path.dirname(
    BEST_CHECKPOINT
)

os.makedirs(
    output_dir33,
    exist_ok=True
)

summary_path33 = os.path.join(
    output_dir33,
    'step33_v13_official_coco_all80.json'
)

summary33 = {
    'experiment': CONFIG['experiment_name'],
    'checkpoint_epoch': ckpt33_epoch,
    'dataset': 'COCO 2017 val',
    'num_images': len(seen33),
    'num_categories': 80,
    'num_predictions': len(preds33),
    'metrics': metrics33,
    'AP_per_IoU': AP_iou33
}

with open(
    summary_path33 + '.tmp',
    'w',
    encoding='utf-8'
) as f33:

    json.dump(
        summary33,
        f33,
        indent=2,
        allow_nan=False
    )

os.replace(
    summary_path33 + '.tmp',
    summary_path33
)

print(
    '\nResults saved:',
    summary_path33
)

print('\nSTEP 33 PASSED')
print('No training performed.')
print('No checkpoint weights changed.')
