# ============================================================
# STEP 33 — OFFICIAL FULL COCOeval
# ALL 80 CLASSES / ALL 5000 COCO-VAL IMAGES
# NO TRAINING
# ============================================================

import os
import json
import math
import io
import contextlib
import gc

import numpy as np
import torch

from pycocotools.cocoeval import COCOeval

assert len(val_dataset) == 5000
assert not val_dataset.augment
assert os.path.isfile(BEST_CHECKPOINT)

OFFICIAL_COCO_CACHE = globals().get(
    "OFFICIAL_COCO_CACHE", {}
)


# ============================================================
# 1. REUSABLE OFFICIAL EVALUATOR
# ============================================================

def evaluate_official_checkpoint(path, label):

    if path in OFFICIAL_COCO_CACHE:
        print("Using cached COCOeval:", label)
        return OFFICIAL_COCO_CACHE[path]

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt

    model_eval = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    model_eval.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )

    model_eval.eval()

    detections = []
    seen = set()

    # ========================================================
    # 2. FULL COCO-VAL INFERENCE
    # ========================================================

    with torch.inference_mode():

        for batch in source_val_loader:

            images, targets = batch[0], batch[1]

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                raw = model_eval(images)

            outputs = {
                k: v.float()
                for k, v in raw.items()
            }

            predictions = decode_centernet(
                outputs,
                K=100,
                score_threshold=0.0
            )

            # =================================================
            # 3. RESTORE ORIGINAL COCO COORDINATES
            # =================================================

            for pred, target in zip(
                predictions, targets
            ):

                image_id = int(
                    target["image_id"].item()
                )

                assert image_id not in seen
                seen.add(image_id)

                height, width = map(
                    int,
                    target["original_size"].tolist()
                )

                pad_x, pad_y = map(
                    float,
                    target["pad"].tolist()
                )

                scale = float(
                    target["scale"].item()
                )

                assert scale > 0

                boxes = (
                    pred["boxes"]
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                )

                scores = (
                    pred["scores"]
                    .detach()
                    .cpu()
                    .float()
                    .tolist()
                )

                labels = (
                    pred["labels"]
                    .detach()
                    .cpu()
                    .long()
                    .tolist()
                )

                boxes[:, [0, 2]] = (
                    (boxes[:, [0, 2]] - pad_x) / scale
                ).clamp(0, width)

                boxes[:, [1, 3]] = (
                    (boxes[:, [1, 3]] - pad_y) / scale
                ).clamp(0, height)

                for box, score, cls in zip(
                    boxes.tolist(),
                    scores,
                    labels
                ):

                    x1, y1, x2, y2 = map(
                        float, box
                    )

                    if (
                        x2 <= x1
                        or y2 <= y1
                        or not math.isfinite(score)
                    ):
                        continue

                    assert 0 <= cls < 80

                    detections.append({
                        "image_id": image_id,
                        "category_id": int(
                            val_dataset.cat_ids[cls]
                        ),
                        "bbox": [
                            x1, y1,
                            x2 - x1,
                            y2 - y1
                        ],
                        "score": float(score)
                    })

    assert seen == set(
        map(int, val_dataset.image_ids)
    )

    assert detections

    # ========================================================
    # 4. OFFICIAL COCOeval
    # ========================================================

    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        dt = val_dataset.coco.loadRes(
            detections
        )

        ce = COCOeval(
            val_dataset.coco,
            dt,
            iouType="bbox"
        )

        ce.params.imgIds = sorted(
            map(int, val_dataset.image_ids)
        )

        ce.params.catIds = sorted(
            map(int, val_dataset.cat_ids)
        )

        ce.params.maxDets = [
            1, 10, 100
        ]

        ce.params.iouThrs = np.linspace(
            0.50, 0.95, 10
        )

        ce.evaluate()
        ce.accumulate()
        ce.summarize()

    # ========================================================
    # 5. AP AT EVERY IoU THRESHOLD
    # ========================================================

    precision = ce.eval["precision"]

    area_idx = ce.params.areaRngLbl.index(
        "all"
    )

    det_idx = ce.params.maxDets.index(
        100
    )

    per_iou = {}

    for idx, threshold in enumerate(
        ce.params.iouThrs
    ):

        values = precision[
            idx,
            :,
            :,
            area_idx,
            det_idx
        ]

        valid = values[values >= 0]

        per_iou[f"{threshold:.2f}"] = (
            float(valid.mean())
            if len(valid)
            else None
        )

    # ========================================================
    # 6. COCO AP / AR METRICS
    # ========================================================

    names = [
        "mAP_50_95",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large"
    ]

    metrics = {
        name: float(value)
        for name, value in zip(
            names, ce.stats
        )
    }

    metrics["AP95"] = per_iou["0.95"]

    result = {
        "label": label,
        "checkpoint": path,
        "epoch": int(ckpt["epoch"]),
        "images": len(seen),
        "detections": len(detections),
        "metrics": metrics,
        "AP_by_IoU": per_iou
    }

    OFFICIAL_COCO_CACHE[path] = result

    del model_eval
    del ce
    del dt
    del ckpt

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 7. EVALUATE SELECTED CHECKPOINT
# ============================================================

print("=" * 70)
print("STEP 33 — OFFICIAL COCO EVALUATION")
print("=" * 70)

step33_result = evaluate_official_checkpoint(
    BEST_CHECKPOINT,
    "V1.2_SELECTED"
)

for key, value in step33_result["metrics"].items():
    print(
        f"{key:18s}: {100*value:.4f}%"
    )

print("\nAP PER IoU")

for key, value in step33_result["AP_by_IoU"].items():
    print(
        key,
        f"{100*value:.4f}%"
    )

print("STEP 33 PASSED — NO TRAINING")
