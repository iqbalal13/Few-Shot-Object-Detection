
# ============================================================
# STEP 33 — B1 OFFICIAL COCO EVALUATION
# 80 CLASSES / ALL 5000 COCO-VAL IMAGES
# FIXED 160K + BEST CHECKPOINT
# NO TRAINING
# ============================================================

import os
import gc
import math
import io
import contextlib

import numpy as np
import torch

from pycocotools.cocoeval import COCOeval


assert len(val_dataset) == 5000
assert not val_dataset.augment
assert os.path.isfile(BEST_CHECKPOINT)
assert os.path.isfile(milestone30(20))

OFFICIAL_COCO_CACHE = {}


def evaluate_official_checkpoint(path, label):
    if path in OFFICIAL_COCO_CACHE:
        return OFFICIAL_COCO_CACHE[path]

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert "model_state_dict" in ckpt

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
                raw_outputs = model_eval(images)

            outputs = {
                k: v.float()
                for k, v in raw_outputs.items()
            }

            predictions = decode_centernet(
                outputs,
                K=100,
                score_threshold=0.0
            )

            for pred, target in zip(
                predictions,
                targets
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

                    detections.append({
                        "image_id": image_id,
                        "category_id": int(
                            val_dataset.cat_ids[int(cls)]
                        ),
                        "bbox": [
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1
                        ],
                        "score": float(score)
                    })

    assert seen == set(
        map(int, val_dataset.image_ids)
    )

    assert len(detections) > 0

    with contextlib.redirect_stdout(io.StringIO()):
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

        ce.params.maxDets = [1, 10, 100]
        ce.params.iouThrs = np.linspace(
            0.50, 0.95, 10
        )

        ce.evaluate()
        ce.accumulate()
        ce.summarize()

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

    precision = ce.eval["precision"]
    area_idx = ce.params.areaRngLbl.index("all")
    det_idx = ce.params.maxDets.index(100)

    ap_by_iou = {}

    for idx, threshold in enumerate(
        ce.params.iouThrs
    ):
        values = precision[
            idx, :, :, area_idx, det_idx
        ]

        valid = values[values >= 0]

        ap_by_iou[f"{threshold:.2f}"] = (
            float(valid.mean())
            if len(valid)
            else None
        )

    metrics["AP95"] = ap_by_iou["0.95"]

    # Person category in original COCO IDs = 1
    person_idx = ce.params.catIds.index(1)

    person_precision = precision[
        :, :, person_idx, area_idx, det_idx
    ]

    def valid_average(array):
        v = array[array >= 0]
        return float(v.mean()) if len(v) else None

    person_metrics = {
        "person_AP_50_95": valid_average(
            person_precision
        ),
        "person_AP50": valid_average(
            person_precision[0]
        ),
        "person_AP75": valid_average(
            person_precision[5]
        )
    }

    result = {
        "label": label,
        "checkpoint": path,
        "epoch": int(ckpt["epoch"]),
        "images": len(seen),
        "detections": len(detections),
        "metrics": metrics,
        "person_metrics": person_metrics,
        "AP_by_IoU": ap_by_iou
    }

    OFFICIAL_COCO_CACHE[path] = result

    del model_eval
    del ce
    del ckpt

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# EVALUATE FIXED 160K FIRST
# ============================================================

print("=" * 70)
print("STEP 33 — B1 OFFICIAL COCOeval")
print("=" * 70)

step33_160k_result = evaluate_official_checkpoint(
    milestone30(20),
    "B1_FIXED_160K"
)

# If best is epoch 20, weights are identical.
# Avoid a redundant full evaluation.
if int(checkpoint["epoch"]) == 20:
    step33_result = dict(step33_160k_result)
    step33_result["label"] = "B1_BEST_EPOCH20"
    step33_result["checkpoint"] = BEST_CHECKPOINT
else:
    step33_result = evaluate_official_checkpoint(
        BEST_CHECKPOINT,
        "B1_BEST"
    )


def print_official_result(result):
    print("\n" + "=" * 55)
    print(result["label"])
    print("=" * 55)

    print("Epoch      :", result["epoch"])
    print("Images     :", result["images"])
    print("Detections :", result["detections"])

    print("\nALL 80 CLASSES")

    for name, value in result["metrics"].items():
        if value is not None:
            print(
                f"{name:20s}: {100 * value:.4f}%"
            )

    print("\nPERSON CLASS")

    for name, value in result["person_metrics"].items():
        if value is not None:
            print(
                f"{name:20s}: {100 * value:.4f}%"
            )

    print("\nAP BY IoU")

    for name, value in result["AP_by_IoU"].items():
        if value is not None:
            print(
                f"IoU {name}: {100 * value:.4f}%"
            )


print_official_result(step33_160k_result)

if int(checkpoint["epoch"]) != 20:
    print_official_result(step33_result)

print("\nSTEP 33 PASSED — NO TRAINING")
