
# ============================================================
# STEP 33 — OFFICIAL COCO mAP / IoU EVALUATION
# ALL 80 CLASSES — BALANCED80K V1.3
# NO TRAINING — NO WEIGHT UPDATES
# ============================================================

import os
import json
import time
import math
import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("=" * 72)
print("STEP 33 — OFFICIAL COCO BBOX mAP / IoU")
print("=" * 72)

# ============================================================
# 1. CHECK CONFIGURATION AND CHECKPOINT
# ============================================================

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

assert CONFIG["num_classes"] == 80
assert len(val_dataset) == 5000
assert len(val_dataset.cat_ids) == 80
assert torch.cuda.is_available()
assert os.path.isfile(BEST_CHECKPOINT)

coco33_ckpt = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert coco33_ckpt["experiment_name"] == (
    CONFIG["experiment_name"]
)

assert coco33_ckpt["source_images_total"] == 80000

if "source_manifest_sha256" in globals():
    assert (
        coco33_ckpt["source_manifest_sha256"]
        == source_manifest_sha256
    )

coco33_epoch = int(coco33_ckpt["epoch"])

# Always reload the saved best model
coco33_model = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

coco33_model.load_state_dict(
    coco33_ckpt["model_state_dict"],
    strict=True
)

coco33_model.eval()

del coco33_ckpt

# Use original COCO annotations, not letterboxed GT
coco33_gt = val_dataset.coco

coco33_label_to_catid = tuple(
    int(x) for x in val_dataset.cat_ids
)

coco33_image_ids = list(
    map(int, val_dataset.image_ids)
)

assert set(coco33_gt.getImgIds()) == set(
    coco33_image_ids
)

# ============================================================
# 2. EVALUATION CONFIGURATION
# ============================================================

COCO33_TOPK = 100

# Important: do not use confidence 0.50 for AP.
# AP evaluates the ranking across confidence scores.
COCO33_MIN_SCORE = 0.0

COCO33_SAVE_PREDICTIONS = True

COCO33_SAVE_DIR = (
    FULL_CHECKPOINT_DIR
    if "FULL_CHECKPOINT_DIR" in globals()
    else CONFIG["checkpoint_root"]
)

os.makedirs(
    COCO33_SAVE_DIR,
    exist_ok=True
)

# ============================================================
# 3. CONVERT PREDICTIONS TO ORIGINAL COCO COORDINATES
# ============================================================

def coco33_convert_predictions(prediction, target):

    image_id = int(
        target["image_id"].item()
    )

    orig_h, orig_w = [
        int(x)
        for x in target["original_size"].tolist()
    ]

    pad_x, pad_y = [
        float(x)
        for x in target["pad"].tolist()
    ]

    scale = float(
        target["scale"].item()
    )

    assert scale > 0
    assert math.isfinite(scale)

    boxes = (
        prediction["boxes"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .copy()
    )

    scores = (
        prediction["scores"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    labels = (
        prediction["labels"]
        .detach()
        .long()
        .cpu()
        .numpy()
    )

    if boxes.size == 0:
        return []

    # Undo 640x640 letterbox preprocessing
    boxes[:, [0, 2]] = (
        boxes[:, [0, 2]] - pad_x
    ) / scale

    boxes[:, [1, 3]] = (
        boxes[:, [1, 3]] - pad_y
    ) / scale

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        orig_w
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        orig_h
    )

    records = []

    for box, score, label in zip(
        boxes, scores, labels
    ):

        label = int(label)

        assert 0 <= label < 80

        if not (
            np.isfinite(box).all()
            and math.isfinite(float(score))
        ):
            continue

        x1, y1, x2, y2 = map(float, box)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        if float(score) < COCO33_MIN_SCORE:
            continue

        records.append({
            "image_id": image_id,
            "category_id": coco33_label_to_catid[label],
            "bbox": [x1, y1, w, h],
            "score": float(score)
        })

    return records


# ============================================================
# 4. FULL COCO-VAL INFERENCE
# ============================================================

coco33_predictions = []
coco33_seen_ids = []

start_time = time.perf_counter()

with torch.inference_mode():

    for batch_index, batch in enumerate(
        val_loader,
        start=1
    ):

        images, targets = batch[0], batch[1]

        images = images.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            outputs = coco33_model(images)

        decode_outputs = {
            name: tensor.float()
            for name, tensor in outputs.items()
        }

        predictions = decode_centernet(
            decode_outputs,
            K=COCO33_TOPK,
            score_threshold=COCO33_MIN_SCORE
        )

        assert len(predictions) == len(targets)

        for prediction, target in zip(
            predictions,
            targets
        ):

            image_id = int(
                target["image_id"].item()
            )

            coco33_seen_ids.append(image_id)

            coco33_predictions.extend(
                coco33_convert_predictions(
                    prediction,
                    target
                )
            )

        if batch_index == 1 or batch_index % 100 == 0:

            print(
                f"Processed {len(coco33_seen_ids)}/5000 "
                f"| detections={len(coco33_predictions)}"
            )

torch.cuda.synchronize()

inference_seconds = (
    time.perf_counter() - start_time
)

assert len(coco33_seen_ids) == 5000

assert len(set(coco33_seen_ids)) == 5000

assert set(coco33_seen_ids) == set(
    coco33_image_ids
)

print(
    "Total valid detections:",
    len(coco33_predictions)
)

# ============================================================
# 5. OFFICIAL COCO EVALUATION
# ============================================================

if coco33_predictions:

    coco33_dt = coco33_gt.loadRes(
        coco33_predictions
    )

else:

    # Handle empty detection list safely
    coco33_dt = COCO()

    coco33_dt.dataset = {
        "images": list(coco33_gt.dataset["images"]),
        "categories": list(coco33_gt.dataset["categories"]),
        "annotations": []
    }

    coco33_dt.createIndex()

coco33_eval = COCOeval(
    coco33_gt,
    coco33_dt,
    iouType="bbox"
)

coco33_eval.params.imgIds = sorted(
    coco33_image_ids
)

coco33_eval.params.catIds = sorted(
    coco33_label_to_catid
)

coco33_eval.params.iouThrs = np.linspace(
    0.50,
    0.95,
    10
)

coco33_eval.params.maxDets = [
    1, 10, 100
]

print("\nRunning official COCOeval...")

coco33_eval.evaluate()
coco33_eval.accumulate()
coco33_eval.summarize()

# ============================================================
# 6. COMPUTE AP FOR EACH IoU — INCLUDING AP95
# ============================================================

# Precision tensor shape:
# [IoU, Recall, Categories, Area, MaxDets]

precision_tensor = coco33_eval.eval["precision"]

assert precision_tensor.shape[2] == 80

area_all_index = (
    coco33_eval.params.areaRngLbl.index("all")
)

maxdet100_index = (
    coco33_eval.params.maxDets.index(100)
)

def coco33_ap_for_iou(iou_index):

    values = precision_tensor[
        iou_index,
        :,
        :,
        area_all_index,
        maxdet100_index
    ]

    # COCOeval uses -1 for unavailable values
    valid = values[values > -1]

    if valid.size == 0:
        return float("nan")

    return float(valid.mean())


ap_per_iou = {}

for index, iou in enumerate(
    coco33_eval.params.iouThrs
):

    ap_per_iou[f"{float(iou):.2f}"] = (
        coco33_ap_for_iou(index)
    )

stats = coco33_eval.stats

coco33_metrics = {
    "mAP_50_95": float(stats[0]),
    "AP50": float(stats[1]),
    "AP75": float(stats[2]),
    "AP95": ap_per_iou["0.95"],
    "AP_small": float(stats[3]),
    "AP_medium": float(stats[4]),
    "AP_large": float(stats[5]),
    "AR_1": float(stats[6]),
    "AR_10": float(stats[7]),
    "AR_100": float(stats[8]),
    "AR_small": float(stats[9]),
    "AR_medium": float(stats[10]),
    "AR_large": float(stats[11])
}

# ============================================================
# 7. FINAL RESULTS — ALL 80 CLASSES COMBINED
# ============================================================

print("\n" + "=" * 72)
print("CENTERNET BALANCED80K — OFFICIAL COCO RESULTS")
print("ALL 80 CLASSES — NO PER-CLASS BREAKDOWN")
print("=" * 72)

for key in (
    "mAP_50_95",
    "AP50",
    "AP75",
    "AP95"
):

    value = coco33_metrics[key]

    print(
        f"{key:14s}: {value:.5f} "
        f"({value * 100:.2f}%)"
    )

print("\nAP at each IoU threshold:")

for iou, ap in ap_per_iou.items():

    print(
        f"IoU={iou}: {ap * 100:.3f}%"
    )

print("\nAdditional COCO metrics:")

for key in (
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large"
):

    value = coco33_metrics[key]

    print(
        f"{key:14s}: {value:.5f} "
        f"({value * 100:.2f}%)"
    )

print(
    f"\nInference + decoding: "
    f"{inference_seconds / 60:.2f} minutes"
)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

summary = {
    "experiment_name": CONFIG["experiment_name"],
    "checkpoint_epoch": coco33_epoch,
    "checkpoint_path": BEST_CHECKPOINT,
    "dataset": "COCO 2017 val",
    "evaluation": "pycocotools.COCOeval bbox",
    "num_images": 5000,
    "num_classes": 80,
    "num_detections": len(coco33_predictions),
    "decoder_topk": COCO33_TOPK,
    "decoder_score_threshold": COCO33_MIN_SCORE,
    "inference_seconds": inference_seconds,
    "metrics": coco33_metrics,
    "ap_per_iou": ap_per_iou
}

summary_path = os.path.join(
    COCO33_SAVE_DIR,
    "step33_coco_bbox_map_all80_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )

if COCO33_SAVE_PREDICTIONS:

    prediction_path = os.path.join(
        COCO33_SAVE_DIR,
        "step33_coco_bbox_all80_detections.json"
    )

    with open(
        prediction_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            coco33_predictions,
            f,
            separators=(",", ":"),
            allow_nan=False
        )

    print(
        "Predictions saved:",
        prediction_path
    )

print(
    "Summary saved:",
    summary_path
)

print("\nSTEP 33 PASSED")
print("No training performed.")
print("No checkpoint weights changed.")
