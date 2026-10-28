
# ============================================================
# STEP 31 — OFFICIAL COCOeval A3 vs A3-v2
# FULL 5000 COCO-VAL / NO TRAINING
# ============================================================

import os
import io
import json
import math
import gc
import contextlib

import numpy as np
import torch

from pycocotools.cocoeval import COCOeval

assert "SELECTED_A3_V2" in globals(), (
    "Run STEP 30 first."
)

assert SELECTED_A3_V2 in (
    A3_CKPT,
    V2_BEST
)

assert len(val_dataset) == 5000
assert not val_dataset.augment
assert os.path.isfile(SELECTED_A3_V2)

# Raw targets cukup untuk official COCOeval.
# Tidak perlu membuat heatmap GT pada DataLoader validasi.

val31, _ = build_val_loader(
    int(SEED) + 3301,
    encoded=False
)

assert len(val31) == 625


def evaluate_checkpoint_a3v2(path, label):

    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] in (
        "A3_neighbor_R1",
        "A3_v2"
    )

    assert (
        ck["gaussian_radius_mode"]
        == "v1_2_original"
    )

    assert float(ck["neighbor_lambda"]) == 0.25

    net31 = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    net31.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )

    net31.eval()

    detections = []
    seen = set()

    # --------------------------------------------------------
    # 1. INFERENCE
    # --------------------------------------------------------

    with torch.inference_mode():

        for images, raw_targets in val31:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):

                logits = net31(
                    images.to(
                        device,
                        non_blocking=True
                    )
                )

            preds = decode_centernet(
                {
                    k: v.float()
                    for k, v in logits.items()
                },
                K=100,
                score_threshold=0.0
            )

            for pred, target in zip(
                preds,
                raw_targets
            ):

                image_id = int(
                    target["image_id"]
                )

                assert image_id not in seen
                seen.add(image_id)

                H, W = map(
                    int,
                    target["original_size"].tolist()
                )

                s = float(
                    target["scale"]
                )

                pad_x, pad_y = map(
                    float,
                    target["pad"].tolist()
                )

                assert s > 0

                boxes = (
                    pred["boxes"]
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                )

                # Reverse letterbox transform.
                boxes[:, [0, 2]] = (
                    (boxes[:, [0, 2]] - pad_x) / s
                ).clamp(0, W)

                boxes[:, [1, 3]] = (
                    (boxes[:, [1, 3]] - pad_y) / s
                ).clamp(0, H)

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

                for bb, score, cls in zip(
                    boxes.tolist(),
                    scores,
                    labels
                ):

                    x1, y1, x2, y2 = map(
                        float, bb
                    )

                    if (
                        x2 <= x1
                        or y2 <= y1
                        or not math.isfinite(float(score))
                    ):
                        continue

                    assert 0 <= int(cls) < 80

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
    ), "Missing COCO-Val images"

    assert detections, (
        "Decoder produced zero valid detections."
    )

    # --------------------------------------------------------
    # 2. OFFICIAL COCOeval
    # --------------------------------------------------------

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

        ce.params.maxDets = [1, 10, 100]

        ce.params.iouThrs = np.linspace(
            0.5, 0.95, 10
        )

        ce.evaluate()
        ce.accumulate()
        ce.summarize()

    keys = (
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
    )

    metrics = {
        key: float(x)
        for key, x in zip(keys, ce.stats)
    }

    # --------------------------------------------------------
    # 3. PERSON CLASS AP
    # --------------------------------------------------------

    person_k = ce.params.catIds.index(1)

    all_area = (
        ce.params.areaRngLbl.index("all")
    )

    person_precision = ce.eval["precision"][
        :,
        :,
        person_k,
        all_area,
        2
    ]

    def valid_mean(x):
        good = x[x >= 0]

        return (
            float(good.mean())
            if good.size
            else None
        )

    person_metrics = {
        "person_AP_50_95": valid_mean(
            person_precision
        ),
        "person_AP50": valid_mean(
            person_precision[0]
        ),
        "person_AP75": valid_mean(
            person_precision[5]
        )
    }

    out = {
        "label": label,
        "checkpoint": path,
        "images": len(seen),
        "detections": len(detections),
        "metrics": metrics,
        "person_metrics": person_metrics
    }

    del net31, ck, ce, dt

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# ============================================================
# 4. SELECT CHECKPOINTS
# ============================================================

paths31 = {
    "A3_176K_reference": A3_CKPT
}

if (
    os.path.abspath(SELECTED_A3_V2)
    != os.path.abspath(A3_CKPT)
):
    paths31["A3_v2_selected"] = (
        SELECTED_A3_V2
    )


# ============================================================
# 5. RUN OFFICIAL EVALUATION
# ============================================================

results31 = {}

for label, path in paths31.items():

    print("\nCOCOeval:", label)
    print("Checkpoint:", path)

    results31[label] = (
        evaluate_checkpoint_a3v2(
            path,
            label
        )
    )

    current_map = (
        results31[label]
        ["metrics"]["mAP_50_95"]
    )

    print(
        "mAP50:95 =",
        round(100 * current_map, 4),
        "%"
    )


# ============================================================
# 6. COMPARE A3-v2 MINUS A3
# ============================================================

if "A3_v2_selected" in results31:

    old = results31[
        "A3_176K_reference"
    ]

    new = results31[
        "A3_v2_selected"
    ]

    print(
        "\nDELTA A3-v2 minus A3 "
        "(percentage points)"
    )

    for k in old["metrics"]:

        delta = 100 * (
            new["metrics"][k]
            - old["metrics"][k]
        )

        print(
            f"{k:16s} {delta:+.4f} pp"
        )

    for k in old["person_metrics"]:

        x = old["person_metrics"][k]
        y = new["person_metrics"][k]

        if x is not None and y is not None:

            print(
                f"{k:16s} {100*(y-x):+.4f} pp"
            )

else:

    print(
        "No A3-v2 checkpoint beat A3 reference."
    )

    print(
        "Original A3 176K retained."
    )


# ============================================================
# 7. SAVE JSON
# ============================================================

out_path31 = os.path.join(
    V2_ROOT,
    "A3_v2_official_COCOeval.json"
)

with open(
    out_path31,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results31,
        f,
        indent=2,
        allow_nan=False
    )

print("Saved:", out_path31)
print("STEP 31 PASS — official COCOeval complete")
