
# ============================================================
# STEP 31 — OFFICIAL COCOeval
# A3 176K vs BEST A3-v2
# NO TRAINING
# ============================================================

import os
import math
import json
import io
import contextlib
import gc

import numpy as np
import torch

from pycocotools.cocoeval import COCOeval


assert len(val_dataset) == 5000
assert not val_dataset.augment

source_val_loader, _ = build_val_loader(
    SEED + 3301,
    encoded=True
)

assert len(source_val_loader) == 625


def eval31(path, label):

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
        int(ck["additional_exposures"])
        >= 16000
    )

    assert (
        ck["gaussian_radius_mode"]
        == "v1_2_original"
    )

    model_eval = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    model_eval.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )

    model_eval.eval()

    detections = []
    seen = set()

    with torch.inference_mode():

        for batch in source_val_loader:

            images, targets = (
                batch[0],
                batch[1]
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                raw = model_eval(
                    images.to(
                        device,
                        non_blocking=True
                    )
                )

            predictions = decode_centernet(
                {
                    k: v.float()
                    for k, v in raw.items()
                },
                K=100,
                score_threshold=0.0
            )

            for pred, t in zip(
                predictions,
                targets
            ):
                image_id = int(
                    t["image_id"]
                )

                assert image_id not in seen
                seen.add(image_id)

                scale = float(
                    t["scale"]
                )

                px, py = map(
                    float,
                    t["pad"].tolist()
                )

                H, W = map(
                    int,
                    t["original_size"].tolist()
                )

                boxes = (
                    pred["boxes"]
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                )

                # Reverse letterbox.
                boxes[:, [0, 2]] = (
                    (boxes[:, [0, 2]] - px) / scale
                ).clamp(0, W)

                boxes[:, [1, 3]] = (
                    (boxes[:, [1, 3]] - py) / scale
                ).clamp(0, H)

                scores = (
                    pred["scores"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                labels = (
                    pred["labels"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                for bb, sc, cls in zip(
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
                        or not math.isfinite(sc)
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
                        "score": float(sc)
                    })

    assert seen == set(
        map(int, val_dataset.image_ids)
    )

    # Official COCO evaluation.
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
            0.5, 0.95, 10
        )

        ce.evaluate()
        ce.accumulate()
        ce.summarize()

    names = (
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

    metrics = dict(
        zip(
            names,
            map(float, ce.stats)
        )
    )

    # Person COCO category id = 1.
    precision = ce.eval["precision"]

    person_idx = ce.params.catIds.index(
        1
    )

    pi = precision[
        :,
        :,
        person_idx,
        ce.params.areaRngLbl.index("all"),
        2
    ]

    def positive_mean(values):
        good = values[values >= 0]

        return (
            float(good.mean())
            if len(good)
            else None
        )

    person = {
        "person_AP_50_95": positive_mean(pi),
        "person_AP50": positive_mean(pi[0]),
        "person_AP75": positive_mean(pi[5])
    }

    result = {
        "label": label,
        "checkpoint": path,
        "images": len(seen),
        "detections": len(detections),
        "metrics": metrics,
        "person_metrics": person
    }

    del model_eval, ce, ck

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# EVALUATE REFERENCE AND SELECTED MODEL
# ============================================================

SELECTED_A3_V2 = (
    V2_BEST
    if os.path.isfile(V2_BEST)
    else A3_CKPT
)

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

results31 = {}

for label, path in paths31.items():

    assert os.path.isfile(path), path

    print(
        "Official COCOeval:",
        label,
        path
    )

    results31[label] = eval31(
        path,
        label
    )

    print(
        "mAP:",
        results31[label]["metrics"]["mAP_50_95"]
    )


# ============================================================
# METRIC DELTAS
# ============================================================

if "A3_v2_selected" in results31:

    a = results31[
        "A3_176K_reference"
    ]

    b = results31[
        "A3_v2_selected"
    ]

    print(
        "\nA3-v2 minus A3 reference "
        "(percentage points)"
    )

    for key in a["metrics"]:
        print(
            key,
            round(
                100 * (
                    b["metrics"][key]
                    - a["metrics"][key]
                ),
                4
            )
        )

    for key in a["person_metrics"]:

        if (
            a["person_metrics"][key] is not None
            and b["person_metrics"][key] is not None
        ):
            print(
                key,
                round(
                    100 * (
                        b["person_metrics"][key]
                        - a["person_metrics"][key]
                    ),
                    4
                )
            )

else:
    print(
        "No A3-v2 checkpoint beat baseline. "
        "Retain A3 176K."
    )


# ============================================================
# SAVE
# ============================================================

with open(
    os.path.join(
        V2_ROOT,
        "A3_v2_official_COCOeval.json"
    ),
    "w"
) as f:
    json.dump(
        results31,
        f,
        indent=2,
        allow_nan=False
    )

print("STEP 31 PASS")
