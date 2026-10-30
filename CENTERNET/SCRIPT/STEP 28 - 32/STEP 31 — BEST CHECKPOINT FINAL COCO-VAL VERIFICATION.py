
# ============================================================
# STEP 31 — THREE-ARM OFFICIAL COCOeval
# K=300, maxDets=[1,10,100]
# A3 176K / OLD A3-v2 / NEW PEAK-GEOMETRY
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

assert int(CONFIG["eval_topk"]) == 300

assert len(val_dataset) == 5000
assert not val_dataset.augment

for path in (
    A3_CKPT,
    OLD_V2_BEST,
    NEW_V2_BEST
):
    assert os.path.isfile(path), path

val31, _ = build_val_loader(
    int(SEED) + 3301,
    encoded=False
)

assert len(val31) == 625


def eval31_peakgeom(path, label):

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

    if label == "A3_v2_peakgeom_candidate":

        assert (
            ck["method_tag"]
            == "peak_aligned_geometry"
        )

        assert (
            float(ck["offset_neighbor_lambda"])
            == 0.25
        )

        assert (
            0 < int(ck["additional_exposures"])
            <= 32000
        )

    if label == "A3_v2_old_208K":

        assert (
            int(ck["additional_exposures"])
            == 32000
        )

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
    # INFERENCE — K300
    # --------------------------------------------------------

    with torch.inference_mode():

        for images, raw_targets in val31:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):

                raw = net31(
                    images.to(
                        device,
                        non_blocking=True
                    )
                )

            preds = decode_centernet(
                {
                    k: v.float()
                    for k, v in raw.items()
                },
                K=CONFIG["eval_topk"],
                score_threshold=0.0
            )

            for pred, t in zip(
                preds,
                raw_targets
            ):

                image_id = int(t["image_id"])

                assert image_id not in seen

                seen.add(image_id)

                H, W = map(
                    int,
                    t["original_size"].tolist()
                )

                scale = float(t["scale"])

                px, py = map(
                    float,
                    t["pad"].tolist()
                )

                assert scale > 0

                boxes = (
                    pred["boxes"]
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )

                # Reverse letterbox transform.
                boxes[:, [0, 2]] = (
                    (boxes[:, [0, 2]] - px)
                    / scale
                ).clamp(0, W)

                boxes[:, [1, 3]] = (
                    (boxes[:, [1, 3]] - py)
                    / scale
                ).clamp(0, H)

                scores = (
                    pred["scores"]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )

                classes = (
                    pred["labels"]
                    .detach()
                    .long()
                    .cpu()
                    .tolist()
                )

                for b, score, cls in zip(
                    boxes.tolist(),
                    scores,
                    classes
                ):

                    x1, y1, x2, y2 = map(
                        float, b
                    )

                    if (
                        x2 <= x1
                        or y2 <= y1
                        or not math.isfinite(score)
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
    )

    assert len(detections) > 0

    # --------------------------------------------------------
    # OFFICIAL COCOeval
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

        ce.params.maxDets = [
            1, 10, 100
        ]

        ce.params.iouThrs = np.linspace(
            0.50,
            0.95,
            10
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
        k: float(v)
        for k, v in zip(keys, ce.stats)
    }

    # Person category ID in COCO = 1.
    pi = ce.eval["precision"][
        :,
        :,
        ce.params.catIds.index(1),
        ce.params.areaRngLbl.index("all"),
        2
    ]

    def validmean(x):
        good = x[x >= 0]

        return (
            float(good.mean())
            if good.size
            else None
        )

    person = {
        "person_AP_50_95": validmean(pi),
        "person_AP50": validmean(pi[0]),
        "person_AP75": validmean(pi[5])
    }

    result = {
        "label": label,
        "checkpoint": path,
        "images": len(seen),
        "detections": len(detections),
        "topk": 300,
        "metrics": metrics,
        "person_metrics": person
    }

    del net31, ck, dt, ce

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ------------------------------------------------------------
# 3 CHECKPOINTS, SAME EVALUATION
# ------------------------------------------------------------

paths31 = {
    "A3_176K_reference": A3_CKPT,
    "A3_v2_old_208K": OLD_V2_BEST,
    "A3_v2_peakgeom_candidate": NEW_V2_BEST
}

results31 = {}

for label, path in paths31.items():

    print("\nEvaluating:", label)

    results31[label] = eval31_peakgeom(
        path,
        label
    )

    print(
        "COCO mAP50:95:",
        f"{100 * results31[label]['metrics']['mAP_50_95']:.4f}%"
    )

    print(
        "Person AP:",
        f"{100 * results31[label]['person_metrics']['person_AP_50_95']:.4f}%"
    )


# ------------------------------------------------------------
# SELECT WINNER BASED ON OFFICIAL mAP
# ------------------------------------------------------------

old = results31["A3_v2_old_208K"]
new = results31["A3_v2_peakgeom_candidate"]

selected_name = (
    "A3_v2_peakgeom_candidate"
    if new["metrics"]["mAP_50_95"]
       > old["metrics"]["mAP_50_95"]
    else "A3_v2_old_208K"
)

selected = results31[selected_name]

# Keep existing QC variable contract.
results31["A3_v2_selected"] = selected

SELECTED_A3_V2 = selected["checkpoint"]

# QC reports are stored in new experiment folder.
V2_ROOT = NEW_V2_ROOT

# History follows whichever checkpoint was selected.
V2_HISTORY = (
    NEW_V2_HISTORY
    if selected_name == "A3_v2_peakgeom_candidate"
    else OLD_V2_HISTORY
)

V2_BEST = SELECTED_A3_V2

ref = results31["A3_176K_reference"]

print("=" * 65)

print("WINNER:", selected_name)
print("Selected:", SELECTED_A3_V2)

print(
    "Delta NEW minus OLD (mAP pp):",
    100 * (
        new["metrics"]["mAP_50_95"]
        - old["metrics"]["mAP_50_95"]
    )
)

print(
    "Delta NEW minus OLD (person AP pp):",
    100 * (
        new["person_metrics"]["person_AP_50_95"]
        - old["person_metrics"]["person_AP_50_95"]
    )
)

print(
    "Selected minus A3 176K (mAP pp):",
    100 * (
        selected["metrics"]["mAP_50_95"]
        - ref["metrics"]["mAP_50_95"]
    )
)

output31 = os.path.join(
    NEW_V2_ROOT,
    "peakgeom_official_COCOeval_3arms.json"
)

with open(
    output31,
    "w"
) as f:
    json.dump(
        results31,
        f,
        indent=2,
        allow_nan=False
    )

print("Saved:", output31)
print("STEP 31 PASS — official COCOeval K300")
