
# STEP 31 — Official COCO bbox evaluation A2 vs B2

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

source_val_loader, _ = build_val_loader(
    SEED + 3301,
    encoded=True
)

assert len(source_val_loader) == 625

B2_CKPT = os.path.join(
    B2_DIR,
    "epoch2_add16k.pth"
)

paths = {
    "A2_control": A2_CONTROL_CKPT,
    B2_ARM: B2_CKPT
}


def b2_official_eval(path, arm):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == arm
    assert int(ck["additional_exposures"]) == 16000
    assert ck["gaussian_radius_mode"] == "v1_2_original"

    net = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    if arm == "A2_control":
        load_a2_into_b2(
            net,
            ck["model_state_dict"]
        )
    else:
        net.load_state_dict(
            ck["model_state_dict"],
            strict=True
        )

    net.eval()
    detections = []
    seen = set()

    with torch.inference_mode():
        for images, targets, _ in source_val_loader:
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                outputs = net(
                    images.to(
                        device,
                        non_blocking=True
                    )
                )

            batch_predictions = decode_centernet(
                {
                    k: v.float()
                    for k, v in outputs.items()
                },
                K=100,
                score_threshold=0.0
            )

            for pred, t in zip(
                batch_predictions,
                targets
            ):
                image_id = int(t["image_id"])

                assert image_id not in seen
                seen.add(image_id)

                scale = float(t["scale"])
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

                boxes[:, [0, 2]] = (
                    (boxes[:, [0, 2]] - px) / scale
                ).clamp(0, W)

                boxes[:, [1, 3]] = (
                    (boxes[:, [1, 3]] - py) / scale
                ).clamp(0, H)

                for bb, score, cl in zip(
                    boxes.tolist(),
                    pred["scores"].detach().cpu().tolist(),
                    pred["labels"].detach().cpu().tolist()
                ):
                    x1, y1, x2, y2 = map(float, bb)

                    if (
                        x2 <= x1
                        or y2 <= y1
                        or not math.isfinite(score)
                    ):
                        continue

                    detections.append({
                        "image_id": image_id,
                        "category_id": int(
                            val_dataset.cat_ids[int(cl)]
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

    with contextlib.redirect_stdout(io.StringIO()):
        dt = val_dataset.coco.loadRes(detections)

        ce = COCOeval(
            val_dataset.coco,
            dt,
            iouType="bbox"
        )

        ce.params.imgIds = sorted(seen)
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
        zip(names, map(float, ce.stats))
    )

    person_pos = ce.params.catIds.index(1)

    precision = ce.eval["precision"][
        :, :,
        person_pos,
        ce.params.areaRngLbl.index("all"),
        2
    ]

    def valid_mean(v):
        v = v[v >= 0]
        return (
            float(v.mean())
            if len(v) > 0 else None
        )

    person = {
        "person_AP_50_95": valid_mean(precision),
        "person_AP50": valid_mean(precision[0]),
        "person_AP75": valid_mean(precision[5])
    }

    result = {
        "arm": arm,
        "checkpoint": path,
        "images": len(seen),
        "detections": len(detections),
        "metrics": metrics,
        "person_metrics": person
    }

    del net, ck, ce
    gc.collect()
    torch.cuda.empty_cache()

    return result


results31 = {
    arm: b2_official_eval(path, arm)
    for arm, path in paths.items()
}

for field in results31["A2_control"]["metrics"]:
    a = results31["A2_control"]["metrics"][field]
    b = results31[B2_ARM]["metrics"][field]

    print(
        f"{field:<18} "
        f"A2={100*a:.4f}% "
        f"B2={100*b:.4f}% "
        f"Δ={100*(b-a):+.4f}pp"
    )

for field in results31["A2_control"]["person_metrics"]:
    a = results31["A2_control"]["person_metrics"][field]
    b = results31[B2_ARM]["person_metrics"][field]

    print(
        f"{field:<18} "
        f"A2={100*a:.4f}% "
        f"B2={100*b:.4f}% "
        f"Δ={100*(b-a):+.4f}pp"
    )

with open(
    os.path.join(
        PILOT_ROOT,
        "pilot_A2_vs_B2_COCOeval.json"
    ),
    "w"
) as f:
    json.dump(
        results31,
        f,
        indent=2,
        allow_nan=False
    )

print("STEP 31 PASS — official A2 vs B2")
