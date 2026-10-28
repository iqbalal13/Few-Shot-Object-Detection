
# ============================================================
# STEP 30 — A3-v2 RESUMABLE SOURCE CONTINUATION
# A3 176K -> maximum +96K = 272K exposures
# NO B2 / NO TRANSFORMER
# ============================================================

import os
import json
import hashlib
import random
import gc
import time

import numpy as np
import torch

from torch.utils.data import Sampler
from google.colab import drive


# ============================================================
# 1. DRIVE AND CHECKPOINT PATHS
# ============================================================

drive.mount(
    "/content/drive",
    force_remount=False
)

A3_ROOT = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_A3_R1_PILOT_FROM_A2_160K"
)

A3_CKPT = os.path.join(
    A3_ROOT,
    "A3_neighbor_R1",
    "epoch2_add16k.pth"
)

V2_ROOT = os.path.join(
    "/content/drive/MyDrive",
    CONFIG["experiment_name"]
)

os.makedirs(
    V2_ROOT,
    exist_ok=True
)

V2_LATEST = os.path.join(
    V2_ROOT,
    "latest.pth"
)

V2_BEST = os.path.join(
    V2_ROOT,
    "best_model.pth"
)

V2_HISTORY = os.path.join(
    V2_ROOT,
    "history.json"
)

V2_MANIFEST = os.path.join(
    V2_ROOT,
    "v2_mixed_manifest.jsonl"
)

assert os.path.isfile(A3_CKPT), A3_CKPT

assert train_dataset.augment
assert not val_dataset.augment

assert CONFIG["a3_neighbor_lambda"] == 0.25
assert CONFIG["gaussian_radius_mode"] == "v1_2_original"

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert CONFIG["v2_uniform_fraction"] == 0.5
assert CONFIG["v2_images_per_epoch"] == 8000


# ============================================================
# 2. VERIFY A3 BASE CHECKPOINT
# ============================================================

base_ck = torch.load(
    A3_CKPT,
    map_location="cpu",
    weights_only=False
)

assert base_ck["arm"] == "A3_neighbor_R1"
assert int(base_ck["additional_exposures"]) == 16000
assert float(base_ck["neighbor_lambda"]) == 0.25

assert (
    base_ck["gaussian_radius_mode"]
    == "v1_2_original"
)

A3_BASE_SHA = (
    "9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f"
)

assert (
    base_ck["base_source_sha256"]
    == A3_BASE_SHA
)

print("A3 checkpoint verified:", A3_CKPT)


# ============================================================
# 3. BUILD MIXED SAMPLING MANIFEST
# 50% uniform + 50% class-balanced
# ============================================================

per_epoch = int(
    CONFIG["v2_images_per_epoch"]
)

N = int(
    CONFIG["v2_epochs"]
)

nu = int(
    per_epoch * CONFIG["v2_uniform_fraction"]
)

nb = per_epoch - nu

assert nb % 80 == 0

assert (
    nu * N
    <= len(train_dataset)
)

imageids = list(
    map(int, train_dataset.image_ids)
)

id_to_idx = {
    im: i
    for i, im in enumerate(imageids)
}

ids_sha = hashlib.sha256(
    ",".join(
        map(str, imageids)
    ).encode()
).hexdigest()

pools = {
    cid: sorted({
        id_to_idx[int(im)]
        for im in train_dataset.coco.catToImgs[cid]
        if int(im) in id_to_idx
    })
    for cid in train_dataset.cat_ids
}

assert len(pools) == 80
assert all(pools.values())


def make_v2_manifest():
    rng = random.Random(
        int(SEED) + 55331
    )

    uniform = list(
        range(len(train_dataset))
    )

    rng.shuffle(uniform)

    manifest = []

    for ep in range(1, N + 1):

        samples = [
            (idx, "uniform")
            for idx in uniform[
                (ep - 1) * nu: ep * nu
            ]
        ]

        for cid in train_dataset.cat_ids:
            pool = pools[cid]
            count = nb // 80

            if count <= len(pool):
                picked = rng.sample(
                    pool, count
                )
            else:
                picked = rng.choices(
                    pool, k=count
                )

            samples.extend(
                (idx, "balanced")
                for idx in picked
            )

        assert len(samples) == per_epoch

        rng.shuffle(samples)

        manifest.extend({
            "epoch": ep,
            "position": j,
            "dataset_index": int(idx),
            "image_id": imageids[int(idx)],
            "stream": stream
        } for j, (idx, stream) in enumerate(samples))

    return manifest


expected_manifest = make_v2_manifest()

with open(
    V2_MANIFEST,
    "w",
    encoding="utf-8"
) as f:
    for item in expected_manifest:
        f.write(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":")
            ) + "\n"
        )

with open(V2_MANIFEST, "rb") as f:
    manifest_sha = hashlib.sha256(
        f.read()
    ).hexdigest()

assert (
    len(expected_manifest)
    == N * per_epoch
)

print("Mixed manifest SHA256:", manifest_sha)
print("Maximum additional exposures:", N * per_epoch)


# ============================================================
# 4. SAMPLER AND UTILITY FUNCTIONS
# ============================================================

class FixedIndexSampler(Sampler):

    def __init__(self, indices):
        self.indices = list(
            map(int, indices)
        )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def v2_reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def v2_atomic_save(obj, path):
    temporary = path + ".tmp"

    torch.save(
        obj,
        temporary
    )

    os.replace(
        temporary,
        path
    )


# ============================================================
# 5. LOAD A3 INTO A3-v2
# ============================================================

criterion.neighbor_lambda = 0.25

v2_reset_seed(
    int(SEED) + 55332
)

net = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

net.load_state_dict(
    base_ck["model_state_dict"],
    strict=True
)

del base_ck

print(
    "A3 model loaded with strict=True"
)


# ============================================================
# 6. OPTIMIZER: DIFFERENTIAL LEARNING RATES
# ============================================================

opt = torch.optim.AdamW(
    [
        {
            "params": list(
                net.backbone.parameters()
            ),
            "lr": CONFIG["v2_backbone_lr"],
            "initial_lr": CONFIG["v2_backbone_lr"]
        },
        {
            "params": (
                list(net.neck.parameters())
                + list(net.heads.parameters())
            ),
            "lr": CONFIG["v2_neck_head_lr"],
            "initial_lr": CONFIG["v2_neck_head_lr"]
        }
    ],
    weight_decay=CONFIG["weight_decay"]
)

sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=N,
    eta_min=CONFIG["v2_eta_min"]
)


# ============================================================
# 7. RESUME PREVIOUS A3-v2 TRAINING
# ============================================================

start = 0

best_map = float(
    CONFIG["v2_starting_map"]
)

best_epoch = 0
no_improve = 0
history = []

if os.path.exists(V2_LATEST):

    ck = torch.load(
        V2_LATEST,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == "A3_v2"
    assert ck["base_checkpoint"] == A3_CKPT

    assert (
        ck["manifest_sha256"]
        == manifest_sha
    )

    assert (
        ck["dataset_ids_sha256"]
        == ids_sha
    )

    assert ck["neighbor_lambda"] == 0.25
    assert ck["v2_epochs"] == N

    assert ck["train_aug_cfg"] == {
        k: CONFIG[k]
        for k in (
            "train_horizontal_flip_p",
            "train_scale_min",
            "train_scale_max",
            "train_translate_frac",
            "train_color_jitter",
            "train_visible_box_min"
        )
    }

    start = int(ck["epoch"])

    assert 0 <= start <= N

    net.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )

    opt.load_state_dict(
        ck["optimizer_state_dict"]
    )

    sch.load_state_dict(
        ck["scheduler_state_dict"]
    )

    history = ck["history"]
    best_map = float(ck["best_map"])
    best_epoch = int(ck["best_epoch"])
    no_improve = int(ck["no_improve"])

    print(
        "Resumed A3-v2 at epoch:",
        start
    )

    del ck


# ============================================================
# 8. VALIDATION LOADER
# ============================================================

val_v2, _ = build_val_loader(
    int(SEED) + 88001,
    encoded=True
)

assert len(val_v2) == 625


# ============================================================
# 9. TRAIN + OFFICIAL COCOeval + SAVE CHECKPOINT
# ============================================================

if no_improve >= CONFIG["v2_early_stop_evals"]:
    print("Previous run already reached early stopping.")

else:

    for ep in range(start + 1, N + 1):

        print(
            f"\nA3-v2 epoch {ep}/{N} "
            f"(+{ep * per_epoch} exposures)"
        )

        rows = expected_manifest[
            (ep - 1) * per_epoch:
            ep * per_epoch
        ]

        assert all(
            x["epoch"] == ep
            and x["position"] == i
            for i, x in enumerate(rows)
        )

        idx = [
            x["dataset_index"]
            for x in rows
        ]

        loader, _ = build_train_loader(
            int(SEED) + 88001 + ep,
            sampler=FixedIndexSampler(idx),
            encoded=True
        )

        assert len(loader) == 1000

        v2_reset_seed(
            int(SEED) + 99001 + ep
        )

        start_t = time.perf_counter()

        stats = train_centernet_epoch(
            net,
            loader,
            opt
        )

        assert stats["batches"] == 1000

        del loader
        gc.collect()

        sch.step()

        result = {
            "epoch": ep,
            "additional_exposures": ep * per_epoch,
            "losses": stats,
            "train_seconds": (
                time.perf_counter() - start_t
            ),
            "lr_backbone": opt.param_groups[0]["lr"],
            "lr_neck_head": opt.param_groups[1]["lr"]
        }

        # Official COCOeval every 2 extra epochs.
        if (
            ep % CONFIG["v2_eval_interval"] == 0
            or ep == N
        ):
            ev = evaluate_centernet(
                net,
                val_v2,
                max_batches=None,
                compute_loss=False,
                compute_official_ap=True,
                score_threshold=0.5,
                iou_threshold=0.5
            )

            assert ev["official_map"] is not None

            ap = float(
                ev["official_map"]
            )

            result["official_map"] = ap
            result["official_ap50"] = ev["official_ap50"]
            result["official_ap75"] = ev["official_ap75"]

            if (
                ap
                > best_map + CONFIG["v2_min_improvement"]
            ):

                best_map = ap
                best_epoch = ep
                no_improve = 0

                v2_atomic_save(
                    {
                        "arm": "A3_v2",
                        "epoch": ep,
                        "additional_exposures": ep * per_epoch,
                        "base_checkpoint": A3_CKPT,
                        "manifest_sha256": manifest_sha,
                        "dataset_ids_sha256": ids_sha,
                        "gaussian_radius_mode": "v1_2_original",
                        "neighbor_lambda": 0.25,
                        "model_state_dict": net.state_dict(),
                        "official_map": ap,
                        "config": dict(CONFIG)
                    },
                    V2_BEST
                )

                print(
                    "NEW BEST official mAP:",
                    100 * ap,
                    "%"
                )

            else:
                no_improve += 1

            print(
                f"Official COCO mAP={100*ap:.4f}% "
                f"(best={100*best_map:.4f}%, "
                f"epoch={best_epoch})"
            )

        history.append(result)

        # Resumable checkpoint for each epoch.
        save = {
            "arm": "A3_v2",
            "epoch": ep,
            "additional_exposures": ep * per_epoch,
            "base_checkpoint": A3_CKPT,
            "manifest_sha256": manifest_sha,
            "dataset_ids_sha256": ids_sha,
            "gaussian_radius_mode": "v1_2_original",
            "neighbor_lambda": 0.25,
            "v2_epochs": N,

            "train_aug_cfg": {
                k: CONFIG[k]
                for k in (
                    "train_horizontal_flip_p",
                    "train_scale_min",
                    "train_scale_max",
                    "train_translate_frac",
                    "train_color_jitter",
                    "train_visible_box_min"
                )
            },

            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),

            "best_map": best_map,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "history": history,
            "config": dict(CONFIG)
        }

        v2_atomic_save(
            save,
            V2_LATEST
        )

        with open(
            V2_HISTORY,
            "w"
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
                allow_nan=False
            )

        if (
            no_improve
            >= CONFIG["v2_early_stop_evals"]
        ):
            print(
                "EARLY STOP: "
                "3 evaluation cycles "
                "without improvement."
            )
            break


# ============================================================
# 10. FINAL MODEL SELECTION
# ============================================================

if best_epoch > 0:
    assert os.path.isfile(V2_BEST), (
        "Best checkpoint metadata exists "
        "but best_model.pth is missing."
    )

SELECTED_A3_V2 = (
    V2_BEST
    if best_epoch > 0
    else A3_CKPT
)

print("=" * 70)
print("STEP 30 COMPLETE")
print("Selected:", SELECTED_A3_V2)
print("Selected mAP reference:", best_map)
print("=" * 70)

del net, opt, sch, val_v2

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
