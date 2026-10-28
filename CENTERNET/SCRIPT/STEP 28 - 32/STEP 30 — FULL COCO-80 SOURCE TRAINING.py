
# ============================================================
# STEP 30 — A3-v2 RESUMABLE CONTINUATION
# A3 176K -> MAX +96K = 272K exposures
# FIRST STAGE +32K = 208K exposures
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

drive.mount(
    "/content/drive",
    force_remount=False
)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

A3_CKPT = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_A3_R1_PILOT_FROM_A2_160K/"
    "A3_neighbor_R1/epoch2_add16k.pth"
)

V2_ROOT = os.path.join(
    "/content/drive/MyDrive",
    CONFIG["experiment_name"]
)

os.makedirs(V2_ROOT, exist_ok=True)

V2_LATEST = os.path.join(
    V2_ROOT, "latest.pth"
)

V2_BEST = os.path.join(
    V2_ROOT, "best_model.pth"
)

V2_HISTORY = os.path.join(
    V2_ROOT, "history.json"
)

V2_MANIFEST = os.path.join(
    V2_ROOT, "v2_mixed_manifest.jsonl"
)

# ------------------------------------------------------------
# 2. TRAINING BUDGET
# ------------------------------------------------------------

# Jangan ubah N saat resume.
# Untuk melanjutkan, ubah PAUSE_AFTER_EPOCH:
# 4 = +32K
# 8 = +64K
# 12 = +96K

N = int(CONFIG["v2_epochs"])
PAUSE_AFTER_EPOCH = 4

PER_EPOCH = int(
    CONFIG["v2_images_per_epoch"]
)

BATCH = int(
    CONFIG["batch_size"]
)

assert N == 12
assert 1 <= PAUSE_AFTER_EPOCH <= N
assert PER_EPOCH == 8000
assert BATCH == 8

assert CONFIG["v2_uniform_fraction"] == 0.5
assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert train_dataset.augment
assert not val_dataset.augment

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert CONFIG["a3_neighbor_radius"] == 1
assert CONFIG["a3_neighbor_lambda"] == 0.25

criterion.neighbor_lambda = 0.25

# ------------------------------------------------------------
# 3. VERIFY ORIGINAL A3 CHECKPOINT
# ------------------------------------------------------------

assert os.path.isfile(A3_CKPT), A3_CKPT

base = torch.load(
    A3_CKPT,
    map_location="cpu",
    weights_only=False
)

assert base["arm"] == "A3_neighbor_R1"
assert int(base["additional_exposures"]) == 16000
assert float(base["neighbor_lambda"]) == 0.25

assert (
    base["gaussian_radius_mode"]
    == "v1_2_original"
)

EXPECTED_A2_SOURCE_SHA = (
    "9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f"
)

assert (
    base["base_source_sha256"]
    == EXPECTED_A2_SOURCE_SHA
)

print("A3 source checkpoint verified.")

# ------------------------------------------------------------
# 4. DETERMINISTIC MIXED SAMPLING
# ------------------------------------------------------------

ids = list(
    map(int, train_dataset.image_ids)
)

ids_sha = hashlib.sha256(
    ",".join(
        map(str, ids)
    ).encode()
).hexdigest()

id_to_idx = {
    image_id: i
    for i, image_id in enumerate(ids)
}

pools = {
    cat: sorted({
        id_to_idx[int(image_id)]
        for image_id
        in train_dataset.coco.catToImgs[cat]
        if int(image_id) in id_to_idx
    })
    for cat in train_dataset.cat_ids
}

assert len(pools) == 80
assert all(pools.values())

uniform_per_epoch = int(
    PER_EPOCH * CONFIG["v2_uniform_fraction"]
)

balanced_per_epoch = (
    PER_EPOCH - uniform_per_epoch
)

assert balanced_per_epoch % 80 == 0
assert N * uniform_per_epoch <= len(train_dataset)

rng = random.Random(
    int(SEED) + 55331
)

uniform = list(
    range(len(train_dataset))
)

rng.shuffle(uniform)

manifest = []

for ep in range(1, N + 1):

    sample = [
        (idx, "uniform")
        for idx in uniform[
            (ep - 1) * uniform_per_epoch:
            ep * uniform_per_epoch
        ]
    ]

    for category in train_dataset.cat_ids:

        pool = pools[category]
        k = balanced_per_epoch // 80

        if len(pool) >= k:
            chosen = rng.sample(pool, k)
        else:
            chosen = rng.choices(pool, k=k)

        sample.extend(
            (idx, "balanced")
            for idx in chosen
        )

    rng.shuffle(sample)

    assert len(sample) == PER_EPOCH

    manifest.extend({
        "epoch": ep,
        "position": j,
        "dataset_index": int(idx),
        "image_id": ids[idx],
        "stream": stream
    } for j, (idx, stream) in enumerate(sample))

manifest_bytes = "".join(
    json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":")
    ) + "\n"
    for row in manifest
).encode()

manifest_sha = hashlib.sha256(
    manifest_bytes
).hexdigest()

# Manifest tidak boleh berubah saat resume.
if os.path.isfile(V2_MANIFEST):

    with open(V2_MANIFEST, "rb") as f:
        previous_sha = hashlib.sha256(
            f.read()
        ).hexdigest()

    assert previous_sha == manifest_sha, (
        "Training manifest changed!"
    )

else:
    with open(V2_MANIFEST, "wb") as f:
        f.write(manifest_bytes)

print("Manifest SHA256:", manifest_sha)

# ------------------------------------------------------------
# 5. SAMPLER / SEED / ATOMIC SAVE
# ------------------------------------------------------------

class FixedIndexSampler(Sampler):

    def __init__(self, indices):
        self.indices = list(
            map(int, indices)
        )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def set_epoch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


aug_keys = (
    "train_horizontal_flip_p",
    "train_scale_min",
    "train_scale_max",
    "train_translate_frac",
    "train_color_jitter",
    "train_visible_box_min"
)

aug_cfg = {
    k: CONFIG[k]
    for k in aug_keys
}

# ------------------------------------------------------------
# 6. LOAD A3 WEIGHTS STRICTLY
# ------------------------------------------------------------

set_epoch_seed(
    int(SEED) + 55332
)

net = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

net.load_state_dict(
    base["model_state_dict"],
    strict=True
)

assert not any(
    "gate_c" in k or "lateral_c" in k
    for k in net.state_dict()
)

del base

print("A3 weights loaded with strict=True.")

# ------------------------------------------------------------
# 7. OPTIMIZER AND 12-EPOCH SCHEDULER
# ------------------------------------------------------------

opt = torch.optim.AdamW(
    [
        {
            "params": list(
                net.backbone.parameters()
            ),
            "lr": CONFIG["v2_backbone_lr"]
        },
        {
            "params": (
                list(net.neck.parameters())
                + list(net.heads.parameters())
            ),
            "lr": CONFIG["v2_neck_head_lr"]
        }
    ],
    weight_decay=CONFIG["weight_decay"]
)

sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=N,
    eta_min=CONFIG["v2_eta_min"]
)

# ------------------------------------------------------------
# 8. CHECKPOINT RESUME
# ------------------------------------------------------------

start = 0
best_epoch = 0
no_improve = 0

best_map = float(
    CONFIG["v2_starting_map"]
)

history = []

if os.path.isfile(V2_LATEST):

    ck = torch.load(
        V2_LATEST,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == "A3_v2"
    assert ck["base_checkpoint"] == A3_CKPT

    assert ck["manifest_sha256"] == manifest_sha
    assert ck["dataset_ids_sha256"] == ids_sha
    assert ck["train_aug_cfg"] == aug_cfg

    assert ck["neighbor_lambda"] == 0.25
    assert ck["v2_epochs"] == N

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

    best_map = float(ck["best_map"])
    best_epoch = int(ck["best_epoch"])
    no_improve = int(ck["no_improve"])

    history = ck["history"]

    assert len(history) == start

    del ck

    print(
        "RESUME — epochs completed:",
        start
    )

# ------------------------------------------------------------
# 9. OFFICIAL VALIDATION LOADER
# ------------------------------------------------------------

val_v2, _ = build_val_loader(
    int(SEED) + 88001,
    encoded=False
)

assert len(val_v2) == 625

# ------------------------------------------------------------
# 10. TRAINING + COCOeval
# ------------------------------------------------------------

if no_improve >= CONFIG["v2_early_stop_evals"]:

    print(
        "Training already reached early stopping."
    )

else:

    for ep in range(
        start + 1,
        min(N, PAUSE_AFTER_EPOCH) + 1
    ):

        print(
            f"\nA3-v2 epoch {ep}/{N} "
            f"| total exposures {176 + ep * 8}K"
        )

        rows = manifest[
            (ep - 1) * PER_EPOCH:
            ep * PER_EPOCH
        ]

        assert len(rows) == PER_EPOCH
        assert all(
            row["epoch"] == ep
            for row in rows
        )

        indices = [
            row["dataset_index"]
            for row in rows
        ]

        train_v2, _ = build_train_loader(
            int(SEED) + 88001 + ep,
            sampler=FixedIndexSampler(indices),
            encoded=True
        )

        assert len(train_v2) == PER_EPOCH // BATCH

        set_epoch_seed(
            int(SEED) + 99001 + ep
        )

        t0 = time.perf_counter()

        stats = train_centernet_epoch(
            net,
            train_v2,
            opt
        )

        assert stats["batches"] == PER_EPOCH // BATCH

        del train_v2
        gc.collect()

        sch.step()

        entry = {
            "epoch": ep,
            "additional_exposures": ep * PER_EPOCH,
            "train": stats,
            "train_seconds": (
                time.perf_counter() - t0
            ),
            "lr_backbone": opt.param_groups[0]["lr"],
            "lr_neck_head": opt.param_groups[1]["lr"]
        }

        # COCOeval every +16K.
        if (
            ep % CONFIG["v2_eval_interval"] == 0
            or ep == N
        ):

            result = evaluate_centernet(
                net,
                val_v2,
                max_batches=None,
                compute_loss=False,
                compute_official_ap=True,
                score_threshold=0.5,
                iou_threshold=0.5
            )

            ap = float(
                result["official_map"]
            )

            assert np.isfinite(ap)

            entry["official_map"] = ap
            entry["official_ap50"] = (
                result["official_ap50"]
            )
            entry["official_ap75"] = (
                result["official_ap75"]
            )

            print(
                f"Epoch {ep} — "
                f"official mAP: {100 * ap:.4f}%"
            )

            if (
                ap
                > best_map + CONFIG["v2_min_improvement"]
            ):

                best_map = ap
                best_epoch = ep
                no_improve = 0

                save_atomic(
                    {
                        "arm": "A3_v2",
                        "epoch": ep,
                        "additional_exposures": ep * PER_EPOCH,
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

                print("NEW BEST SAVED:", V2_BEST)

            else:
                no_improve += 1

        history.append(entry)

        # Resumable checkpoint after each epoch.
        save_atomic(
            {
                "arm": "A3_v2",
                "epoch": ep,
                "additional_exposures": ep * PER_EPOCH,
                "base_checkpoint": A3_CKPT,
                "manifest_sha256": manifest_sha,
                "dataset_ids_sha256": ids_sha,
                "gaussian_radius_mode": "v1_2_original",
                "neighbor_lambda": 0.25,
                "v2_epochs": N,
                "train_aug_cfg": aug_cfg,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "best_map": best_map,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "history": history,
                "config": dict(CONFIG)
            },
            V2_LATEST
        )

        with open(V2_HISTORY, "w") as f:
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
                "EARLY STOP — 3 validation cycles "
                "without sufficient improvement."
            )
            break

# ------------------------------------------------------------
# 11. VERIFY BEST CHECKPOINT
# ------------------------------------------------------------

if best_epoch > 0:

    assert os.path.isfile(V2_BEST), (
        "Best checkpoint missing!"
    )

    bck = torch.load(
        V2_BEST,
        map_location="cpu",
        weights_only=False
    )

    assert bck["arm"] == "A3_v2"
    assert int(bck["epoch"]) == best_epoch
    assert bck["manifest_sha256"] == manifest_sha

    assert abs(
        float(bck["official_map"]) - best_map
    ) < 1e-9

    del bck

SELECTED_A3_V2 = (
    V2_BEST if best_epoch > 0 else A3_CKPT
)

print("=" * 65)
print("STEP 30 COMPLETE")
print("Selected:", SELECTED_A3_V2)
print("Best mAP reference:", 100 * best_map, "%")
print("Completed:", len(history), "/", N, "epochs")
print("=" * 65)

print(
    "To continue later, change PAUSE_AFTER_EPOCH "
    "to 8 or 12 and rerun STEP 30."
)

del net, opt, sch, val_v2
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
