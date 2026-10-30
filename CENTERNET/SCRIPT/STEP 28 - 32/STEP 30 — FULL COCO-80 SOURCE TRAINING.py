
# ============================================================
# STEP 30 — A3-v2 PEAK-ALIGNED GEOMETRY
# STRICT MAXIMUM: 208K EXPOSURES
# A3 176K + SAME 4 x 8K AS PREVIOUS A3-v2
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
# 1. CHECKPOINT PATHS
# ------------------------------------------------------------

DRIVE = "/content/drive/MyDrive"

A3_CKPT = os.path.join(
    DRIVE,
    "CenterNet_COCO80_A3_R1_PILOT_FROM_A2_160K",
    "A3_neighbor_R1",
    "epoch2_add16k.pth"
)

OLD_V2_ROOT = os.path.join(
    DRIVE,
    CONFIG["experiment_name"]
)

OLD_V2_BEST = os.path.join(
    OLD_V2_ROOT, "best_model.pth"
)

OLD_V2_LATEST = os.path.join(
    OLD_V2_ROOT, "latest.pth"
)

OLD_V2_HISTORY = os.path.join(
    OLD_V2_ROOT, "history.json"
)

OLD_MANIFEST = os.path.join(
    OLD_V2_ROOT, "v2_mixed_manifest.jsonl"
)

# Separate folder. Never overwrite old A3-v2.
NEW_V2_ROOT = os.path.join(
    DRIVE,
    "CenterNet_COCO80_A3_V2_PEAKGEOM_MAX208K"
)

os.makedirs(
    NEW_V2_ROOT,
    exist_ok=True
)

NEW_V2_LATEST = os.path.join(
    NEW_V2_ROOT, "latest.pth"
)

NEW_V2_BEST = os.path.join(
    NEW_V2_ROOT, "best_model.pth"
)

NEW_V2_HISTORY = os.path.join(
    NEW_V2_ROOT, "history.json"
)

NEW_V2_MANIFEST = os.path.join(
    NEW_V2_ROOT, "v2_mixed_manifest.jsonl"
)

for path in (
    A3_CKPT,
    OLD_V2_BEST,
    OLD_V2_LATEST,
    OLD_MANIFEST
):
    assert os.path.isfile(path), (
        "Missing reference file: " + path
    )

# ------------------------------------------------------------
# 2. LOCK CONFIG
# ------------------------------------------------------------

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert train_dataset.augment
assert not val_dataset.augment

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert CONFIG["a3_neighbor_radius"] == 1
assert CONFIG["a3_neighbor_lambda"] == 0.25

assert CONFIG["v2_uniform_fraction"] == 0.5
assert CONFIG["v2_images_per_epoch"] == 8000
assert CONFIG["batch_size"] == 8

# Keep the SAME original cosine scheduler trajectory.
assert CONFIG["v2_epochs"] == 12

assert criterion.neighbor_lambda == 0.25
assert criterion.offset_neighbor_lambda == 0.25
assert int(CONFIG["eval_topk"]) == 300

MAX_NEW_EPOCH = 4

assert (
    176000 + MAX_NEW_EPOCH * 8000
) == 208000

# ------------------------------------------------------------
# 3. REUSE ORIGINAL A3-v2 MANIFEST
# ------------------------------------------------------------

with open(OLD_MANIFEST, "rb") as f:
    raw_manifest = f.read()

manifest_sha = hashlib.sha256(
    raw_manifest
).hexdigest()

rows = [
    json.loads(x)
    for x in raw_manifest.decode("utf-8").splitlines()
    if x
]

assert len(rows) == (
    CONFIG["v2_epochs"] * 8000
)

# Validate the four epochs actually being used.
for ep in range(1, 5):

    chunk = rows[
        (ep - 1) * 8000:
        ep * 8000
    ]

    assert len(chunk) == 8000

    assert all(
        x["epoch"] == ep
        and x["position"] == i
        for i, x in enumerate(chunk)
    )

    assert all(
        0 <= int(x["dataset_index"]) < len(train_dataset)
        for x in chunk
    )

    assert all(
        int(
            train_dataset.image_ids[x["dataset_index"]]
        ) == x["image_id"]
        for x in chunk
    )

# Store identical manifest in new experiment folder.
if os.path.isfile(NEW_V2_MANIFEST):

    with open(NEW_V2_MANIFEST, "rb") as f:
        assert f.read() == raw_manifest

else:

    with open(NEW_V2_MANIFEST, "wb") as f:
        f.write(raw_manifest)

ids_sha = hashlib.sha256(
    ",".join(
        map(str, train_dataset.image_ids)
    ).encode()
).hexdigest()

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

# Verify old A3-v2 really used this manifest/config.
old_latest = torch.load(
    OLD_V2_LATEST,
    map_location="cpu",
    weights_only=False
)

assert old_latest["manifest_sha256"] == manifest_sha
assert old_latest["dataset_ids_sha256"] == ids_sha
assert old_latest["train_aug_cfg"] == aug_cfg
assert int(old_latest["epoch"]) >= 4

assert old_latest["base_checkpoint"] == A3_CKPT

del old_latest

print("Original manifest verified.")
print("Manifest SHA256:", manifest_sha)

# ------------------------------------------------------------
# 4. SAMPLER / SEED / ATOMIC CHECKPOINT
# ------------------------------------------------------------

class FixedIndexSampler(Sampler):

    def __init__(self, indices):
        self.indices = list(map(int, indices))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def set_epoch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):

    temporary = path + ".tmp"

    torch.save(obj, temporary)
    os.replace(temporary, path)


# ------------------------------------------------------------
# 5. LOAD ORIGINAL A3 176K
# ------------------------------------------------------------

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

del base

assert not any(
    "gate_c" in k or "lateral_c" in k
    for k in net.state_dict()
)

print("Original A3 176K loaded — strict=True")

# ------------------------------------------------------------
# 6. OPTIMIZER AND COSINE SCHEDULER
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

# Keep original T_max=12, even though run stops at 4.
sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=12,
    eta_min=CONFIG["v2_eta_min"]
)

# ------------------------------------------------------------
# 7. RESUME NEW EXPERIMENT ONLY
# ------------------------------------------------------------

start = 0
best_epoch = 0
best_map = -1.0
history = []

if os.path.isfile(NEW_V2_LATEST):

    ck = torch.load(
        NEW_V2_LATEST,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == "A3_v2"
    assert ck["method_tag"] == "peak_aligned_geometry"

    assert ck["base_checkpoint"] == A3_CKPT
    assert ck["manifest_sha256"] == manifest_sha
    assert ck["dataset_ids_sha256"] == ids_sha
    assert ck["train_aug_cfg"] == aug_cfg

    assert ck["max_new_epoch"] == MAX_NEW_EPOCH
    assert ck["offset_neighbor_lambda"] == 0.25
    assert ck["eval_topk"] == 300

    start = int(ck["epoch"])

    assert 0 <= start <= 4

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
    assert len(history) == start

    best_epoch = int(ck["best_epoch"])
    best_map = float(ck["best_map"])

    del ck

    print(
        "Resume PEAK-GEOMETRY from epoch:",
        start
    )

# ------------------------------------------------------------
# 8. VALIDATION
# ------------------------------------------------------------

val_v2, _ = build_val_loader(
    int(SEED) + 88001,
    encoded=False
)

assert len(val_v2) == 625

# ------------------------------------------------------------
# 9. TRAIN 4 ADDITIONAL EPOCHS MAXIMUM
# ------------------------------------------------------------

for ep in range(
    start + 1,
    MAX_NEW_EPOCH + 1
):

    assert (
        176000 + ep * 8000 <= 208000
    )

    selected = rows[
        (ep - 1) * 8000:
        ep * 8000
    ]

    indices = [
        r["dataset_index"]
        for r in selected
    ]

    loader, _ = build_train_loader(
        int(SEED) + 88001 + ep,
        sampler=FixedIndexSampler(indices),
        encoded=True
    )

    assert len(loader) == 1000

    set_epoch_seed(
        int(SEED) + 99001 + ep
    )

    print(
        f"PEAK-GEOMETRY epoch {ep}/4 "
        f"| total {176 + 8 * ep}K"
    )

    t0 = time.perf_counter()

    stats = train_centernet_epoch(
        net,
        loader,
        opt
    )

    assert stats["batches"] == 1000

    del loader
    gc.collect()

    sch.step()

    entry = {
        "epoch": ep,
        "additional_exposures": ep * 8000,
        "train": stats,
        "train_seconds": time.perf_counter() - t0,
        "lr_backbone": opt.param_groups[0]["lr"],
        "lr_neck_head": opt.param_groups[1]["lr"]
    }

    # Official COCOeval at 192K and 208K.
    if ep % 2 == 0:

        ev = evaluate_centernet(
            net,
            val_v2,
            max_batches=None,
            compute_loss=False,
            compute_official_ap=True,
            score_threshold=0.5,
            iou_threshold=0.5
        )

        ap = float(ev["official_map"])

        assert np.isfinite(ap)

        entry.update({
            "official_map": ap,
            "official_ap50": float(ev["official_ap50"]),
            "official_ap75": float(ev["official_ap75"])
        })

        print(
            f"COCOeval @ {176 + 8 * ep}K: "
            f"mAP={100 * ap:.4f}%"
        )

        if ap > best_map:

            best_map = ap
            best_epoch = ep

            save_atomic(
                {
                    "arm": "A3_v2",
                    "method_tag": "peak_aligned_geometry",
                    "epoch": ep,
                    "additional_exposures": ep * 8000,
                    "base_checkpoint": A3_CKPT,
                    "manifest_sha256": manifest_sha,
                    "dataset_ids_sha256": ids_sha,

                    "gaussian_radius_mode": "v1_2_original",
                    "neighbor_lambda": 0.25,
                    "offset_neighbor_lambda": 0.25,
                    "eval_topk": 300,

                    "model_state_dict": net.state_dict(),
                    "official_map": ap,
                    "config": dict(CONFIG)
                },
                NEW_V2_BEST
            )

            print(
                "NEW PEAK-GEOMETRY BEST:",
                NEW_V2_BEST
            )

    history.append(entry)

    # Save resumable checkpoint every epoch.
    save_atomic(
        {
            "arm": "A3_v2",
            "method_tag": "peak_aligned_geometry",
            "epoch": ep,
            "additional_exposures": ep * 8000,
            "base_checkpoint": A3_CKPT,
            "manifest_sha256": manifest_sha,
            "dataset_ids_sha256": ids_sha,

            "gaussian_radius_mode": "v1_2_original",
            "neighbor_lambda": 0.25,
            "offset_neighbor_lambda": 0.25,
            "max_new_epoch": 4,
            "eval_topk": 300,
            "train_aug_cfg": aug_cfg,

            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),

            "best_epoch": best_epoch,
            "best_map": best_map,
            "history": history,
            "config": dict(CONFIG)
        },
        NEW_V2_LATEST
    )

    with open(
        NEW_V2_HISTORY, "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
            allow_nan=False
        )

# ------------------------------------------------------------
# 10. CHECKPOINT INTEGRITY
# ------------------------------------------------------------

assert os.path.isfile(NEW_V2_BEST), (
    "No new evaluated checkpoint found."
)

assert os.path.isfile(OLD_V2_BEST), (
    "Original A3-v2 checkpoint missing."
)

new_best_ck = torch.load(
    NEW_V2_BEST,
    map_location="cpu",
    weights_only=False
)

assert (
    new_best_ck["method_tag"]
    == "peak_aligned_geometry"
)

assert int(new_best_ck["epoch"]) in (2, 4)

print("=" * 65)
print("STEP 30 PASS")
print("Old and new checkpoints are separate.")
print("New:", NEW_V2_BEST)
print("Old:", OLD_V2_BEST)
print(
    "New best mAP K300:",
    100 * float(new_best_ck["official_map"]),
    "%"
)
print("=" * 65)

del new_best_ck, net, opt, sch, val_v2

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
