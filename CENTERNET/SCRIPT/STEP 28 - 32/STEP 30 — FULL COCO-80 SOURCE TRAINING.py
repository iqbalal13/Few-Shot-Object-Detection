
# ============================================================
# STEP 30 — CENTERNET-FSOD BALANCED80K V1.3 AUG+GAUSS
# FRESH SOURCE TRAINING: 10 epochs x 8000 images
# 100 selections/class/epoch, batch=8
# 1000 updates/epoch = 10000 updates total
# ============================================================

import os
import gc
import json
import math
import time
import random
import hashlib
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler
from google.colab import drive

# ============================================================
# 1. LOCKED PROTOCOL CHECKS
# ============================================================

assert torch.cuda.is_available(), "Select GPU in Colab Runtime."
assert torch.cuda.is_bf16_supported(), "BF16 GPU required."

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

assert CONFIG["source_epochs"] == 10
assert CONFIG["source_images_per_epoch"] == 8000
assert CONFIG["source_images_total"] == 80000
assert CONFIG["source_images_per_class_per_epoch"] == 100
assert CONFIG["num_classes"] == 80
assert CONFIG["batch_size"] == 8
assert CONFIG["max_objects"] == 128

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

EPOCHS = CONFIG["source_epochs"]
IMAGES_PER_EPOCH = CONFIG["source_images_per_epoch"]

STEPS_PER_EPOCH = (
    IMAGES_PER_EPOCH // CONFIG["batch_size"]
)

SCHEDULER_TMAX = 25  # held constant to compare against V1.2
MIN_GEOMETRY_DELTA = 1e-4

MANIFEST_SEED = (
    int(SEED)
    + int(CONFIG["source_manifest_seed_offset"])
)

# ============================================================
# 2. INDEPENDENT 80K EXPERIMENT DIRECTORY
# ============================================================

drive.mount(
    "/content/drive",
    force_remount=False
)

FULL_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_Source_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

os.makedirs(
    FULL_CHECKPOINT_DIR,
    exist_ok=True
)

BEST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_last.pth"
)

MANIFEST_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "balanced80k_v13_manifest.jsonl"
)

HISTORY_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "training_history.json"
)

# ============================================================
# 3. CATEGORY POOL — SAME RULES AS STEPS 10/11/16
# ============================================================

def valid_target_categories(image_id):

    coco = train_dataset.coco
    info = coco.imgs[image_id]

    size = int(CONFIG["input_size"])

    scale = min(
        size / info["width"],
        size / info["height"]
    )

    pad_x = (
        size - round(info["width"] * scale)
    ) // 2

    pad_y = (
        size - round(info["height"] * scale)
    ) // 2

    annotations = [
        ann
        for ann in coco.imgToAnns.get(image_id, [])
        if not ann.get("iscrowd", 0)
        and ann["bbox"][2] > 1
        and ann["bbox"][3] > 1
        and ann["category_id"]
        in train_dataset.cat_id_to_contiguous
    ]

    if not annotations:
        return set()

    boxes = np.asarray(
        [
            [x, y, x + w, y + h]
            for ann in annotations
            for x, y, w, h in [ann["bbox"]]
        ],
        dtype=np.float32
    )

    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    boxes[:, [0, 2]] += pad_x
    boxes[:, [1, 3]] += pad_y

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        size - 1
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        size - 1
    )

    valid = (
        (boxes[:, 2] - boxes[:, 0] > 1)
        & (boxes[:, 3] - boxes[:, 1] > 1)
    )

    centers_x = (
        (boxes[:, 0] + boxes[:, 2])
        / (2 * CONFIG["output_stride"])
    )

    centers_y = (
        (boxes[:, 1] + boxes[:, 3])
        / (2 * CONFIG["output_stride"])
    )

    out_size = (
        size // CONFIG["output_stride"]
    )

    valid &= (
        (centers_x >= 0)
        & (centers_x < out_size)
    )

    valid &= (
        (centers_y >= 0)
        & (centers_y < out_size)
    )

    result = set()
    included = 0

    for ann, good in zip(annotations, valid):

        if not good:
            continue

        result.add(
            ann["category_id"]
        )

        included += 1

        if included >= CONFIG["max_objects"]:
            break

    return result


# ============================================================
# 4. DETERMINISTIC BALANCED80K MANIFEST
# ============================================================

def build_balanced80k_manifest():

    image_to_index = {
        image_id: idx
        for idx, image_id in enumerate(
            train_dataset.image_ids
        )
    }

    pools = {
        cat_id: []
        for cat_id in train_dataset.cat_ids
    }

    for image_id in train_dataset.image_ids:

        categories = valid_target_categories(
            image_id
        )

        for cat_id in categories:
            pools[cat_id].append(image_id)

    assert len(pools) == 80

    assert all(
        len(pool) > 0
        for pool in pools.values()
    ), "Empty category pool detected."

    streams = {}

    for cat_id, pool in pools.items():

        ids = sorted(pool)

        rng = random.Random(
            MANIFEST_SEED
            + int(cat_id) * 1000003
        )

        rng.shuffle(ids)

        streams[cat_id] = {
            "ids": ids,
            "rng": rng,
            "cursor": 0
        }

    def next_image(cat_id):

        stream = streams[cat_id]

        if stream["cursor"] >= len(stream["ids"]):

            stream["rng"].shuffle(
                stream["ids"]
            )

            stream["cursor"] = 0

        image_id = stream["ids"][
            stream["cursor"]
        ]

        stream["cursor"] += 1

        return image_id

    manifest = []

    for epoch in range(1, EPOCHS + 1):

        categories = [
            cat_id
            for cat_id in train_dataset.cat_ids
            for _ in range(
                CONFIG[
                    "source_images_per_class_per_epoch"
                ]
            )
        ]

        random.Random(
            MANIFEST_SEED + 3000000 + epoch
        ).shuffle(categories)

        used_in_epoch = set()

        for position, cat_id in enumerate(
            categories
        ):

            chosen = None

            for _ in range(
                len(streams[cat_id]["ids"])
            ):

                candidate = next_image(
                    cat_id
                )

                if candidate not in used_in_epoch:
                    chosen = candidate
                    break

            if chosen is None:
                chosen = next_image(cat_id)

            used_in_epoch.add(chosen)

            manifest.append({
                "epoch": epoch,
                "position": position,
                "selected_category_id": int(cat_id),
                "image_id": int(chosen),
                "dataset_index": int(
                    image_to_index[chosen]
                )
            })

    return manifest


# ============================================================
# 5. EPOCH-SPECIFIC SAMPLER
# ============================================================

class BalancedManifestSampler(Sampler):

    def __init__(self, manifest):

        self.indices_by_epoch = {
            epoch: [
                row["dataset_index"]
                for row in manifest
                if row["epoch"] == epoch
            ]
            for epoch in range(
                1,
                EPOCHS + 1
            )
        }

        assert all(
            len(indices) == IMAGES_PER_EPOCH
            for indices
            in self.indices_by_epoch.values()
        )

        self.epoch = 1

    def set_epoch(self, epoch):

        assert 1 <= epoch <= EPOCHS

        self.epoch = epoch

    def __iter__(self):

        return iter(
            self.indices_by_epoch[self.epoch]
        )

    def __len__(self):

        return len(
            self.indices_by_epoch[self.epoch]
        )


# ============================================================
# 6. BUILD AND VALIDATE MANIFEST
# ============================================================

print(
    "Building Balanced80K manifest "
    "from COCO train2017..."
)

source_manifest = build_balanced80k_manifest()

assert len(source_manifest) == (
    CONFIG["source_images_total"]
)

for epoch in range(1, EPOCHS + 1):

    counts = Counter(
        row["selected_category_id"]
        for row in source_manifest
        if row["epoch"] == epoch
    )

    assert len(counts) == 80

    assert all(
        counts[cat_id] == 100
        for cat_id in train_dataset.cat_ids
    )

    assert sum(counts.values()) == IMAGES_PER_EPOCH


manifest_bytes = b"".join(
    (
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":")
        ) + "\n"
    ).encode("utf-8")
    for row in source_manifest
)

source_manifest_sha256 = hashlib.sha256(
    manifest_bytes
).hexdigest()

if os.path.isfile(MANIFEST_FILE):

    with open(MANIFEST_FILE, "rb") as f:

        if f.read() != manifest_bytes:
            raise RuntimeError(
                "Manifest mismatch: use a separate "
                "experiment directory."
            )

else:

    with open(
        MANIFEST_FILE + ".tmp",
        "wb"
    ) as f:
        f.write(manifest_bytes)

    os.replace(
        MANIFEST_FILE + ".tmp",
        MANIFEST_FILE
    )

print(
    "Manifest SHA256:",
    source_manifest_sha256
)

print(
    "Image exposures:",
    len(source_manifest)
)

print(
    "Unique image IDs:",
    len({
        row["image_id"]
        for row in source_manifest
    })
)

print(
    "Optimizer updates:",
    EPOCHS * STEPS_PER_EPOCH
)


# ============================================================
# 7. FRESH MODEL INITIALIZATION
# INTERNAL RESUME ONLY FOR INTERRUPTED 80K RUNS
# ============================================================

resume = None

if os.path.isfile(LAST_CHECKPOINT):

    resume = torch.load(
        LAST_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    assert resume.get(
        "experiment_name"
    ) == CONFIG["experiment_name"]

    assert resume.get(
        "source_manifest_sha256"
    ) == source_manifest_sha256

    assert resume.get(
        "source_images_total"
    ) == 80000

    assert (
        1 <= int(resume.get("epoch", -1))
        <= EPOCHS
    )

    assert len(resume.get(
        "history", []
    )) == int(resume["epoch"])

    assert "rng_state" in resume

    assert os.path.isfile(BEST_CHECKPOINT), (
        "Last checkpoint exists but best is missing."
    )

else:

    if os.path.isfile(BEST_CHECKPOINT):
        raise RuntimeError(
            "Best exists but last is missing; "
            "inspect experiment folder."
        )

    random.seed(MANIFEST_SEED)
    np.random.seed(MANIFEST_SEED)

    torch.manual_seed(MANIFEST_SEED)
    torch.cuda.manual_seed_all(MANIFEST_SEED)


model = CenterNetResNet101(
    num_classes=CONFIG["num_classes"],
    pretrained_backbone=(resume is None)
).to(device)

optimizer = build_source_optimizer(
    model
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=SCHEDULER_TMAX
)

start_epoch = 0
best_geometry = -1.0
best_ap50 = -1.0
best_val_loss = float("inf")
history = []

if resume is not None:

    model.load_state_dict(
        resume["model_state_dict"]
    )

    optimizer.load_state_dict(
        resume["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        resume["scheduler_state_dict"]
    )

    start_epoch = int(
        resume["epoch"]
    )

    best_geometry = float(resume["best_geometry"])
    best_ap50 = float(resume["best_ap50"])

    best_val_loss = float(
        resume["best_val_loss"]
    )

    history = list(
        resume["history"]
    )

    rs = resume["rng_state"]

    random.setstate(
        rs["python"]
    )

    np.random.set_state(
        rs["numpy"]
    )

    torch.set_rng_state(
        rs["torch_cpu"].cpu()
    )

    torch.cuda.set_rng_state_all(
        rs["torch_cuda"]
    )

    print(
        "Resuming Balanced80K after epoch:",
        start_epoch
    )

    del resume

else:

    print(
        "Fresh 80K training from ImageNet "
        "pretrained backbone"
    )


# ============================================================
# 8. OPTIMIZED WORKER DATALOADERS
# ============================================================

if "train_loader" in globals():
    del train_loader

gc.collect()

source_sampler = BalancedManifestSampler(
    source_manifest
)

full_train_loader, full_train_generator = (
    build_train_loader(
        MANIFEST_SEED,
        sampler=source_sampler,
        encoded=True
    )
)

source_val_loader, source_val_generator = (
    build_val_loader(
        SEED + 3301,
        encoded=True
    )
)

assert STEPS_PER_EPOCH == 1000
assert len(full_train_loader) == 1000

assert len(source_val_loader) == math.ceil(
    5000 / CONFIG["batch_size"]
)

print(
    "Train batches per epoch:",
    len(full_train_loader)
)

print(
    "Validation batches:",
    len(source_val_loader)
)


# ============================================================
# 9. BF16 NUMERICAL HEALTH CHECK
# ============================================================

model.eval()

health_image, health_target = val_dataset[0]

with torch.inference_mode():

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        health_outputs = model(
            health_image.unsqueeze(0).to(
                device
            )
        )

    assert_outputs_finite(
        health_outputs,
        "Step30 80K BF16 health"
    )

    encoded_health = build_batch_centernet_targets(
        [health_target],
        device=device
    )

    health_loss = criterion(
        health_outputs,
        encoded_health
    )["loss_total"]

    assert bool(
        torch.isfinite(health_loss).item()
    ), "Health loss NaN/Inf"

del health_image
del health_target
del health_outputs
del encoded_health
del health_loss


# ============================================================
# 10. ATOMIC CHECKPOINT SAVE
# ============================================================

def atomic_torch_save(payload, path):

    tmp = path + ".tmp"

    torch.save(payload, tmp)

    os.replace(tmp, path)


# ============================================================
# 11. TRAINING — 10 EPOCHS
# NO EARLY STOPPING
# FULL COCO-VAL EACH EPOCH
# ============================================================

wall_start = time.perf_counter()

for epoch in range(
    start_epoch + 1,
    EPOCHS + 1
):

    source_sampler.set_epoch(
        epoch
    )

    full_train_generator.manual_seed(
        MANIFEST_SEED + epoch
    )

    print(
        f"\nBALANCED80K epoch {epoch}/{EPOCHS} "
        f"({STEPS_PER_EPOCH} steps)"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_stats = train_centernet_epoch(
        model,
        full_train_loader,
        optimizer,
        max_batches=None
    )

    torch.cuda.synchronize()

    train_seconds = (
        time.perf_counter() - t0
    )

    assert train_stats["batches"] == STEPS_PER_EPOCH

    # --------------------------------------------------------
    # FULL COCO VALIDATION
    # --------------------------------------------------------

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    val_stats = evaluate_centernet(
        model,
        source_val_loader,
        max_batches=None,
        score_threshold=CONFIG["score_threshold"],
        iou_threshold=CONFIG["iou_threshold"],
        compute_official_ap=True
    )

    torch.cuda.synchronize()

    val_seconds = (
        time.perf_counter() - t0
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT CRITERION
    # --------------------------------------------------------

    geometry = float(
        val_stats["geometry50"]
    )

    val_loss = float(
        val_stats["loss"]
    )

    official_ap50 = float(val_stats["official_ap50"])
    assert math.isfinite(official_ap50)
    # Selection metric: official COCO AP50 across all 80 classes.
    is_best = (
        official_ap50 > best_ap50 + 1e-6
        or (abs(official_ap50-best_ap50) <= 1e-6
            and val_loss < best_val_loss)
    )

    # --------------------------------------------------------
    # EPOCH STATISTICS
    # --------------------------------------------------------

    record = {
        "epoch": epoch,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_seconds": train_seconds,
        "val_seconds": val_seconds,
        "seconds_per_step": (
            train_seconds / STEPS_PER_EPOCH
        ),
        "learning_rate_before_scheduler_step": (
            optimizer.param_groups[0]["lr"]
        ),
        "exposures_seen": epoch * IMAGES_PER_EPOCH
    }

    history.append(record)

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"| train={train_seconds / 60:.2f}min "
        f"| val={val_seconds / 60:.2f}min "
        f"| {record['seconds_per_step']:.3f}s/step "
        f"| AP50={official_ap50:.5f} "
        f"| Geometry50={geometry:.5f} "
        f"| val_loss={val_loss:.5f}"
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    if is_best:

        best_geometry = geometry
        best_ap50 = official_ap50
        best_val_loss = val_loss

        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "actual_images_seen_at_checkpoint": (
                    epoch * IMAGES_PER_EPOCH
                ),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "config": dict(CONFIG),
                "experiment_name": CONFIG[
                    "experiment_name"
                ],
                "source_manifest_sha256": (
                    source_manifest_sha256
                ),
                "source_images_total": CONFIG[
                    "source_images_total"
                ],
                "precision_mode": "bfloat16"
            },
            BEST_CHECKPOINT
        )

        print(
            "Saved new best checkpoint at epoch",
            epoch
        )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step()

    # --------------------------------------------------------
    # LAST CHECKPOINT — INCLUDES RNG STATE
    # --------------------------------------------------------

    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_geometry": best_geometry,
            "best_ap50": best_ap50,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": dict(CONFIG),
            "experiment_name": CONFIG[
                "experiment_name"
            ],
            "source_manifest_sha256": (
                source_manifest_sha256
            ),
            "source_images_total": CONFIG[
                "source_images_total"
            ],
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
            }
        },
        LAST_CHECKPOINT
    )

    # --------------------------------------------------------
    # JSON TRAINING HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_FILE + ".tmp",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    os.replace(
        HISTORY_FILE + ".tmp",
        HISTORY_FILE
    )

    elapsed = (
        time.perf_counter() - wall_start
    ) / 60

    print(
        "Saved last checkpoint; "
        f"total Step30 elapsed: {elapsed:.2f} min"
    )


# ============================================================
# 12. FINAL STATUS
# ============================================================

assert len(history) == EPOCHS

assert history[-1]["exposures_seen"] == 80000

print(
    "\nSTEP 30 FINISHED: "
    "10 epochs / 80,000 exposures / 10,000 updates"
)

print("Best AP50:",best_ap50)
print("Geometry50 at best checkpoint:",best_geometry)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Last checkpoint:",
    LAST_CHECKPOINT
)

print(
    "Manifest SHA256:",
    source_manifest_sha256
)
