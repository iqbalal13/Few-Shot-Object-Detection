
# ============================================================
# STEP 30 — EXPERIMENT B1
# FRESH SOURCE TRAINING — COCO 80 CLASS
# GAUSSIAN V1.3 + MASKED L1 WH
# 160K EXPOSURES / 20 EPOCHS
# ============================================================

import os
import json
import hashlib
import random
import time
import math
import gc

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler
from google.colab import drive


# ============================================================
# 1. LOCK TRAINING PROTOCOL
# ============================================================

EPOCHS = 20
IMAGES_PER_EPOCH = 8000
STEPS_PER_EPOCH = 1000
EVAL_EPOCHS = (5, 10, 15, 20)
SCHEDULER_TMAX = 25

MANIFEST_SEED = (
    int(SEED)
    + int(CONFIG["source_manifest_seed_offset"])
)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

assert not train_dataset.augment
assert not val_dataset.augment

assert CONFIG["source_epochs"] == EPOCHS
assert CONFIG["source_images_total"] == 160000
assert CONFIG["source_images_per_epoch"] == IMAGES_PER_EPOCH
assert CONFIG["source_images_per_class_per_epoch"] == 100

assert CONFIG["num_classes"] == 80
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4
assert CONFIG["batch_size"] == 8
assert CONFIG["max_objects"] == 128

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"
assert CONFIG["gaussian_min_overlap"] == 0.7
assert CONFIG["train_horizontal_flip_p"] == 0.0

assert CONFIG["wh_loss_type"] == "masked_l1"

assert isinstance(criterion, CenterNetLoss)
assert criterion.wh_loss_type == "masked_l1"

assert criterion.hm_weight == 1.0
assert criterion.wh_weight == 0.1
assert criterion.offset_weight == 1.0

assert not RUN_TINY_OVERFIT
assert not RUN_ENGINEERING_SANITY

assert callable(train_centernet_epoch)
assert callable(evaluate_centernet)
assert callable(build_source_optimizer)

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_B1_GAUSSIAN_V13_FRESH_160K_BS8"
)

# Check active Gaussian formula.
assert int(
    gaussian_radius(
        40, 40,
        mode="v1_3_corrected"
    )
) == int(gaussian_radius(40, 40))

print("=" * 70)
print("STEP 30 — B1 TRAINING PROTOCOL")
print("=" * 70)
print("Experiment :", CONFIG["experiment_name"])
print("Gaussian   :", CONFIG["gaussian_radius_mode"])
print("WH loss    :", CONFIG["wh_loss_type"])
print("Epochs     :", EPOCHS)
print("Exposures  :", 160000)
print("Batch size :", CONFIG["batch_size"])


# ============================================================
# 2. B1 CHECKPOINT DIRECTORY
# ============================================================

drive.mount("/content/drive", force_remount=False)

ROOT30 = "/content/drive/MyDrive"

FULL_CHECKPOINT_DIR = os.path.join(
    ROOT30,
    CONFIG["experiment_name"]
)

os.makedirs(FULL_CHECKPOINT_DIR, exist_ok=True)

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
    "balanced160k_manifest.jsonl"
)

HISTORY_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "training_history.json"
)


def milestone30(epoch):
    assert epoch in EVAL_EPOCHS

    return os.path.join(
        FULL_CHECKPOINT_DIR,
        f"epoch{epoch}_{epoch * 8}k.pth"
    )


# ============================================================
# 3. CATEGORY VALIDATION
# UNCHANGED FROM ORIGINAL STEP 30
# ============================================================

def valid_target_categories(image_id):
    coco = train_dataset.coco
    info = coco.imgs[int(image_id)]

    sz = int(CONFIG["input_size"])

    scale = min(
        sz / info["width"],
        sz / info["height"]
    )

    px = (
        sz - round(info["width"] * scale)
    ) // 2

    py = (
        sz - round(info["height"] * scale)
    ) // 2

    anns = [
        a
        for a in coco.imgToAnns.get(image_id, [])
        if not a.get("iscrowd", 0)
        and a["bbox"][2] > 1
        and a["bbox"][3] > 1
        and a["category_id"]
        in train_dataset.cat_id_to_contiguous
    ]

    if not anns:
        return set()

    boxes = np.asarray(
        [
            [x, y, x + w, y + h]
            for a in anns
            for x, y, w, h in [a["bbox"]]
        ],
        dtype=np.float32
    )

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]] * scale + px,
        0,
        sz - 1
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]] * scale + py,
        0,
        sz - 1
    )

    good = (
        (boxes[:, 2] - boxes[:, 0] > 1)
        & (boxes[:, 3] - boxes[:, 1] > 1)
    )

    cx = (
        boxes[:, 0] + boxes[:, 2]
    ) / (2 * CONFIG["output_stride"])

    cy = (
        boxes[:, 1] + boxes[:, 3]
    ) / (2 * CONFIG["output_stride"])

    dim = sz // CONFIG["output_stride"]

    good &= (
        (cx >= 0)
        & (cx < dim)
        & (cy >= 0)
        & (cy < dim)
    )

    found = set()
    accepted = 0

    for ann, keep in zip(anns, good):
        if not keep:
            continue

        found.add(ann["category_id"])
        accepted += 1

        if accepted >= CONFIG["max_objects"]:
            break

    return found


# ============================================================
# 4. DETERMINISTIC BALANCED MANIFEST
# SAME ALGORITHM AS A2
# ============================================================

def build_balanced_manifest30():
    index_of = {
        int(image_id): i
        for i, image_id in enumerate(
            train_dataset.image_ids
        )
    }

    pools = {
        cat_id: []
        for cat_id in train_dataset.cat_ids
    }

    for image_id in train_dataset.image_ids:
        categories = valid_target_categories(image_id)

        for cat_id in categories:
            pools[cat_id].append(image_id)

    assert len(pools) == 80
    assert all(pools.values())

    streams = {}

    for cat_id, pool in pools.items():
        ids = sorted(pool)

        rng = random.Random(
            MANIFEST_SEED + int(cat_id) * 1000003
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
            stream["rng"].shuffle(stream["ids"])
            stream["cursor"] = 0

        selected = stream["ids"][
            stream["cursor"]
        ]

        stream["cursor"] += 1

        return selected

    rows = []

    for epoch in range(1, EPOCHS + 1):
        categories = [
            cat_id
            for cat_id in train_dataset.cat_ids
            for _ in range(
                CONFIG["source_images_per_class_per_epoch"]
            )
        ]

        random.Random(
            MANIFEST_SEED + 3000000 + epoch
        ).shuffle(categories)

        seen = set()

        for position, cat_id in enumerate(categories):
            selected = None

            for _ in range(
                len(streams[cat_id]["ids"])
            ):
                candidate = next_image(cat_id)

                if candidate not in seen:
                    selected = candidate
                    break

            if selected is None:
                selected = next_image(cat_id)

            seen.add(selected)

            rows.append({
                "epoch": epoch,
                "position": position,
                "selected_category_id": int(cat_id),
                "image_id": int(selected),
                "dataset_index": int(index_of[selected])
            })

    return rows


class BalancedManifestSampler(Sampler):
    def __init__(self, rows):
        self.by_epoch = {
            epoch: []
            for epoch in range(1, EPOCHS + 1)
        }

        for row in rows:
            self.by_epoch[row["epoch"]].append(
                row["dataset_index"]
            )

        assert all(
            len(indices) == IMAGES_PER_EPOCH
            for indices in self.by_epoch.values()
        )

        self.epoch = 1

    def set_epoch(self, epoch):
        assert 1 <= epoch <= EPOCHS
        self.epoch = epoch

    def __iter__(self):
        return iter(self.by_epoch[self.epoch])

    def __len__(self):
        return IMAGES_PER_EPOCH


# ============================================================
# 5. SERIALIZATION HELPERS
# ============================================================

def serialize30(rows):
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":")
            ) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def atomic_save30(obj, path):
    temporary = path + ".tmp"
    torch.save(obj, temporary)
    os.replace(temporary, path)


# ============================================================
# 6. BUILD AND VALIDATE 160K MANIFEST
# ============================================================

print("\nBuilding B1 balanced manifest...")

source_manifest = build_balanced_manifest30()

assert len(source_manifest) == 160000

for epoch in range(1, EPOCHS + 1):
    quota = Counter(
        row["selected_category_id"]
        for row in source_manifest
        if row["epoch"] == epoch
    )

    assert len(quota) == 80

    assert all(
        count == 100
        for count in quota.values()
    )

manifest_data = serialize30(source_manifest)

source_manifest_sha256 = hashlib.sha256(
    manifest_data
).hexdigest()

prefix80_sha = hashlib.sha256(
    serialize30(source_manifest[:80000])
).hexdigest()

EXPECTED_PREFIX80_SHA = (
    "f143cc86cd49fe97ba289ade2482387605358fdc8cdd6a7c197def20e31af079"
)

EXPECTED_160K_SHA = (
    "9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f"
)

assert prefix80_sha == EXPECTED_PREFIX80_SHA, (
    "First 80K manifest differs from V1.2"
)

assert source_manifest_sha256 == EXPECTED_160K_SHA, (
    "B1 manifest differs from the locked A2/A1 manifest"
)

if os.path.isfile(MANIFEST_FILE):
    with open(MANIFEST_FILE, "rb") as file:
        assert file.read() == manifest_data
else:
    temporary = MANIFEST_FILE + ".tmp"

    with open(temporary, "wb") as file:
        file.write(manifest_data)

    os.replace(temporary, MANIFEST_FILE)

print("Manifest SHA256:", source_manifest_sha256)
print(
    "Unique image IDs:",
    len({
        row["image_id"]
        for row in source_manifest
    })
)

print("Manifest verification: PASSED")


# ============================================================
# 7. STRICT B1 RESUME
# ============================================================

resume = None

if os.path.isfile(LAST_CHECKPOINT):
    resume = torch.load(
        LAST_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    assert resume["experiment_name"] == (
        CONFIG["experiment_name"]
    )

    assert resume["source_manifest_sha256"] == (
        source_manifest_sha256
    )

    assert resume["source_images_total"] == 160000
    assert 1 <= int(resume["epoch"]) <= EPOCHS

    assert len(resume["history"]) == int(
        resume["epoch"]
    )

    resume_keys = (
        "gaussian_radius_mode",
        "gaussian_min_overlap",
        "wh_loss_type",
        "hm_loss_weight",
        "wh_loss_weight",
        "offset_loss_weight",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "source_epochs",
        "source_images_per_epoch",
        "source_manifest_seed_offset",
        "input_size",
        "output_stride",
        "num_classes",
        "train_horizontal_flip_p"
    )

    for key in resume_keys:
        assert resume["config"][key] == CONFIG[key], (
            f"B1 resume mismatch: {key}"
        )

    assert (
        os.path.isfile(BEST_CHECKPOINT)
        or int(resume["epoch"]) < 5
    )

elif os.path.isfile(BEST_CHECKPOINT):
    raise RuntimeError(
        "B1 BEST checkpoint exists without LAST. "
        "Inspect the experiment folder first."
    )


# ============================================================
# 8. CREATE ENCODED LOADERS AND MODEL
# ============================================================

start_epoch = (
    int(resume["epoch"])
    if resume is not None
    else 0
)

if start_epoch == EPOCHS:
    print(
        "B1 training already completed. "
        "Skipping optimizer updates."
    )

else:
    sampler30 = BalancedManifestSampler(
        source_manifest
    )

    full_train_loader, full_train_generator = (
        build_train_loader(
            MANIFEST_SEED,
            sampler=sampler30,
            encoded=True
        )
    )

    source_val_loader, source_val_generator = (
        build_val_loader(
            SEED + 3301,
            encoded=True
        )
    )

    assert len(full_train_loader) == STEPS_PER_EPOCH
    assert len(source_val_loader) == 625

    sampler30.set_epoch(start_epoch + 1)

    # --------------------------------------------------------
    # WORKER TARGET CHECK
    # --------------------------------------------------------

    _, smoke_raw, smoke_encoded = next(
        iter(full_train_loader)
    )

    assert smoke_encoded["heatmap"].shape == (
        8, 80, 160, 160
    )

    expected_encoded = build_centernet_target(
        smoke_raw[0]
    )

    for key in (
        "heatmap",
        "wh",
        "offset",
        "indices",
        "mask"
    ):
        assert torch.equal(
            smoke_encoded[key][0],
            expected_encoded[key]
        ), (
            f"B1 worker target mismatch: {key}"
        )

    print("Encoded DataLoader target test: PASSED")

    del smoke_raw
    del smoke_encoded
    del expected_encoded

    # --------------------------------------------------------
    # FRESH RANDOM INITIALIZATION
    # --------------------------------------------------------

    if resume is None:
        random.seed(MANIFEST_SEED)
        np.random.seed(MANIFEST_SEED)
        torch.manual_seed(MANIFEST_SEED)
        torch.cuda.manual_seed_all(MANIFEST_SEED)

    model30 = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=(resume is None)
    ).to(device)

    optimizer30 = build_source_optimizer(
        model30
    )

    scheduler30 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer30,
        T_max=SCHEDULER_TMAX
    )

    history30 = []
    best_map30 = -1.0
    best_val_loss30 = float("inf")

    # --------------------------------------------------------
    # RESUME MODEL / OPTIMIZER / SCHEDULER / RNG
    # --------------------------------------------------------

    if resume is not None:
        model30.load_state_dict(
            resume["model_state_dict"],
            strict=True
        )

        optimizer30.load_state_dict(
            resume["optimizer_state_dict"]
        )

        scheduler30.load_state_dict(
            resume["scheduler_state_dict"]
        )

        assert scheduler30.last_epoch == start_epoch

        history30 = list(resume["history"])
        best_map30 = float(resume["best_map"])
        best_val_loss30 = float(
            resume["best_val_loss"]
        )

        rng = resume["rng_state"]

        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])

        torch.set_rng_state(
            rng["torch_cpu"].cpu()
        )

        torch.cuda.set_rng_state_all(
            rng["torch_cuda"]
        )

        print(
            "RESUMING B1 FROM EPOCH",
            start_epoch
        )

    else:
        print(
            "FRESH B1 FROM IMAGENET-PRETRAINED RESNET-101"
        )
        print("160K exposures / 20 epochs")


    # ========================================================
    # 9. TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch + 1,
        EPOCHS + 1
    ):
        sampler30.set_epoch(epoch)

        full_train_generator.manual_seed(
            MANIFEST_SEED + epoch
        )

        print(
            f"\nB1 EPOCH {epoch}/{EPOCHS} "
            f"| exposures={epoch * 8000:,}/160,000"
        )

        start_time = time.perf_counter()

        train_result = train_centernet_epoch(
            model30,
            full_train_loader,
            optimizer30
        )

        duration = (
            time.perf_counter() - start_time
        )

        assert train_result["batches"] == (
            STEPS_PER_EPOCH
        )

        validation = None
        selected = False

        # ----------------------------------------------------
        # OFFICIAL EVALUATION MILESTONES
        # ----------------------------------------------------

        if epoch in EVAL_EPOCHS:
            validation = evaluate_centernet(
                model30,
                source_val_loader,
                max_batches=None,
                score_threshold=CONFIG["score_threshold"],
                iou_threshold=CONFIG["iou_threshold"],
                compute_official_ap=True
            )

            current_map = float(
                validation["official_map"]
            )

            current_loss = float(
                validation["loss"]
            )

            assert math.isfinite(current_map)
            assert math.isfinite(current_loss)

            selected = (
                current_map > best_map30 + 1e-8
                or (
                    abs(current_map - best_map30) <= 1e-8
                    and current_loss < best_val_loss30
                )
            )

            print(
                f"COCO mAP={current_map:.6f} "
                f"| AP50={float(validation['official_ap50']):.6f} "
                f"| AP75={float(validation['official_ap75']):.6f}"
            )

        history30.append({
            "epoch": epoch,
            "exposures": epoch * IMAGES_PER_EPOCH,
            "train_stats": train_result,
            "val_stats": validation,
            "train_seconds": duration,
            "lr_before_scheduler_step": (
                optimizer30.param_groups[0]["lr"]
            )
        })

        # ----------------------------------------------------
        # CHECKPOINT SELECTION
        # ----------------------------------------------------

        if epoch in EVAL_EPOCHS:
            milestone_payload = {
                "model_state_dict": model30.state_dict(),
                "epoch": epoch,
                "experiment_name": CONFIG["experiment_name"],
                "source_manifest_sha256": (
                    source_manifest_sha256
                ),
                "source_images_total": 160000,
                "actual_images_seen_at_checkpoint": (
                    epoch * 8000
                ),
                "val_stats": validation,
                "config": dict(CONFIG)
            }

            atomic_save30(
                milestone_payload,
                milestone30(epoch)
            )

            if selected:
                best_map30 = current_map
                best_val_loss30 = current_loss

                atomic_save30(
                    milestone_payload,
                    BEST_CHECKPOINT
                )

                print(
                    "New B1 best checkpoint: epoch",
                    epoch
                )

        # ----------------------------------------------------
        # ADVANCE SCHEDULER
        # ----------------------------------------------------

        scheduler30.step()

        # ----------------------------------------------------
        # SAVE RESUMABLE LAST CHECKPOINT
        # ----------------------------------------------------

        last_payload = {
            "model_state_dict": model30.state_dict(),
            "optimizer_state_dict": (
                optimizer30.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler30.state_dict()
            ),
            "epoch": epoch,
            "experiment_name": CONFIG["experiment_name"],
            "source_manifest_sha256": (
                source_manifest_sha256
            ),
            "source_images_total": 160000,
            "best_map": best_map30,
            "best_val_loss": best_val_loss30,
            "history": history30,
            "config": dict(CONFIG),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
            }
        }

        atomic_save30(
            last_payload,
            LAST_CHECKPOINT
        )

        # ----------------------------------------------------
        # SAVE JSON HISTORY
        # ----------------------------------------------------

        with open(
            HISTORY_FILE + ".tmp",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                history30,
                file,
                indent=2,
                allow_nan=False
            )

        os.replace(
            HISTORY_FILE + ".tmp",
            HISTORY_FILE
        )

        print(
            f"Epoch {epoch} saved "
            f"| loss={train_result['loss']:.5f} "
            f"| {duration / 60:.2f} min"
        )

    del model30
    del optimizer30
    del scheduler30

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 10. VERIFY COMPLETION
# ============================================================

last30 = torch.load(
    LAST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert int(last30["epoch"]) == 20
assert len(last30["history"]) == 20

assert all(
    row["train_stats"]["batches"] == 1000
    for row in last30["history"]
)

assert last30["source_manifest_sha256"] == (
    source_manifest_sha256
)

assert last30["experiment_name"] == (
    CONFIG["experiment_name"]
)

assert last30["config"]["wh_loss_type"] == (
    "masked_l1"
)

assert last30["config"]["gaussian_radius_mode"] == (
    "v1_3_corrected"
)

assert os.path.isfile(BEST_CHECKPOINT)

for epoch in EVAL_EPOCHS:
    assert os.path.isfile(
        milestone30(epoch)
    )

if "source_val_loader" not in globals():
    source_val_loader, source_val_generator = (
        build_val_loader(
            SEED + 3301,
            encoded=True
        )
    )

print("=" * 70)
print("STEP 30 COMPLETE — EXPERIMENT B1")
print("=" * 70)

print("Epochs:", 20)
print("Exposures:", 160000)
print("Optimizer updates:", 20000)
print("Best mAP:", last30["best_map"])
print("Best checkpoint:", BEST_CHECKPOINT)
print("Fixed 160K:", milestone30(20))
print("STEP 30 COMPLETE")
