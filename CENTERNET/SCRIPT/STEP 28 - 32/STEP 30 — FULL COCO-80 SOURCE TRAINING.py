# ============================================================
# STEP 30 — EXCLUSIVE SOURCE TRAINING / CONTINUATION V1.2
# 80K -> 120K -> 160K
# ALL OPTIMIZER UPDATES ARE HERE
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

drive.mount("/content/drive", force_remount=False)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert len(train_dataset) == 118287
assert len(val_dataset) == 5000
assert not train_dataset.augment
assert not val_dataset.augment

assert CONFIG["source_epochs"] == 20
assert CONFIG["source_images_total"] == 160000
assert CONFIG["train_horizontal_flip_p"] == 0.0
assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert not RUN_TINY_OVERFIT
assert not RUN_ENGINEERING_SANITY
assert CONFIG["max_objects"] == 128

EPOCHS = 20
IMAGES_PER_EPOCH = 8000
STEPS_PER_EPOCH = 1000
MANIFEST_SEED = int(SEED) + 3000

ROOT30 = "/content/drive/MyDrive"

BASE_DIR30 = os.path.join(
    ROOT30,
    "CenterNet_COCO80_Source_BALANCED80K_V1_2_BS8"
)

FULL_CHECKPOINT_DIR = os.path.join(
    ROOT30,
    "CenterNet_COCO80_V12_CONTINUE_160K"
)

os.makedirs(FULL_CHECKPOINT_DIR, exist_ok=True)

BASE_LAST30 = os.path.join(
    BASE_DIR30, "centernet_coco80_last.pth"
)

BASE_BEST30 = os.path.join(
    BASE_DIR30, "centernet_coco80_best.pth"
)

CONTINUE_LAST30 = os.path.join(
    FULL_CHECKPOINT_DIR, "continue_last.pth"
)

MILESTONE15_30 = os.path.join(
    FULL_CHECKPOINT_DIR, "epoch15_120k.pth"
)

MILESTONE20_30 = os.path.join(
    FULL_CHECKPOINT_DIR, "epoch20_160k.pth"
)

MANIFEST_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "continue_manifest_160k.jsonl"
)

HISTORY_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "history.json"
)

assert os.path.isfile(BASE_LAST30), BASE_LAST30
assert os.path.isfile(BASE_BEST30), BASE_BEST30


# ============================================================
# 1. VALID COCO CATEGORY POOL
# SAME LOGIC AS ORIGINAL V1.2 SOURCE TRAINING
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
        a for a in coco.imgToAnns.get(image_id, [])
        if not a.get("iscrowd", 0)
        and a["bbox"][2] > 1
        and a["bbox"][3] > 1
        and a["category_id"]
        in train_dataset.cat_id_to_contiguous
    ]

    if not annotations:
        return set()

    boxes = np.asarray(
        [
            [x, y, x + w, y + h]
            for a in annotations
            for x, y, w, h in [a["bbox"]]
        ],
        dtype=np.float32
    )

    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    boxes[:, [0, 2]] += pad_x
    boxes[:, [1, 3]] += pad_y

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, size - 1
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, size - 1
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

    dim = size // CONFIG["output_stride"]

    good &= (
        (cx >= 0)
        & (cx < dim)
        & (cy >= 0)
        & (cy < dim)
    )

    result = set()
    included = 0

    for ann, valid in zip(annotations, good):
        if not valid:
            continue

        result.add(ann["category_id"])
        included += 1

        if included >= CONFIG["max_objects"]:
            break

    return result


# ============================================================
# 2. DETERMINISTIC BALANCED MANIFEST
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
        for cat_id in valid_target_categories(image_id):
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
                candidate = next_image(cat_id)

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


class BalancedManifestSampler(Sampler):

    def __init__(self, manifest):
        self.by_epoch = {
            e: []
            for e in range(1, EPOCHS + 1)
        }

        for row in manifest:
            self.by_epoch[
                row["epoch"]
            ].append(
                row["dataset_index"]
            )

        assert all(
            len(v) == 8000
            for v in self.by_epoch.values()
        )

        self.epoch = 1

    def set_epoch(self, epoch):
        assert 1 <= epoch <= EPOCHS
        self.epoch = epoch

    def __iter__(self):
        return iter(
            self.by_epoch[self.epoch]
        )

    def __len__(self):
        return 8000


def atomic_torch_save(payload, path):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ============================================================
# 3. BUILD AND VERIFY MANIFEST
# ============================================================

print("Rebuilding deterministic 160K manifest...")

source_manifest = build_balanced80k_manifest()

assert len(source_manifest) == 160000

for e in range(1, 21):
    counts = Counter(
        r["selected_category_id"]
        for r in source_manifest
        if r["epoch"] == e
    )

    assert len(counts) == 80
    assert all(
        v == 100
        for v in counts.values()
    )


def serialize_manifest(rows):
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":")
            ) + "\n"
        ).encode("utf8")
        for row in rows
    )


manifest_bytes = serialize_manifest(
    source_manifest
)

source_manifest_sha256 = hashlib.sha256(
    manifest_bytes
).hexdigest()

prefix_sha30 = hashlib.sha256(
    serialize_manifest(
        source_manifest[:80000]
    )
).hexdigest()


# ============================================================
# 4. VERIFY ORIGINAL V1.2 BASE CHECKPOINT
# ============================================================

base30 = torch.load(
    BASE_LAST30,
    map_location="cpu",
    weights_only=False
)

assert base30["experiment_name"] == (
    "CenterNet_COCO80_FSOD_BALANCED80K_V1_2_BS8"
)

assert int(base30["epoch"]) == 10

assert int(
    base30["source_images_total"]
) == 80000

assert (
    base30["source_manifest_sha256"]
    == prefix_sha30
), "Original 80K manifest differs. STOP."

for key in [
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state"
]:
    assert key in base30


# ============================================================
# 5. VERIFY OR SAVE 160K MANIFEST
# ============================================================

if os.path.isfile(MANIFEST_FILE):

    with open(MANIFEST_FILE, "rb") as f:
        existing = f.read()

    assert existing == manifest_bytes, (
        "160K manifest differs. STOP."
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

print("80K SHA256 :", prefix_sha30)
print("160K SHA256:", source_manifest_sha256)

print(
    "Unique images:",
    len({
        r["image_id"]
        for r in source_manifest
    })
)


# ============================================================
# 6. SELECT CONTINUATION STATE
# ============================================================

resume30 = None

if os.path.isfile(CONTINUE_LAST30):

    resume30 = torch.load(
        CONTINUE_LAST30,
        map_location="cpu",
        weights_only=False
    )

    assert resume30["source_base"] == "V1.2"

    assert (
        resume30["manifest_sha256"]
        == source_manifest_sha256
    )

    assert 11 <= int(resume30["epoch"]) <= 20

    assert (
        len(resume30["history"])
        == int(resume30["epoch"]) - 10
    )


state30 = (
    resume30
    if resume30 is not None
    else base30
)

start_epoch30 = int(state30["epoch"])

print(
    "Starting from completed epoch:",
    start_epoch30
)


# ============================================================
# 7. TRAIN ONLY IF EPOCH 20 IS NOT COMPLETE
# ============================================================

if start_epoch30 < 20:

    model = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    model.load_state_dict(
        state30["model_state_dict"],
        strict=True
    )

    optimizer = build_source_optimizer(model)

    optimizer.load_state_dict(
        state30["optimizer_state_dict"]
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=25
        )
    )

    scheduler.load_state_dict(
        state30["scheduler_state_dict"]
    )

    assert scheduler.last_epoch == start_epoch30

    rs = state30["rng_state"]

    random.setstate(rs["python"])
    np.random.set_state(rs["numpy"])

    torch.set_rng_state(
        rs["torch_cpu"].cpu()
    )

    torch.cuda.set_rng_state_all(
        rs["torch_cuda"]
    )

    history = (
        list(resume30["history"])
        if resume30 is not None
        else []
    )

    source_sampler = BalancedManifestSampler(
        source_manifest
    )

    (
        full_train_loader,
        full_train_generator
    ) = build_train_loader(
        MANIFEST_SEED,
        sampler=source_sampler,
        encoded=True
    )

    (
        source_val_loader,
        source_val_generator
    ) = build_val_loader(
        SEED + 3301,
        encoded=True
    )

    assert len(full_train_loader) == STEPS_PER_EPOCH
    assert len(source_val_loader) == 625

    # Test real worker target encoding.
    source_sampler.set_epoch(
        start_epoch30 + 1
    )

    _, _, encoded_smoke30 = next(
        iter(full_train_loader)
    )

    assert encoded_smoke30["heatmap"].shape == (
        8, 80, 160, 160
    )

    del encoded_smoke30

    # ========================================================
    # 8. EXCLUSIVE TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch30 + 1,
        21
    ):

        source_sampler.set_epoch(epoch)

        full_train_generator.manual_seed(
            MANIFEST_SEED + epoch
        )

        print(
            f"\nSTEP 30 TRAIN epoch {epoch}/20"
            f" | exposures {epoch*8000:,}"
        )

        t0 = time.perf_counter()

        train_stats = train_centernet_epoch(
            model,
            full_train_loader,
            optimizer
        )

        assert train_stats["batches"] == STEPS_PER_EPOCH

        scheduler.step()

        record30 = {
            "epoch": epoch,
            "exposures": epoch * 8000,
            "train": train_stats,
            "train_seconds": (
                time.perf_counter() - t0
            ),
            "learning_rate_next": (
                optimizer.param_groups[0]["lr"]
            )
        }

        # ====================================================
        # 9. EVALUATE 120K AND 160K MILESTONES
        # ====================================================

        if epoch in (15, 20):

            val_stats = evaluate_centernet(
                model,
                source_val_loader,
                max_batches=None,
                score_threshold=0.50,
                iou_threshold=0.50,
                compute_official_ap=True
            )

            record30["validation"] = val_stats

            assert val_stats["official_map"] is not None

            milestone = (
                MILESTONE15_30
                if epoch == 15
                else MILESTONE20_30
            )

            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "source_base": "V1.2",
                    "source_images_total": epoch * 8000,
                    "manifest_sha256": (
                        source_manifest_sha256
                    ),
                    "val_stats": val_stats
                },
                milestone
            )

            print(
                "COCO mAP:",
                val_stats["official_map"]
            )

            print(
                "AP50:",
                val_stats["official_ap50"]
            )

            print(
                "AP75:",
                val_stats["official_ap75"]
            )

        # ====================================================
        # 10. SAVE RESUMABLE CHECKPOINT
        # ====================================================

        history.append(record30)

        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "epoch": epoch,
                "source_base": "V1.2",
                "manifest_sha256": (
                    source_manifest_sha256
                ),
                "history": history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all()
                    )
                }
            },
            CONTINUE_LAST30
        )

        with open(
            HISTORY_FILE + ".tmp",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                history, f, indent=2
            )

        os.replace(
            HISTORY_FILE + ".tmp",
            HISTORY_FILE
        )

        print(
            "Saved epoch", epoch,
            "| train_loss", train_stats["loss"]
        )

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

else:
    print(
        "160K checkpoint already finished."
    )
    print(
        "NO retraining needed."
    )


# ============================================================
# 11. FINAL CHECKPOINT VALIDATION
# ============================================================

assert os.path.isfile(CONTINUE_LAST30), (
    "Missing resumable checkpoint."
)

last30 = torch.load(
    CONTINUE_LAST30,
    map_location="cpu",
    weights_only=False
)

assert int(last30["epoch"]) == 20

assert (
    last30["manifest_sha256"]
    == source_manifest_sha256
)

assert os.path.isfile(MILESTONE15_30)
assert os.path.isfile(MILESTONE20_30)


# ============================================================
# 12. PICK BEST MILESTONE BY OFFICIAL mAP
# ============================================================

candidates30 = [
    MILESTONE15_30,
    MILESTONE20_30
]


def stored_map30(path):
    c = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert c["source_base"] == "V1.2"

    assert (
        c["manifest_sha256"]
        == source_manifest_sha256
    )

    return float(
        c["val_stats"]["official_map"]
    )


BEST_CHECKPOINT = max(
    candidates30,
    key=stored_map30
)

LAST_CHECKPOINT = CONTINUE_LAST30

best_meta30 = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

# The validation loader must exist even if training was skipped.
if "source_val_loader" not in globals():
    (
        source_val_loader,
        source_val_generator
    ) = build_val_loader(
        SEED + 3301,
        encoded=True
    )

print("\n" + "=" * 65)
print("STEP 30 COMPLETE")
print("=" * 65)

print("Completed epoch:", last30["epoch"])
print("Selected checkpoint:", BEST_CHECKPOINT)
print("Best checkpoint epoch:", best_meta30["epoch"])

print(
    "Best checkpoint mAP:",
    best_meta30["val_stats"]["official_map"]
)

print("Original checkpoints remain unchanged.")
