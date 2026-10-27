
# ============================================================
# STEP 30 — B2 C3/C4 GATED FUSION WARM-START PILOT
# A2 160K -> B2 +16K
# ============================================================

import os
import json
import hashlib
import random
import time
import gc
import numpy as np
import torch

from torch.utils.data import Sampler
from collections import Counter
from google.colab import drive

drive.mount(
    "/content/drive",
    force_remount=False
)

A2_ROOT = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8"
)

A2_CKPT = os.path.join(
    A2_ROOT, "epoch20_160k.pth"
)

A2_MANIFEST = os.path.join(
    A2_ROOT, "balanced160k_manifest.jsonl"
)

PILOT_ROOT = os.path.join(
    "/content/drive/MyDrive",
    CONFIG["experiment_name"]
)

B2_ARM = "B2_C3C4_gated"
B2_DIR = os.path.join(PILOT_ROOT, B2_ARM)

os.makedirs(B2_DIR, exist_ok=True)

A2_CONTROL_CKPT = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_A3_R1_PILOT_FROM_A2_160K/"
    "A2_control/epoch2_add16k.pth"
)

LOCKED_SHA = (
    "9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f"
)

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"
assert CONFIG["a3_neighbor_lambda"] == 0.0

assert not train_dataset.augment
assert not val_dataset.augment

assert os.path.isfile(A2_CKPT)
assert os.path.isfile(A2_MANIFEST)

assert os.path.isfile(A2_CONTROL_CKPT), (
    "A2 continuation control is required."
)

with open(A2_MANIFEST, "rb") as f:
    raw_manifest = f.read()

assert (
    hashlib.sha256(raw_manifest).hexdigest()
    == LOCKED_SHA
)

rows = [
    json.loads(line)
    for line in raw_manifest.splitlines()
    if line
]

assert len(rows) == 160000

pilot_rows = rows[:16000]
image_ids = list(map(int, train_dataset.image_ids))

for epoch in (1, 2):
    one = pilot_rows[
        (epoch - 1) * 8000:epoch * 8000
    ]

    assert len(one) == 8000

    assert set(
        Counter(
            x["selected_category_id"]
            for x in one
        ).values()
    ) == {100}

    assert all(
        int(x["epoch"]) == epoch
        and int(x["position"]) == j
        and image_ids[int(x["dataset_index"])]
        == int(x["image_id"])
        for j, x in enumerate(one)
    )

old = torch.load(
    A2_CKPT,
    map_location="cpu",
    weights_only=False
)

assert int(old["epoch"]) == 20
assert old["config"]["gaussian_radius_mode"] == "v1_2_original"
assert old["config"]["wh_loss_type"] == "hybrid_abs_relative"
assert old["source_manifest_sha256"] == LOCKED_SHA

control_meta = torch.load(
    A2_CONTROL_CKPT,
    map_location="cpu",
    weights_only=False
)

assert control_meta["arm"] == "A2_control"
assert control_meta["additional_exposures"] == 16000
assert control_meta["base_source_sha256"] == LOCKED_SHA
assert control_meta["base_checkpoint"] == A2_CKPT
assert control_meta["neighbor_lambda"] == 0.0

assert float(
    control_meta["pilot_lr"]
) == float(CONFIG["b2_base_lr"])

del control_meta


def b2_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class B2PilotSampler(Sampler):
    def __init__(self):
        self.epoch = 1

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return 8000

    def __iter__(self):
        sl = pilot_rows[
            (self.epoch - 1) * 8000:
            self.epoch * 8000
        ]

        return iter(
            int(x["dataset_index"])
            for x in sl
        )


def b2_save(payload, path):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


rng_seed = int(SEED) + 3170
b2_seed(rng_seed)

sampler = B2PilotSampler()

train_pilot, train_gen = build_train_loader(
    rng_seed,
    sampler=sampler,
    encoded=True
)

assert len(train_pilot) == 1000

criterion.neighbor_lambda = 0.0

net = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

load_a2_into_b2(
    net,
    old["model_state_dict"]
)

# Verify exact zero-gate functional identity.
net.eval()

probe_images, _, _ = next(iter(train_pilot))

with torch.inference_mode():
    px = probe_images[:1].to(device)

    prediction = net(px)
    features = net.backbone(px)

    baseline_deconv = net.neck.deconv3(
        net.neck.deconv2(
            net.neck.deconv1(features["c5"])
        )
    )

    reference = net.heads(baseline_deconv)

    for k in ("heatmap", "wh", "offset"):
        assert torch.allclose(
            prediction[k],
            reference[k],
            rtol=1e-6,
            atol=1e-6
        ), k

print("A2 -> B2 checkpoint loading and zero-gate equivalence PASS")

del (
    probe_images,
    px,
    prediction,
    features,
    baseline_deconv,
    reference,
    old
)

b2_seed(rng_seed)

optimizer = build_b2_optimizer(net)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG["b2_pilot_epochs"],
    eta_min=CONFIG["b2_eta_min"]
)

latest = os.path.join(
    B2_DIR, "latest.pth"
)

history = []
start = 0

if os.path.isfile(latest):
    ck = torch.load(
        latest,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == B2_ARM
    assert ck["base_checkpoint"] == A2_CKPT
    assert ck["base_source_sha256"] == LOCKED_SHA
    assert ck["gaussian_radius_mode"] == "v1_2_original"
    assert ck["neighbor_lambda"] == 0.0
    assert ck["optimizer_base_lr"] == CONFIG["b2_base_lr"]
    assert ck["optimizer_lateral_lr"] == CONFIG["b2_lateral_lr"]

    start = int(ck["epoch"])
    assert start in (1, 2)

    net.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )
    optimizer.load_state_dict(
        ck["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        ck["scheduler_state_dict"]
    )

    history = ck["history"]

    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(
        ck["rng"]["torch_cpu"].cpu()
    )
    torch.cuda.set_rng_state_all(
        ck["rng"]["torch_cuda"]
    )

    del ck

for epoch in range(start + 1, 3):
    sampler.set_epoch(epoch)
    train_gen.manual_seed(rng_seed + epoch)

    t0 = time.perf_counter()

    stats = train_centernet_epoch(
        net,
        train_pilot,
        optimizer
    )

    assert stats["batches"] == 1000

    scheduler.step()

    print(
        "epoch", epoch,
        "loss", stats["loss"],
        "gates",
        float(net.neck.gate_c3.detach()),
        float(net.neck.gate_c4.detach())
    )

    history.append({
        "epoch": epoch,
        "additional_exposures": epoch * 8000,
        "train_stats": stats,
        "seconds": time.perf_counter() - t0
    })

    payload = {
        "arm": B2_ARM,
        "epoch": epoch,
        "additional_exposures": epoch * 8000,
        "base_checkpoint": A2_CKPT,
        "base_source_sha256": LOCKED_SHA,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "gaussian_radius_mode": "v1_2_original",
        "neighbor_lambda": 0.0,
        "optimizer_base_lr": CONFIG["b2_base_lr"],
        "optimizer_lateral_lr": CONFIG["b2_lateral_lr"],
        "history": history,
        "config": dict(CONFIG),

        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
        }
    }

    b2_save(payload, latest)

    b2_save(
        payload,
        os.path.join(
            B2_DIR,
            f"epoch{epoch}_add{8 * epoch}k.pth"
        )
    )

assert os.path.isfile(
    os.path.join(
        B2_DIR, "epoch2_add16k.pth"
    )
)

print("STEP 30 PASS — B2 +16K complete")

del net, optimizer, scheduler, train_pilot

gc.collect()
torch.cuda.empty_cache()
