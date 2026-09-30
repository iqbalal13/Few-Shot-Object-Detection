# ============================================================
# STEP 28 — TINY OVERFIT TEST
# Fresh model + backbone BN stats frozen
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

print("=" * 70)
print("STEP 28 — TINY OVERFIT TEST")
print("=" * 70)

# ------------------------------------------------------------
# GPU REQUIRED
# ------------------------------------------------------------

assert torch.cuda.is_available(), (
    "CUDA GPU tidak aktif. "
    "Ubah Colab Runtime ke GPU lalu Run All Step 1–27."
)

device = torch.device("cuda")

print("Device :", device)
print("GPU    :", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Freeze ONLY backbone BatchNorm running statistics
# Backbone convolution weights remain trainable.
# ------------------------------------------------------------

def set_backbone_bn_eval(model):

    for module in model.backbone.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):
            module.eval()


# ------------------------------------------------------------
# Fresh model builder
# ------------------------------------------------------------

def build_fresh_source_model():

    fresh_model = CenterNetResNet101(
        num_classes=CONFIG["num_classes"],
        pretrained_backbone=True
    ).to(device)

    return fresh_model


# ------------------------------------------------------------
# Optimizer builder
# ------------------------------------------------------------

def build_source_optimizer(model):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    return optimizer


# ------------------------------------------------------------
# Select 8 fixed COCO images containing annotations
# ------------------------------------------------------------

tiny_indices = []

for dataset_index, image_id in enumerate(
    train_dataset.image_ids
):

    ann_ids = train_dataset.coco.getAnnIds(
        imgIds=[image_id],
        iscrowd=False
    )

    if 1 <= len(ann_ids) <= 15:

        tiny_indices.append(
            dataset_index
        )

    if len(tiny_indices) == 8:
        break


assert len(tiny_indices) == 8

tiny_dataset = Subset(
    train_dataset,
    tiny_indices
)

tiny_loader = DataLoader(
    tiny_dataset,
    batch_size=min(
        CONFIG["batch_size"],
        len(tiny_dataset)
    ),
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=centernet_collate_fn
)

print("Tiny images :", len(tiny_dataset))


# ------------------------------------------------------------
# Fresh model
# ------------------------------------------------------------

model = build_fresh_source_model()

optimizer = build_source_optimizer(
    model
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=True
)

TINY_STEPS = 300

tiny_losses = []

tiny_iterator = iter(
    tiny_loader
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for step in range(
    1,
    TINY_STEPS + 1
):

    try:

        images, targets = next(
            tiny_iterator
        )

    except StopIteration:

        tiny_iterator = iter(
            tiny_loader
        )

        images, targets = next(
            tiny_iterator
        )


    images = images.to(
        device,
        non_blocking=True
    )

    encoded_targets = \
        build_batch_centernet_targets(
            targets,
            device=device
        )


    model.train()

    # model.train() activates every BN,
    # therefore set backbone BN back to eval.
    set_backbone_bn_eval(
        model
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        outputs = model(
            images
        )

        losses = criterion(
            outputs,
            encoded_targets
        )

        total_loss = losses[
            "loss_total"
        ]


    assert torch.isfinite(
        total_loss
    ), "Tiny-overfit loss became NaN/Inf."


    scaler.scale(
        total_loss
    ).backward()

    scaler.step(
        optimizer
    )

    scaler.update()


    loss_value = float(
        total_loss.detach()
    )

    tiny_losses.append(
        loss_value
    )


    if (
        step == 1
        or step % 25 == 0
    ):

        print(
            f"Step {step:03d}/{TINY_STEPS} | "
            f"loss = {loss_value:.4f}"
        )


# ------------------------------------------------------------
# Gate
# ------------------------------------------------------------

first_window = np.mean(
    tiny_losses[:20]
)

last_window = np.mean(
    tiny_losses[-20:]
)

reduction = (
    1.0
    - last_window / first_window
)

print("\n" + "=" * 70)
print("TINY OVERFIT RESULT")
print("=" * 70)

print(
    f"First-20 mean loss : "
    f"{first_window:.4f}"
)

print(
    f"Last-20 mean loss  : "
    f"{last_window:.4f}"
)

print(
    f"Loss reduction     : "
    f"{reduction * 100:.2f}%"
)


assert (
    last_window
    < first_window * 0.50
), (
    "Tiny-overfit gate FAILED: "
    "loss tidak turun minimal 50%."
)

print("\nSTEP 28 PASSED — MODEL CAN LEARN / OVERFIT TINY COCO SET")
