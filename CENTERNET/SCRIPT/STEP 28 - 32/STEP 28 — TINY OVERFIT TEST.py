# ============================================================
# STEP 28 — TINY OVERFIT TEST
# BF16 VERSION
# Fresh model + frozen backbone BN statistics
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

print("=" * 70)
print("STEP 28 — TINY OVERFIT TEST (BF16)")
print("=" * 70)


# ------------------------------------------------------------
# GPU / BF16 CHECK
# ------------------------------------------------------------

assert torch.cuda.is_available(), (
    "CUDA GPU tidak aktif. "
    "Gunakan Colab GPU runtime."
)

device = torch.device("cuda")

print("Device :", device)
print("GPU    :", torch.cuda.get_device_name(0))

BF16_SUPPORTED = torch.cuda.is_bf16_supported()

print("BF16 supported :", BF16_SUPPORTED)

assert BF16_SUPPORTED, (
    "GPU ini tidak mendukung BF16. "
    "A100 seharusnya mendukung BF16."
)


# ------------------------------------------------------------
# Freeze ONLY backbone BatchNorm running statistics
# ResNet convolution weights remain trainable
# ------------------------------------------------------------

def set_backbone_bn_eval(model):

    for module in model.backbone.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):
            module.eval()


# ------------------------------------------------------------
# Fresh source model builder
# ------------------------------------------------------------

def build_fresh_source_model():

    fresh_model = CenterNetResNet101(
        num_classes=CONFIG["num_classes"],
        pretrained_backbone=True
    ).to(device)

    return fresh_model


# ------------------------------------------------------------
# Source optimizer builder
# ------------------------------------------------------------

def build_source_optimizer(model):

    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )


# ------------------------------------------------------------
# Select 8 deterministic COCO images
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

TINY_STEPS = 300

GRAD_CLIP_NORM = 10.0

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


    if not torch.isfinite(
        images
    ).all():

        raise FloatingPointError(
            f"Non-finite input image "
            f"at tiny step {step}."
        )


    encoded_targets = \
        build_batch_centernet_targets(
            targets,
            device=device
        )


    model.train()

    set_backbone_bn_eval(
        model
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    # --------------------------------------------------------
    # BF16 model forward
    # --------------------------------------------------------

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        outputs = model(
            images
        )


    # --------------------------------------------------------
    # Check model outputs BEFORE loss
    # --------------------------------------------------------

    for name, tensor in outputs.items():

        if not torch.isfinite(
            tensor
        ).all():

            raise FloatingPointError(
                f"Non-finite model output "
                f"'{name}' at tiny step {step}."
            )


    # --------------------------------------------------------
    # Criterion from Step 24 computes loss in FP32
    # --------------------------------------------------------

    losses = criterion(
        outputs,
        encoded_targets
    )


    total_loss = losses[
        "loss_total"
    ]


    if not torch.isfinite(
        total_loss
    ):

        raise FloatingPointError(
            f"Non-finite tiny loss "
            f"at step {step}. "
            f"hm={float(losses['loss_heatmap'].detach()):.6f}, "
            f"wh={float(losses['loss_wh'].detach()):.6f}, "
            f"off={float(losses['loss_offset'].detach()):.6f}"
        )


    # --------------------------------------------------------
    # BF16 does NOT need GradScaler
    # --------------------------------------------------------

    total_loss.backward()


    grad_norm = \
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
            error_if_nonfinite=False
        )


    if not torch.isfinite(
        grad_norm
    ):

        raise FloatingPointError(
            f"Non-finite gradient "
            f"at tiny step {step}."
        )


    optimizer.step()


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
            f"loss = {loss_value:.4f} | "
            f"grad_norm = "
            f"{float(grad_norm.detach()):.4f}"
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


print("\nPrecision mode : BF16 forward + FP32 loss")
print("GradScaler     : NOT USED")
print("STEP 28 PASSED — MODEL CAN LEARN")
