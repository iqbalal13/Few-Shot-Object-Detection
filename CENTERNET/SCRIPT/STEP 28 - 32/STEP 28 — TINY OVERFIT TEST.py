# ============================================================
# STEP 28 — TINY OVERFIT TEST
# FAST-REPRO V1
#
# - Fresh source model
# - BF16 model forward
# - FP32 stable loss from Step 24
# - No GradScaler
# - Gradient clipping
# - Fixed deterministic 8-image subset
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset
)


print("=" * 70)
print("STEP 28 — TINY OVERFIT TEST — BF16")
print("=" * 70)


# ============================================================
# GPU / BF16
# ============================================================

assert torch.cuda.is_available(), (
    "CUDA GPU tidak aktif."
)


device = torch.device(
    "cuda"
)


assert torch.cuda.is_bf16_supported(), (
    "GPU tidak mendukung BF16."
)


print(
    "Device :",
    device
)

print(
    "GPU    :",
    torch.cuda.get_device_name(0)
)

print(
    "BF16   :",
    torch.cuda.is_bf16_supported()
)


# ============================================================
# BACKBONE BN POLICY
#
# Freeze running statistics only.
# Convolutional weights remain trainable.
# ============================================================

def set_backbone_bn_eval(
    model
):

    for module in \
        model.backbone.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


# ============================================================
# FRESH SOURCE MODEL
# ============================================================

def build_fresh_source_model():

    fresh_model = \
        CenterNetResNet101(

            num_classes=
                CONFIG[
                    "num_classes"
                ],

            pretrained_backbone=
                True

        ).to(
            device
        )


    return fresh_model


# ============================================================
# OPTIMIZER
# ============================================================

def build_source_optimizer(
    model
):

    return torch.optim.AdamW(

        model.parameters(),

        lr=
            CONFIG[
                "learning_rate"
            ],

        weight_decay=
            CONFIG[
                "weight_decay"
            ]
    )


# ============================================================
# SELECT FIXED 8-IMAGE SUBSET
# ============================================================

tiny_indices = []


for dataset_index, image_id \
    in enumerate(
        train_dataset.image_ids
    ):


    ann_ids = \
        train_dataset.coco.getAnnIds(

            imgIds=[
                image_id
            ],

            iscrowd=False
        )


    if (
        1
        <= len(ann_ids)
        <= 15
    ):

        tiny_indices.append(
            dataset_index
        )


    if len(
        tiny_indices
    ) == 8:

        break


assert len(
    tiny_indices
) == 8


tiny_dataset = Subset(
    train_dataset,
    tiny_indices
)


# ============================================================
# DETERMINISTIC TINY LOADER
# ============================================================

tiny_generator = \
    torch.Generator()


tiny_generator.manual_seed(
    SEED + 2800
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

    generator=
        tiny_generator,

    collate_fn=
        centernet_collate_fn
)


print(
    "Tiny images :",
    len(tiny_dataset)
)


# ============================================================
# FRESH MODEL / OPTIMIZER
# ============================================================

model = \
    build_fresh_source_model()


optimizer = \
    build_source_optimizer(
        model
    )


TINY_STEPS = 300


GRAD_CLIP_NORM = \
    CONFIG[
        "grad_clip_norm"
    ]


tiny_losses = []


tiny_iterator = iter(
    tiny_loader
)


# ============================================================
# TRAIN
# ============================================================

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
            f"Non-finite input "
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
    # BF16 MODEL FORWARD
    # --------------------------------------------------------

    with torch.amp.autocast(

        device_type="cuda",

        dtype=
            torch.bfloat16

    ):

        outputs = model(
            images
        )


    # --------------------------------------------------------
    # MODEL OUTPUT FINITE CHECK
    # --------------------------------------------------------

    for name, tensor \
        in outputs.items():

        if not torch.isfinite(
            tensor
        ).all():

            raise FloatingPointError(
                f"Non-finite output "
                f"'{name}' "
                f"at tiny step {step}."
            )


    # --------------------------------------------------------
    # FP32 STABLE LOSS — STEP 24
    # --------------------------------------------------------

    losses = criterion(
        outputs,
        encoded_targets
    )


    total_loss = \
        losses[
            "loss_total"
        ]


    if not torch.isfinite(
        total_loss
    ):

        raise FloatingPointError(

            f"Non-finite tiny loss "
            f"at step {step}. "

            f"hm="
            f"{float(losses['loss_heatmap'].detach()):.6f}, "

            f"wh="
            f"{float(losses['loss_wh'].detach()):.6f}, "

            f"off="
            f"{float(losses['loss_offset'].detach()):.6f}"
        )


    # --------------------------------------------------------
    # BF16 DOES NOT REQUIRE GRADSCALER
    # --------------------------------------------------------

    total_loss.backward()


    grad_norm = \
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=
                GRAD_CLIP_NORM,

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

            f"Step "
            f"{step:03d}/"
            f"{TINY_STEPS} | "

            f"loss="
            f"{loss_value:.4f} | "

            f"grad_norm="
            f"{float(grad_norm.detach()):.4f}"
        )


# ============================================================
# TINY-OVERFIT GATE
# ============================================================

first_window = np.mean(
    tiny_losses[
        :20
    ]
)


last_window = np.mean(
    tiny_losses[
        -20:
    ]
)


reduction = (
    1.0
    - last_window
    / first_window
)


print(
    "\n"
    + "=" * 70
)

print(
    "TINY OVERFIT RESULT"
)

print(
    "=" * 70
)


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
    "Tiny-overfit FAILED: "
    "loss reduction < 50%."
)


print(
    "\nForward precision : BF16"
)

print(
    "Loss precision    : FP32"
)

print(
    "GradScaler        : OFF"
)

print(
    "Gradient clip     :",
    GRAD_CLIP_NORM
)

print(
    "\nSTEP 28 PASSED"
)
