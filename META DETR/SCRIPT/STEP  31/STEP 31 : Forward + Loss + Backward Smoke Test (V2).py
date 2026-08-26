# ==========================================================
# STEP 31 : Forward + Loss + Backward Smoke Test (V2)
# ==========================================================

import torch


print("=" * 70)
print(
    "STEP 31 : V2 BASE MODEL SMOKE TEST"
)
print("=" * 70)


batch = next(
    iter(train_loader)
)


support_images = (

    batch[
        "support_images"
    ]
    .to(
        CONFIG["device"]
    )
)


query_images = (

    batch[
        "query_images"
    ]
    .to(
        CONFIG["device"]
    )
)


targets = (
    move_targets_to_device(

        batch[
            "query_targets"
        ],

        CONFIG["device"]
    )
)


# ==========================================================
# eval() keeps BN statistics fixed.
# Gradient tetap aktif.
# ==========================================================

model.eval()


model.zero_grad(
    set_to_none=True
)


outputs = model(

    support_images,

    query_images
)


assert (
    outputs[
        "pred_logits"
    ].shape
    ==
    (
        support_images.shape[0],

        CONFIG["num_queries"],

        1
    )
)


loss_dict = criterion(

    outputs,

    targets
)


loss = (
    loss_dict[
        "loss_total"
    ]
)


assert torch.isfinite(
    loss
), (
    "Loss contains NaN/Inf."
)


loss.backward()


# ==========================================================
# Finite gradients
# ==========================================================

gradient_parameters = 0


for parameter in (
    model.parameters()
):

    if (
        parameter.grad
        is not None
    ):

        if not torch.isfinite(
            parameter.grad
        ).all():

            raise RuntimeError(
                "Non-finite gradient detected."
            )


        gradient_parameters += 1


assert (
    gradient_parameters
    >
    0
)


# ==========================================================
# Head gradients
# ==========================================================

class_grad = (

    model
    .detection_head
    .class_head
    .weight
    .grad
)


box_grad = (

    model
    .detection_head
    .box_head
    .layers[-1]
    .weight
    .grad
)


assert (
    class_grad
    is not None
)


assert (
    box_grad
    is not None
)


assert torch.isfinite(
    class_grad
).all()


assert torch.isfinite(
    box_grad
).all()


assert (
    class_grad
    .abs()
    .sum()
    .item()
    >
    0.0
)


assert (
    box_grad
    .abs()
    .sum()
    .item()
    >
    0.0
)


print(
    "Total Loss        :",
    loss.item()
)


print(
    "Classification    :",
    loss_dict[
        "loss_cls"
    ].item()
)


print(
    "BBox              :",
    loss_dict[
        "loss_bbox"
    ].item()
)


print(
    "GIoU              :",
    loss_dict[
        "loss_giou"
    ].item()
)


print(
    "Parameters w/ Grad:",
    gradient_parameters
)


print(
    "Class Head Grad   :",
    class_grad
    .abs()
    .mean()
    .item()
)


print(
    "Box Head Grad     :",
    box_grad
    .abs()
    .mean()
    .item()
)


# Clear, NO optimizer step
model.zero_grad(
    set_to_none=True
)


print(
    "Optimizer Step     : NO"
)


print(
    "Scheduler Step     : NO"
)


print("=" * 70)
print("✓ STEP 31 PASSED")
print("=" * 70)
