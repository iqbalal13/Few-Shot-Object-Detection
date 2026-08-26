# ==========================================================
# STEP 28B : Criterion Sanity Test (V2)
# ==========================================================

import torch


model.eval()


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


query_targets = [

    {

        "boxes":

            target[
                "boxes"
            ].to(
                CONFIG["device"]
            ),


        "labels":

            target[
                "labels"
            ].to(
                CONFIG["device"]
            )
    }

    for target
    in batch[
        "query_targets"
    ]
]


with torch.no_grad():

    outputs = model(

        support_images,

        query_images
    )


    losses = criterion(

        outputs,

        query_targets
    )


print("=" * 70)
print(
    "STEP 28B : CRITERION SANITY TEST"
)
print("=" * 70)


print(
    "Pred logits shape:",
    outputs[
        "pred_logits"
    ].shape
)


print(
    "GT objects       :",
    sum(

        len(
            target[
                "boxes"
            ]
        )

        for target
        in query_targets
    )
)


assert (
    outputs[
        "pred_logits"
    ].shape[-1]
    ==
    1
)


for key in [

    "loss_cls",
    "loss_bbox",
    "loss_giou",
    "loss_total"
]:

    value = (
        losses[
            key
        ]
    )


    print(
        f"{key:15s}: "
        f"{value.item():.6f}"
    )


    assert torch.isfinite(
        value
    )


assert (
    losses[
        "loss_total"
    ].item()
    >=
    0.0
)


print("=" * 70)
print("✓ STEP 28B PASSED")
print("=" * 70)
