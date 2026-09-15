# ==========================================================
# STEP 23 : Matcher + Detection Loss Sanity
# ==========================================================

def move_targets_to_device(
    targets,
    device
):

    return [

        {
            key:
                (
                    value.to(
                        device
                    )
                    if torch.is_tensor(
                        value
                    )
                    else value
                )

            for key, value
            in target.items()
        }

        for target
        in targets
    ]


device = CONFIG[
    "device"
]


# ==========================================================
# SYNTHETIC MATCHER TEST
# ==========================================================

synthetic_outputs = {

    "pred_logits":
        torch.tensor(
            [
                [
                    [6.0],
                    [6.0],
                    [-6.0],
                ]
            ],
            device=device,
        ),

    "pred_boxes":
        torch.tensor(
            [
                [
                    [
                        0.2,
                        0.2,
                        0.1,
                        0.1,
                    ],
                    [
                        0.8,
                        0.8,
                        0.1,
                        0.1,
                    ],
                    [
                        0.5,
                        0.5,
                        0.3,
                        0.3,
                    ],
                ]
            ],
            device=device,
        ),
}


synthetic_targets = [

    {
        "boxes":
            synthetic_outputs[
                "pred_boxes"
            ][
                0,
                :2
            ].clone(),

        "labels":
            torch.zeros(
                2,
                dtype=torch.long,
                device=device,
            ),
    }
]


source_index, target_index = (
    matcher(
        synthetic_outputs,
        synthetic_targets
    )[0]
)


assert dict(
    zip(
        source_index.tolist(),
        target_index.tolist()
    )
) == {
    0: 0,
    1: 1,
}


# ==========================================================
# REAL EPISODIC BATCH TEST
# ==========================================================

real_batch = next(
    iter(
        train_loader
    )
)


assert validate_episodic_batch(
    real_batch
)


model.to(
    device
).eval()


with torch.inference_mode():

    targets = (
        move_targets_to_device(
            real_batch[
                "query_targets"
            ],
            device
        )
    )

    outputs = model(

        real_batch[
            "support_images"
        ].to(
            device
        ),

        real_batch[
            "query_images"
        ].to(
            device
        ),
    )

    losses = criterion(
        outputs,
        targets
    )

    assert all(
        torch.isfinite(
            value
        ).all()
        for value
        in losses.values()
    )


print(
    "STEP 23 PASS:",
    {
        name:
            round(
                value.item(),
                6
            )
        for name, value
        in losses.items()
    }
)


del synthetic_outputs
del synthetic_targets
del source_index
del target_index

del real_batch
del targets
del outputs
del losses

model.cpu()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
