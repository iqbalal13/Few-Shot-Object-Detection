# ==========================================================
# STEP 27 : Forward + Loss + Backward Smoke Test
#
# REAL COCO episode
#
# - correct support
# - absent wrong support
# - Hungarian matching
# - detection loss
# - matched-query ranking
# - backward
# - optimizer step
#
# Official `model` remains untouched.
# ==========================================================

print("=" * 70)
print("STEP 27 : FORWARD / LOSS / BACKWARD SMOKE TEST")
print("=" * 70)


smoke_model = copy.deepcopy(
    model

).to(
    CONFIG["device"]
)


(
    smoke_optimizer,
    _
) = build_optimizer_and_scheduler(
    smoke_model
)


batch = next(
    iter(train_loader)
)


smoke_model.train()


freeze_backbone_bn_statistics(
    smoke_model.backbone
)


result = (
    compute_combined_training_loss(

        target_model=
            smoke_model,

        batch=
            batch,

        dataset=
            train_dataset,

        epoch=
            0,

        step=
            0,

        device=
            CONFIG["device"]
    )
)


loss = (
    result[
        "combined_loss"
    ]
)


smoke_optimizer.zero_grad(
    set_to_none=True
)


loss.backward()


# ==========================================================
# GRADIENT STATS
# ==========================================================

def module_gradient_norm(
    module
):

    total_sq = 0.0


    for parameter in (
        module.parameters()
    ):

        if parameter.grad is None:
            continue


        if not torch.isfinite(
            parameter.grad
        ).all():

            raise RuntimeError(
                "Gradient contains NaN/Inf."
            )


        total_sq += float(

            parameter.grad
            .detach()
            .pow(2)
            .sum()
            .item()
        )


    return (
        total_sq ** 0.5
    )


backbone_grad = (
    module_gradient_norm(
        smoke_model.backbone
    )
)


relation_grad = (
    module_gradient_norm(
        smoke_model.relation_module
    )
)


head_grad = (
    module_gradient_norm(
        smoke_model.detection_head
    )
)


torch.nn.utils.clip_grad_norm_(

    smoke_model.parameters(),

    max_norm=
        TRAIN_CONFIG[
            "gradient_clip"
        ]
)


smoke_optimizer.step()


# ==========================================================
# ASSERTIONS
# ==========================================================

assert torch.isfinite(
    loss
)


assert backbone_grad > 0.0

assert relation_grad > 0.0

assert head_grad > 0.0


# ==========================================================
# REPORT
# ==========================================================

det = (
    result[
        "detection_losses"
    ]
)


stats = (
    result[
        "support_rank_stats"
    ]
)


print(
    f"Detection Loss : "
    f"{det['loss_total'].item():.6f}"
)

print(
    f"Support Rank   : "
    f"{result['support_rank_loss'].item():.6f}"
)

print(
    f"Combined Loss  : "
    f"{loss.item():.6f}"
)

print("-" * 70)

print(
    f"Correct Sim    : "
    f"{stats['correct_similarity']:.6f}"
)

print(
    f"Wrong Sim      : "
    f"{stats['wrong_similarity']:.6f}"
)

print(
    f"Observed Margin: "
    f"{stats['observed_margin']:.6f}"
)

print("-" * 70)

print(
    f"Backbone Grad  : "
    f"{backbone_grad:.6f}"
)

print(
    f"Relation Grad  : "
    f"{relation_grad:.6f}"
)

print(
    f"Head Grad      : "
    f"{head_grad:.6f}"
)


del smoke_model
del smoke_optimizer


if torch.cuda.is_available():

    torch.cuda.empty_cache()


print("=" * 70)
print("✓ STEP 27 SMOKE TEST PASSED")
print("✓ official model remains untouched")
print("=" * 70)
