# ==========================================================
# STEP 27: Person Forward / Backward / Optimizer Smoke Test
# ==========================================================

smoke_model = make_trial_model()

smoke_optimizer, smoke_scheduler = (
    build_optimizer_and_scheduler(
        smoke_model
    )
)

smoke_model.train()

freeze_backbone_bn_statistics(
    smoke_model.backbone
)

batch = next(
    iter(train_loader)
)

smoke_optimizer.zero_grad(
    set_to_none=True
)

losses = compute_detection_training_loss(
    smoke_model,
    batch,
    CONFIG["device"],
)

losses["loss_total"].backward()

for name, module in (
    ("backbone", smoke_model.backbone),
    ("support_encoder", smoke_model.support_encoder),
    ("relation", smoke_model.relation_module),
    ("head", smoke_model.detection_head),
):
    grads = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]

    assert grads, (
        f"No gradients found: {name}"
    )

    assert all(
        torch.isfinite(grad).all()
        for grad in grads
    ), name

    norm = sum(
        grad.detach().square().sum().item()
        for grad in grads
    ) ** 0.5

    assert norm > 0, name

    print(
        name,
        "gradient norm:",
        norm
    )

torch.nn.utils.clip_grad_norm_(
    smoke_model.parameters(),
    TRAIN_CONFIG["gradient_clip"],
    error_if_nonfinite=True,
)

smoke_optimizer.step()

print(
    "STEP 27 PASS:",
    {
        key: value.item()
        for key, value in losses.items()
    }
)

del smoke_model
del smoke_optimizer, smoke_scheduler
del losses, batch, module, grads

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
