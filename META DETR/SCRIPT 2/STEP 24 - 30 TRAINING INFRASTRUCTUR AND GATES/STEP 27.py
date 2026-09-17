# ==========================================================
# STEP 27 — FULL REPLACEMENT
# Stage-1 Forward / Backward / Optimizer + Budget Smoke Test
# ==========================================================


# ==========================================================
# BUDGET SANITY
# ==========================================================

assert (
    optimizer_updates_for_episodes(
        800,
        4,
    )
    ==
    200
)


assert (
    STAGE1_UPDATES_PER_EPOCH
    ==
    200
)


assert (
    STAGE1_TOTAL_UPDATES
    ==
    5000
)


# ==========================================================
# MODEL / OPTIMIZER SMOKE TEST
# ==========================================================

smoke_model = (
    make_trial_model()
)


smoke_optimizer, _ = (
    build_optimizer_and_scheduler(

        smoke_model,

        stage='stage1',

        use_scheduler=False,
    )
)


smoke_batch = next(
    iter(
        train_loader
    )
)


prepare_model_for_training(
    smoke_model
)


backbone_parameter = next(

    parameter

    for parameter
    in smoke_model
    .backbone
    .parameters()

    if parameter.requires_grad
)


main_parameter = next(

    parameter

    for parameter
    in smoke_model
    .relation_module
    .parameters()

    if parameter.requires_grad
)


backbone_before = (
    backbone_parameter
    .detach()
    .clone()
)


main_before = (
    main_parameter
    .detach()
    .clone()
)


first_bn = next(

    module

    for module
    in smoke_model
    .backbone
    .modules()

    if isinstance(
        module,
        nn.BatchNorm2d
    )
)


bn_mean_before = (
    first_bn
    .running_mean
    .detach()
    .clone()
)


smoke_optimizer.zero_grad(
    set_to_none=True
)


smoke_losses = (
    compute_detection_training_loss(

        target_model=
            smoke_model,

        batch=
            smoke_batch,

        device=
            CONFIG[
                'device'
            ],
    )
)


smoke_losses[
    'loss_total'
].backward()


backbone_gradient = sum(

    parameter.grad
    .detach()
    .abs()
    .sum()
    .item()

    for parameter
    in smoke_model
    .backbone
    .parameters()

    if (
        parameter.requires_grad
        and
        parameter.grad
        is not None
    )
)


main_gradient = sum(

    parameter.grad
    .detach()
    .abs()
    .sum()
    .item()

    for parameter
    in smoke_model
    .relation_module
    .parameters()

    if (
        parameter.requires_grad
        and
        parameter.grad
        is not None
    )
)


assert backbone_gradient > 0.0
assert main_gradient > 0.0


torch.nn.utils.clip_grad_norm_(

    [
        parameter

        for parameter
        in smoke_model.parameters()

        if parameter.requires_grad
    ],

    max_norm=
        TRAIN_CONFIG[
            'stage1'
        ][
            'gradient_clip'
        ],

    error_if_nonfinite=True,
)


smoke_optimizer.step()


assert not torch.equal(
    backbone_before,
    backbone_parameter.detach(),
)


assert not torch.equal(
    main_before,
    main_parameter.detach(),
)


assert torch.equal(
    bn_mean_before,
    first_bn.running_mean,
)


print('=' * 70)
print('STEP 27 PASS : STAGE-1 SMOKE TEST')
print('=' * 70)

print(
    'Loss               :',
    float(
        smoke_losses[
            'loss_total'
        ].item()
    ),
)

print(
    'Backbone grad      :',
    backbone_gradient,
)

print(
    'Main grad          :',
    main_gradient,
)

print(
    'BN stats           : FROZEN'
)

print(
    'Episodes/epoch     :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'episodes_per_epoch'
    ],
)

print(
    'Updates/epoch      :',
    STAGE1_UPDATES_PER_EPOCH,
)

print(
    'Full total updates :',
    STAGE1_TOTAL_UPDATES,
)

print('=' * 70)


smoke_model.cpu()

del smoke_model
del smoke_optimizer
del smoke_batch
del smoke_losses

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
