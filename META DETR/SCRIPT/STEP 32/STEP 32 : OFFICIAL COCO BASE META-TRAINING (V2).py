 ==========================================================
# STEP 32 : OFFICIAL COCO BASE META-TRAINING (V2)
# ==========================================================

import torch

from tqdm import tqdm


device = (
    CONFIG["device"]
)


# ==========================================================
# Fresh optimizer for official model
# ==========================================================

optimizer, scheduler = (
    build_optimizer_and_scheduler(
        model
    )
)


history = []


best_map50 = (
    -1.0
)


best_val_loss = (
    float(
        "inf"
    )
)


best_epoch = (
    -1
)


print("=" * 70)

print(
    "OFFICIAL SIMPLIFIED META-DETR "
    "V2 COCO BASE TRAINING"
)

print("=" * 70)


print(
    "Epochs             :",
    TRAIN_CONFIG[
        "epochs"
    ]
)


print(
    "Steps / Epoch      :",
    TRAIN_CONFIG[
        "steps_per_epoch"
    ]
)


print(
    "Validation Episodes:",
    TRAIN_CONFIG[
        "validation_episodes"
    ]
)


print(
    "Main LR            :",
    TRAIN_CONFIG[
        "learning_rate"
    ]
)


print(
    "Backbone LR        :",
    TRAIN_CONFIG[
        "backbone_learning_rate"
    ]
)


print("=" * 70)


# ==========================================================
# Training
# ==========================================================

for epoch in range(
    TRAIN_CONFIG[
        "epochs"
    ]
):

    # ======================================================
    # Episode RNG changes each epoch
    # ======================================================

    train_dataset.set_epoch(
        epoch
    )


    model.train()


    freeze_backbone_bn_statistics(
        model.backbone
    )


    running_total = (
        0.0
    )


    running_cls = (
        0.0
    )


    running_bbox = (
        0.0
    )


    running_giou = (
        0.0
    )


    actual_steps = (
        0
    )


    current_main_lr = (

        optimizer
        .param_groups[
            1
        ][
            "lr"
        ]
    )


    current_backbone_lr = (

        optimizer
        .param_groups[
            0
        ][
            "lr"
        ]
    )


    progress_bar = tqdm(

        train_loader,

        desc=(

            f"Epoch "
            f"[{epoch+1}/"
            f"{TRAIN_CONFIG['epochs']}]"
        )
    )


    for batch in (
        progress_bar
    ):

        if (
            actual_steps
            >=
            TRAIN_CONFIG[
                "steps_per_epoch"
            ]
        ):

            break


        support_images = (

            batch[
                "support_images"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        query_images = (

            batch[
                "query_images"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        targets = (
            move_targets_to_device(

                batch[
                    "query_targets"
                ],

                device
            )
        )


        # ==============================================
        # Forward
        # ==============================================

        outputs = model(

            support_images,

            query_images
        )


        # ==============================================
        # Loss
        # ==============================================

        loss_dict = criterion(

            outputs,

            targets
        )


        loss = (
            loss_dict[
                "loss_total"
            ]
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(

                f"Non-finite loss "
                f"at epoch {epoch+1}, "
                f"step {actual_steps+1}: "
                f"{loss.item()}"
            )


        # ==============================================
        # Backprop
        # ==============================================

        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        optimizer.step()


        # ==============================================
        # Statistics
        # ==============================================

        running_total += (
            loss.item()
        )


        running_cls += (
            loss_dict[
                "loss_cls"
            ].item()
        )


        running_bbox += (
            loss_dict[
                "loss_bbox"
            ].item()
        )


        running_giou += (
            loss_dict[
                "loss_giou"
            ].item()
        )


        actual_steps += 1


        progress_bar.set_postfix({

            "Loss":
                f"{loss.item():.4f}",

            "Step":
                (
                    f"{actual_steps}/"
                    f"{TRAIN_CONFIG['steps_per_epoch']}"
                )
        })


    # ======================================================
    # Train metrics
    # ======================================================

    train_metrics = {

        "loss":

            running_total

            /
            actual_steps,


        "loss_cls":

            running_cls

            /
            actual_steps,


        "loss_bbox":

            running_bbox

            /
            actual_steps,


        "loss_giou":

            running_giou

            /
            actual_steps
    }


    # ======================================================
    # Validation
    # ======================================================

    print(
        "\nRunning fixed "
        "COCO validation..."
    )


    val_metrics = (
        evaluate_episodic_model(

            model=
                model,

            data_loader=
                val_loader,

            criterion=
                criterion,

            device=
                device,

            score_threshold=
                TRAIN_CONFIG[
                    "score_threshold"
                ],

            iou_threshold=
                TRAIN_CONFIG[
                    "iou_threshold"
                ],

            show_progress=
                True
        )
    )


    epoch_record = {

        "epoch":
            epoch + 1,

        "main_lr":
            current_main_lr,

        "backbone_lr":
            current_backbone_lr,

        "train_loss":
            train_metrics[
                "loss"
            ],

        "val_loss":
            val_metrics[
                "loss"
            ],

        "val_map50":
            val_metrics[
                "episodic_map50"
            ],

        "val_precision50":
            val_metrics[
                "precision50"
            ],

        "val_recall50":
            val_metrics[
                "recall50"
            ]
    }


    history.append(
        epoch_record
    )


    # ======================================================
    # Epoch summary
    # ======================================================

    print(
        "\n"
        +
        "=" * 70
    )


    print(

        f"EPOCH "
        f"{epoch+1}/"
        f"{TRAIN_CONFIG['epochs']}"
    )


    print("-" * 70)


    print(
        f"Train Loss     : "
        f"{train_metrics['loss']:.4f}"
    )


    print(
        f"Train CLS      : "
        f"{train_metrics['loss_cls']:.4f}"
    )


    print(
        f"Train BBox     : "
        f"{train_metrics['loss_bbox']:.4f}"
    )


    print(
        f"Train GIoU     : "
        f"{train_metrics['loss_giou']:.4f}"
    )


    print("-" * 70)


    print(
        f"Val Loss       : "
        f"{val_metrics['loss']:.4f}"
    )


    print(
        f"Episodic mAP50 : "
        f"{val_metrics['episodic_map50']:.4f}"
    )


    print(
        f"Precision@0.50 : "
        f"{val_metrics['precision50']:.4f}"
    )


    print(
        f"Recall@0.50    : "
        f"{val_metrics['recall50']:.4f}"
    )


    print("-" * 70)


    print(
        f"Main LR        : "
        f"{current_main_lr:.8f}"
    )


    print(
        f"Backbone LR    : "
        f"{current_backbone_lr:.8f}"
    )


    print("=" * 70)


    # ======================================================
    # Best checkpoint
    # ======================================================

    current_map50 = (
        val_metrics[
            "episodic_map50"
        ]
    )


    current_val_loss = (
        val_metrics[
            "loss"
        ]
    )


    is_better = (

        current_map50
        >
        best_map50

        or

        (
            abs(
                current_map50
                -
                best_map50
            )
            <
            1e-12

            and

            current_val_loss
            <
            best_val_loss
        )
    )


    if is_better:

        best_map50 = (
            current_map50
        )


        best_val_loss = (
            current_val_loss
        )


        best_epoch = (
            epoch + 1
        )


        torch.save(

            {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "config":
                    CONFIG,

                "coco_config":
                    COCO_CONFIG,

                "train_config":
                    TRAIN_CONFIG,

                "validation_metrics":
                    val_metrics,

                "history":
                    history
            },

            BEST_CHECKPOINT_PATH
        )


        print(
            "\n✓ NEW BEST BASE MODEL"
        )


        print(
            "  Epoch :",
            best_epoch
        )


        print(
            "  mAP50 :",
            f"{best_map50:.4f}"
        )


    # ======================================================
    # Scheduler
    # ======================================================

    scheduler.step()


    # ======================================================
    # Latest checkpoint
    # ======================================================

    torch.save(

        {

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "config":
                CONFIG,

            "coco_config":
                COCO_CONFIG,

            "train_config":
                TRAIN_CONFIG,

            "validation_metrics":
                val_metrics,

            "history":
                history,

            "best_epoch":
                best_epoch,

            "best_map50":
                best_map50
        },

        LATEST_CHECKPOINT_PATH
    )


print(
    "\n"
    +
    "=" * 70
)


print(
    "COCO BASE META-TRAINING "
    "V2 FINISHED"
)


print("=" * 70)


print(
    "Best Epoch :",
    best_epoch
)


print(
    "Best mAP50 :",
    f"{best_map50:.4f}"
)


print(
    "Best Model :",
    BEST_CHECKPOINT_PATH
)


print("=" * 70)
