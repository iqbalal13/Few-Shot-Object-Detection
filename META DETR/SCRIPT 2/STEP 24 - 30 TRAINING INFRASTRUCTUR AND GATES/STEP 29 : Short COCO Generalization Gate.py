# ==========================================================
# STEP 29 : Short COCO Generalization Gate
#
# 5 epochs × 800 steps
# fixed unseen COCO-Val
#
# Starts from official fresh `model`.
# ==========================================================

print("=" * 70)
print("STEP 29 : SHORT COCO GENERALIZATION GATE")
print("=" * 70)


assert TINY_GATE_PASSED, (
    "STEP 28 must pass first."
)


# ==========================================================
# FRESH MODEL
# ==========================================================

short_model = copy.deepcopy(

    model

).to(
    CONFIG["device"]
)


(
    short_optimizer,
    short_scheduler
) = build_optimizer_and_scheduler(
    short_model
)


# ==========================================================
# BASELINE
# ==========================================================

print()
print(
    "Running fresh-model COCO-Val baseline..."
)


short_baseline_metrics = (
    evaluate_episodic_model(

        model=
            short_model,

        data_loader=
            val_loader,

        criterion=
            criterion,

        device=
            CONFIG["device"],

        score_threshold=
            TRAIN_CONFIG[
                "score_threshold"
            ],

        iou_threshold=
            TRAIN_CONFIG[
                "iou_threshold"
            ],

        show_progress=True
    )
)


print()
print("-" * 70)
print("SHORT BASELINE")
print("-" * 70)

print(
    f"Val Loss      : "
    f"{short_baseline_metrics['loss']:.6f}"
)

print(
    f"mAP50         : "
    f"{short_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"Mean Best IoU : "
    f"{short_baseline_metrics['mean_best_iou']:.6f}"
)

print(
    f"Loc R@0.50    : "
    f"{short_baseline_metrics['localization_recall50']:.6f}"
)

print("-" * 70)


# ==========================================================
# BEST STATE
# ==========================================================

short_history = []

short_best_state = None

short_best_epoch = -1

short_best_map50 = -1.0

short_best_val_loss = float(
    "inf"
)

short_best_metrics = None


# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(

    TRAIN_CONFIG[
        "short_epochs"
    ]
):

    # Different reproducible episodes each epoch.

    train_dataset.set_epoch(
        epoch
    )


    short_model.train()


    freeze_backbone_bn_statistics(
        short_model.backbone
    )


    running_detection = 0.0
    running_rank = 0.0
    running_combined = 0.0

    running_correct_sim = 0.0
    running_wrong_sim = 0.0
    running_margin = 0.0

    actual_steps = 0


    progress = tqdm(

        train_loader,

        desc=(
            f"SHORT "
            f"[{epoch+1}/"
            f"{TRAIN_CONFIG['short_epochs']}]"
        )
    )


    for batch in progress:

        if (
            actual_steps
            >=
            TRAIN_CONFIG[
                "short_steps_per_epoch"
            ]
        ):

            break


        result = (
            compute_combined_training_loss(

                target_model=
                    short_model,

                batch=
                    batch,

                dataset=
                    train_dataset,

                epoch=
                    epoch,

                step=
                    actual_steps,

                device=
                    CONFIG["device"]
            )
        )


        loss = (
            result[
                "combined_loss"
            ]
        )


        short_optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            short_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        short_optimizer.step()


        detection_value = float(

            result[
                "detection_losses"
            ][
                "loss_total"
            ].item()
        )


        rank_value = float(

            result[
                "support_rank_loss"
            ].item()
        )


        stats = (
            result[
                "support_rank_stats"
            ]
        )


        running_detection += (
            detection_value
        )

        running_rank += (
            rank_value
        )

        running_combined += float(
            loss.item()
        )

        running_correct_sim += (
            stats[
                "correct_similarity"
            ]
        )

        running_wrong_sim += (
            stats[
                "wrong_similarity"
            ]
        )

        running_margin += (
            stats[
                "observed_margin"
            ]
        )


        actual_steps += 1


        progress.set_postfix({

            "Det":
                f"{detection_value:.3f}",

            "Rank":
                f"{rank_value:.3f}",

            "Margin":
                (
                    f"{stats['observed_margin']:.3f}"
                ),

            "Step":
                (
                    f"{actual_steps}/"
                    f"{TRAIN_CONFIG['short_steps_per_epoch']}"
                )
        })


    if actual_steps == 0:

        raise RuntimeError(
            "Short training completed zero steps."
        )


    # ======================================================
    # FIXED UNSEEN COCO-VAL
    # ======================================================

    print()
    print(
        "Running fixed unseen COCO-Val..."
    )


    val_metrics = (
        evaluate_episodic_model(

            model=
                short_model,

            data_loader=
                val_loader,

            criterion=
                criterion,

            device=
                CONFIG["device"],

            score_threshold=
                TRAIN_CONFIG[
                    "score_threshold"
                ],

            iou_threshold=
                TRAIN_CONFIG[
                    "iou_threshold"
                ],

            show_progress=True
        )
    )


    record = {

        "epoch":
            epoch + 1,

        "detection_loss":
            running_detection
            /
            actual_steps,

        "support_rank_loss":
            running_rank
            /
            actual_steps,

        "combined_loss":
            running_combined
            /
            actual_steps,

        "correct_similarity":
            running_correct_sim
            /
            actual_steps,

        "wrong_similarity":
            running_wrong_sim
            /
            actual_steps,

        "train_margin":
            running_margin
            /
            actual_steps,

        **val_metrics,
    }


    short_history.append(
        record
    )


    # ======================================================
    # BEST MODEL
    # primary = mAP50
    # tie-break = val loss
    # ======================================================

    is_better = (

        record[
            "episodic_map50"
        ]
        >
        short_best_map50

        or

        (
            abs(
                record[
                    "episodic_map50"
                ]
                -
                short_best_map50
            )
            <
            1e-12

            and

            record[
                "loss"
            ]
            <
            short_best_val_loss
        )
    )


    if is_better:

        short_best_epoch = (
            epoch + 1
        )

        short_best_map50 = float(

            record[
                "episodic_map50"
            ]
        )

        short_best_val_loss = float(

            record[
                "loss"
            ]
        )

        short_best_metrics = dict(
            record
        )


        short_best_state = {

            name:
                tensor
                .detach()
                .cpu()
                .clone()

            for name, tensor
            in short_model
            .state_dict()
            .items()
        }


    # ======================================================
    # REPORT
    # ======================================================

    print()
    print("=" * 70)

    print(
        f"SHORT EPOCH "
        f"{epoch+1}/"
        f"{TRAIN_CONFIG['short_epochs']}"
    )

    print("-" * 70)

    print(
        f"Detection Loss : "
        f"{record['detection_loss']:.4f}"
    )

    print(
        f"Support Rank   : "
        f"{record['support_rank_loss']:.6f}"
    )

    print(
        f"Combined Loss  : "
        f"{record['combined_loss']:.4f}"
    )

    print("-" * 70)

    print(
        f"Correct Sim    : "
        f"{record['correct_similarity']:.6f}"
    )

    print(
        f"Wrong Sim      : "
        f"{record['wrong_similarity']:.6f}"
    )

    print(
        f"Train Margin   : "
        f"{record['train_margin']:.6f}"
    )

    print("-" * 70)

    print(
        f"Val Loss       : "
        f"{record['loss']:.4f}"
    )

    print(
        f"Val mAP50      : "
        f"{record['episodic_map50']:.8f}"
    )

    print(
        f"Val P@0.50     : "
        f"{record['precision50']:.6f}"
    )

    print(
        f"Val R@0.50     : "
        f"{record['recall50']:.6f}"
    )

    print(
        f"Mean Best IoU  : "
        f"{record['mean_best_iou']:.6f}"
    )

    print(
        f"Loc Recall@.30 : "
        f"{record['localization_recall30']:.6f}"
    )

    print(
        f"Loc Recall@.50 : "
        f"{record['localization_recall50']:.6f}"
    )

    print("=" * 70)


    short_scheduler.step()


# ==========================================================
# RESTORE BEST SHORT MODEL
# ==========================================================

assert short_best_state is not None


short_model.load_state_dict(
    short_best_state
)


short_model.to(
    CONFIG["device"]
)


short_model.eval()


# ==========================================================
# GATE DECISION
# ==========================================================

required_map50 = max(

    TRAIN_CONFIG[
        "short_map50_absolute_floor"
    ],

    short_baseline_metrics[
        "episodic_map50"
    ]

    *

    TRAIN_CONFIG[
        "short_map50_relative_factor"
    ]
)


SHORT_GATE_PASSED = bool(

    short_best_map50
    >=
    required_map50
)


print()
print("=" * 70)
print("STEP 29 : SHORT GENERALIZATION RESULT")
print("=" * 70)

print(
    f"Baseline mAP50 : "
    f"{short_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"Required mAP50 : "
    f"{required_map50:.8f}"
)

print(
    f"Best Epoch     : "
    f"{short_best_epoch}"
)

print(
    f"Best Val Loss  : "
    f"{short_best_val_loss:.6f}"
)

print(
    f"Best mAP50     : "
    f"{short_best_map50:.8f}"
)

print(
    f"Best Mean IoU  : "
    f"{short_best_metrics['mean_best_iou']:.6f}"
)

print(
    f"Best Loc R@.50 : "
    f"{short_best_metrics['localization_recall50']:.6f}"
)

print("-" * 70)


if SHORT_GATE_PASSED:

    print(
        "✓ STEP 29 SHORT GENERALIZATION GATE PASSED"
    )

    print(
        "→ STEP 30 is SKIPPED by protocol."
    )

    print(
        "→ Next later: STEP 31 Official COCO Meta-Training."
    )


else:

    print(
        "✗ STEP 29 SHORT GENERALIZATION GATE FAILED"
    )

    print(
        "→ DO NOT RUN official 25-epoch training."
    )

    print(
        "→ RUN STEP 30 H1/H2/H3 diagnosis."
    )


print("=" * 70)

print(
    "✓ short_model restored to BEST short epoch"
)

print(
    "✓ official model remains untouched"
)

print("=" * 70)
