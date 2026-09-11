# ==========================================================
# STEP 29: Short COCO-Person Generalization
#
# Dijalankan hanya jika tiny learning lolos.
# Dimulai dari inisialisasi awal, bukan bobot tiny training.
# ==========================================================

SHORT_GATE_PASSED = False
short_history = []

if not TINY_GATE_PASSED:
    print(
        "STEP 29 SKIPPED: "
        "diagnose tiny learning in Step 30 first."
    )

else:
    short_model = make_trial_model()

    short_optimizer, short_scheduler = (
        build_optimizer_and_scheduler(
            short_model
        )
    )

    short_baseline_metrics = evaluate_person(
        short_model,
        val_loader
    )

    print_person_metrics(
        "SHORT INITIAL",
        short_baseline_metrics
    )

    short_best_state = None
    short_best_metrics = None
    short_best_epoch = -1

    for epoch in range(
        TRAIN_CONFIG["short_epochs"]
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_stats = train_detection_epoch(
            target_model=short_model,
            loader=train_loader,
            optimizer=short_optimizer,
            max_steps=TRAIN_CONFIG[
                "short_steps_per_epoch"
            ],
            description=f"PERSON SHORT {epoch + 1}",
        )

        metrics = evaluate_person(
            short_model,
            val_loader
        )

        short_history.append({
            "epoch": epoch + 1,
            "train": train_stats,
            "validation": metrics,
        })

        print_person_metrics(
            f"SHORT {epoch + 1}",
            metrics
        )

        is_better = (
            short_best_metrics is None
            or
            (
                metrics["person_ap50"],
                -metrics["loss"]
            )
            >
            (
                short_best_metrics["person_ap50"],
                -short_best_metrics["loss"]
            )
        )

        if is_better:
            short_best_metrics = dict(
                metrics
            )

            short_best_epoch = epoch + 1

            short_best_state = {
                key: (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )
                for key, value in short_model.state_dict().items()
            }

        short_scheduler.step()

    short_model.load_state_dict(
        short_best_state
    )

    short_model.eval()

    required_ap = max(
        TRAIN_CONFIG[
            "short_map50_absolute_floor"
        ],

        (
            short_baseline_metrics["person_ap50"]
            *
            TRAIN_CONFIG[
                "short_map50_relative_factor"
            ]
        ),
    )

    good_epochs = sum(
        (
            record["validation"]["person_ap50"]
            >= required_ap
        )
        for record in short_history
    )

    last_epoch_passed = (
        short_history[-1]["validation"]["person_ap50"]
        >= required_ap
    )

    SHORT_GATE_PASSED = bool(
        (
            good_epochs
            >= TRAIN_CONFIG[
                "short_min_epochs_above_floor"
            ]
        )
        and last_epoch_passed
    )

    print("=" * 70)
    print("STEP 29: SHORT TRAINING RESULT")
    print("=" * 70)

    print("Best epoch       :", short_best_epoch)
    print("Required AP      :", required_ap)
    print("Qualifying epochs:", good_epochs)
    print("Last epoch passed:", last_epoch_passed)
    print("SHORT_GATE_PASSED:", SHORT_GATE_PASSED)

    print_person_metrics(
        "BEST SHORT MODEL",
        short_best_metrics
    )

    # Checkpoint diagnosis saja.
    # Belum merupakan hasil full source training.
    checkpoint_config = {
        key: (
            str(value)
            if key == "device"
            else value
        )
        for key, value in CONFIG.items()
    }

    torch.save(
        {
            "model": short_best_state,
            "stage": "short_person_diagnostic",
            "epoch": short_best_epoch,
            "metrics": short_best_metrics,
            "config": checkpoint_config,
            "train_config": TRAIN_CONFIG,
            "person_category_id": PERSON_CAT_ID,
        },
        os.path.join(
            CHECKPOINT_DIR,
            "person_short_diagnostic_best.pth"
        )
    )

    with open(
        os.path.join(
            LOG_DIR,
            "short_person_history.json"
        ),
        "w"
    ) as handle:
        json.dump(
            short_history,
            handle,
            indent=2
        )

    short_model.cpu()

    del short_optimizer
    del short_scheduler
    del short_best_state

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "Step 30 will inspect the best short model. "
        "Full source training and CCTV adaptation are not included yet."
    )
