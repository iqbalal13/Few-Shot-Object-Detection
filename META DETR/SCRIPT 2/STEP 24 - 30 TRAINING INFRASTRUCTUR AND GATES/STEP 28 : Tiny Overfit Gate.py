# ==========================================================
# STEP 28: Fixed Person-Only Tiny Learning
#
# Jika gagal:
# - Step 29 melewati short training.
# - Step 30 mendiagnosis tiny model.
# ==========================================================

train_dataset.set_epoch(0)

tiny_indices = list(
    range(
        min(
            TRAIN_CONFIG["tiny_episodes"],
            len(train_dataset)
        )
    )
)

tiny_loader = make_episode_loader(
    Subset(
        train_dataset,
        tiny_indices
    ),
    num_workers=0,
)


# ==========================================================
# Simpan identitas episode tiny.
# ==========================================================

tiny_manifest = []

for index in tiny_indices:
    item = train_dataset[index]

    tiny_manifest.append({
        "episode_index": index,

        "support_annotation_id": (
            item["support_target"]["annotation_id"].item()
        ),

        "support_image_id": (
            item["support_target"]["image_id"].item()
        ),

        "query_image_id": (
            item["query_target"]["image_id"].item()
        ),
    })

with open(
    os.path.join(
        LOG_DIR,
        "tiny_person_manifest.json"
    ),
    "w"
) as handle:
    json.dump(
        tiny_manifest,
        handle,
        indent=2
    )


# ==========================================================
# Model copy dari inisialisasi awal.
# ==========================================================

tiny_model = make_trial_model()

tiny_optimizer, tiny_scheduler = (
    build_optimizer_and_scheduler(
        tiny_model
    )
)

tiny_initial_metrics = evaluate_person(
    tiny_model,
    tiny_loader
)

print_person_metrics(
    "TINY INITIAL",
    tiny_initial_metrics
)

tiny_history = []


# ==========================================================
# Fixed-set training.
# ==========================================================

for epoch in range(
    TRAIN_CONFIG["tiny_epochs"]
):
    train_stats = train_detection_epoch(
        target_model=tiny_model,
        loader=tiny_loader,
        optimizer=tiny_optimizer,
        max_steps=len(tiny_loader),
        description="Tiny",
        show_progress=False,
    )

    record = {
        "epoch": epoch + 1,
        "train": train_stats,
    }

    should_report = (
        epoch == 0
        or (epoch + 1) % 20 == 0
        or epoch + 1 == TRAIN_CONFIG["tiny_epochs"]
    )

    if should_report:
        metrics = evaluate_person(
            tiny_model,
            tiny_loader
        )

        record[
            "validation_on_tiny_training_set"
        ] = metrics

        print_person_metrics(
            f"TINY {epoch + 1}",
            metrics
        )

    tiny_history.append(
        record
    )


tiny_final_metrics = evaluate_person(
    tiny_model,
    tiny_loader
)

TINY_GATE_PASSED = bool(
    (
        tiny_final_metrics["loss"]
        < tiny_initial_metrics["loss"]
    )
    and
    (
        tiny_final_metrics["person_ap50"]
        >= TRAIN_CONFIG["tiny_ap50_floor"]
    )
    and
    (
        tiny_final_metrics[
            "one_to_one_localization_recall50"
        ]
        >= TRAIN_CONFIG[
            "tiny_localization_recall_floor"
        ]
    )
)

print_person_metrics(
    "TINY FINAL",
    tiny_final_metrics
)

print(
    "TINY_GATE_PASSED:",
    TINY_GATE_PASSED
)

print(
    "Inspect confidence and boxes too. "
    "This is a diagnostic gate, not final source-model validation."
)

with open(
    os.path.join(
        LOG_DIR,
        "tiny_person_history.json"
    ),
    "w"
) as handle:
    json.dump(
        tiny_history,
        handle,
        indent=2
    )


# Tetap tersedia untuk diagnosis, tetapi dipindahkan ke CPU.
tiny_model.cpu()

del tiny_optimizer, tiny_scheduler

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if not TINY_GATE_PASSED:
    print(
        "Tiny gate failed. Step 29 will skip training. "
        "Run Step 30 for tiny diagnosis."
    )
