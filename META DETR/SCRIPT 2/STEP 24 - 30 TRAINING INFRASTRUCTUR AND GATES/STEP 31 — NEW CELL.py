# ==========================================================
# STEP 31 — FULL REPLACEMENT
# COCO-Val PERSON readiness gate
#
# HARD:
#   Precision@0.50 > 0
#   Recall@0.50    > 0
#   internal Geometry50 improves vs initial model
#
# PASS:
#   use generic stable COCO-80 checkpoint
#
# FAIL:
#   activate COCO-person specialization fallback
# ==========================================================


if not COCO80_META_TRAINING_COMPLETE:

    raise RuntimeError(
        'Stable COCO-80 source training '
        'is required before person gate.'
    )


# ==========================================================
# Free full-training objects
# ==========================================================

if 'source_model' in globals():
    source_model.cpu()


for _name in (
    'source_model',
    'source_optimizer',
    'source_scheduler',
):

    globals().pop(
        _name,
        None,
    )


gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================================
# Fixed person-only validation episodes
# ==========================================================

PERSON_GATE_EPISODES = 200


person_val_dataset = COCOEpisodicDataset(

    coco=
        coco_val,

    image_dir=
        VAL_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        PERSON_GATE_EPISODES,

    seed=
        COCO_CONFIG[
            'seed'
        ]
        +
        200_000,

    cat_id_to_label=
        CAT_ID_TO_LABEL,

    category_names=
        CATEGORY_NAMES,

    min_bbox_size=
        COCO_CONFIG[
            'min_bbox_size'
        ],

    allowed_labels=[
        PERSON_LABEL
    ],
)


person_val_dataset.set_epoch(
    0
)


person_val_loader = (
    make_episode_loader(

        person_val_dataset,

        batch_size=1,

        num_workers=
            COCO_CONFIG[
                'num_workers'
            ],
    )
)


# ==========================================================
# ORIGINAL INITIALIZATION BASELINE
# ==========================================================

person_initial_model = (
    make_trial_model()
)


person_initial_report = (
    evaluate_episodic_model(

        person_initial_model,

        person_val_loader,

        show_progress=True,
    )
)


person_initial_model.cpu()

del person_initial_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================================
# STABLE COCO-80 CHECKPOINT
# ==========================================================

person_gate_model = (
    make_trial_model()
)


_source_checkpoint = torch.load(

    COCO80_BEST_CHECKPOINT_PATH,

    map_location=
        CONFIG[
            'device'
        ],

    weights_only=False,
)


person_gate_model.load_state_dict(

    _source_checkpoint[
        'model_state'
    ],

    strict=True,
)


person_source_report = (
    evaluate_episodic_model(

        person_gate_model,

        person_val_loader,

        show_progress=True,
    )
)


# ==========================================================
# READINESS CRITERIA
# ==========================================================

person_initial_geometry = float(

    person_initial_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


person_source_geometry = float(

    person_source_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


person_geometry_improved = bool(

    person_source_geometry
    >
    person_initial_geometry
)


person_precision = float(

    person_source_report[
        'metrics'
    ][
        'precision50'
    ]
)


person_recall = float(

    person_source_report[
        'metrics'
    ][
        'recall50'
    ]
)


person_pr_alive = bool(

    person_precision
    >
    0.0

    and

    person_recall
    >
    0.0
)


COCO_PERSON_GATE_PASSED = bool(

    person_geometry_improved

    and

    person_pr_alive
)


PERSON_SPECIALIZATION_REQUIRED = bool(
    not COCO_PERSON_GATE_PASSED
)


SOURCE_READY_FOR_CCTV = bool(
    COCO_PERSON_GATE_PASSED
)


SOURCE_CHECKPOINT_FOR_CCTV = (

    COCO80_BEST_CHECKPOINT_PATH

    if SOURCE_READY_FOR_CCTV

    else None
)


print('=' * 70)
print('STEP 31 : COCO-VAL PERSON READINESS GATE')
print('=' * 70)

print(
    'Episodes              :',
    PERSON_GATE_EPISODES,
)

print(
    'Precision@0.50        :',
    person_precision,
)

print(
    'Recall@0.50           :',
    person_recall,
)

print(
    'Initial Geometry50    :',
    person_initial_geometry,
)

print(
    'Source Geometry50     :',
    person_source_geometry,
)

print(
    'Geometry improved     :',
    person_geometry_improved,
)

print(
    'Precision/Recall alive:',
    person_pr_alive,
)

print(
    'PERSON GATE PASSED    :',
    COCO_PERSON_GATE_PASSED,
)

print(
    'CCTV READY            :',
    SOURCE_READY_FOR_CCTV,
)

print(
    'Fallback required     :',
    PERSON_SPECIALIZATION_REQUIRED,
)

print(
    'Source checkpoint     :',
    SOURCE_CHECKPOINT_FOR_CCTV,
)

print('=' * 70)


person_gate_model.cpu()

del person_gate_model
del _source_checkpoint

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
