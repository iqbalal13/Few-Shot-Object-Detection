# ==========================================================
# STEP 24: Person-Only Training Configuration and Helpers
# ==========================================================

import gc
import json

from itertools import islice
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm


TRAIN_CONFIG = {
    # Digunakan untuk rencana full source training berikutnya.
    "epochs": 25,
    "steps_per_epoch": 800,

    # Optimizer tetap seperti konfigurasi awal.
    "learning_rate": 1e-4,
    "backbone_learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "gradient_clip": 0.1,

    "milestones": [15, 20],
    "gamma": 0.1,

    # Evaluation
    "score_threshold": 0.5,
    "iou_threshold": 0.5,

    # Tiny learning
    "tiny_episodes": 10,
    "tiny_epochs": 100,

    # Default diagnosis; bukan target performa akhir tesis.
    "tiny_ap50_floor": 0.80,
    "tiny_localization_recall_floor": 0.80,

    # Short generalization
    "short_epochs": 5,
    "short_steps_per_epoch": 800,

    "short_map50_absolute_floor": 0.01,
    "short_map50_relative_factor": 1.25,

    # Menghindari kelulusan hanya karena satu lonjakan epoch.
    "short_min_epochs_above_floor": 2,
}


def reset_trial_seed():
    random.seed(
        CONFIG["seed"]
    )

    np.random.seed(
        CONFIG["seed"]
    )

    torch.manual_seed(
        CONFIG["seed"]
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            CONFIG["seed"]
        )


def make_trial_model():
    # Official model tetap memakai bobot awal.
    # Dipindah ke CPU agar tidak ada dua model penuh di GPU.
    model.cpu()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reset_trial_seed()

    return copy.deepcopy(
        model
    ).to(
        CONFIG["device"]
    )


def freeze_backbone_bn_statistics(
    backbone,
):
    # Membekukan running statistics BN.
    # Parameter backbone tetap trainable.
    for module in backbone.modules():
        if isinstance(
            module,
            nn.BatchNorm2d
        ):
            module.eval()


def build_optimizer_and_scheduler(
    target_model,
):
    backbone_params = [
        parameter
        for parameter in target_model.backbone.parameters()
        if parameter.requires_grad
    ]

    backbone_ids = {
        id(parameter)
        for parameter in backbone_params
    }

    main_params = [
        parameter
        for parameter in target_model.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in backbone_ids
        )
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": main_params,
                "lr": TRAIN_CONFIG["learning_rate"],
            },
            {
                "params": backbone_params,
                "lr": TRAIN_CONFIG["backbone_learning_rate"],
            },
        ],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=TRAIN_CONFIG["milestones"],
        gamma=TRAIN_CONFIG["gamma"],
    )

    return optimizer, scheduler


def compute_detection_training_loss(
    target_model,
    batch,
    device,
):
    assert (
        batch["episode_classes"] == 0
    ).all()

    targets = move_targets_to_device(
        batch["query_targets"],
        device
    )

    outputs = target_model(
        batch["support_images"].to(device),
        batch["query_images"].to(device),
    )

    losses = criterion(
        outputs,
        targets
    )

    if not all(
        torch.isfinite(value).all()
        for value in losses.values()
    ):
        raise RuntimeError(
            "Non-finite detection loss."
        )

    return losses


def train_detection_epoch(
    target_model,
    loader,
    optimizer,
    max_steps,
    description,
    show_progress=True,
):
    target_model.train()

    freeze_backbone_bn_statistics(
        target_model.backbone
    )

    totals = {
        key: 0.0
        for key in (
            "loss_cls",
            "loss_bbox",
            "loss_giou",
            "loss_total",
        )
    }

    count = 0

    iterator = islice(
        loader,
        max_steps
    )

    if show_progress:
        iterator = tqdm(
            iterator,
            total=min(
                max_steps,
                len(loader)
            ),
            desc=description,
        )

    for batch in iterator:
        optimizer.zero_grad(
            set_to_none=True
        )

        losses = compute_detection_training_loss(
            target_model,
            batch,
            CONFIG["device"],
        )

        losses["loss_total"].backward()

        torch.nn.utils.clip_grad_norm_(
            target_model.parameters(),
            TRAIN_CONFIG["gradient_clip"],
            error_if_nonfinite=True,
        )

        optimizer.step()

        for key in totals:
            totals[key] += (
                losses[key]
                .detach()
                .item()
            )

        count += 1

    if count == 0:
        raise RuntimeError(
            "No training updates."
        )

    return {
        **{
            key: value / count
            for key, value in totals.items()
        },
        "updates": count,
    }


def evaluate_person(
    target_model,
    loader,
):
    return evaluate_episodic_model(
        model=target_model,
        data_loader=loader,
        criterion=criterion,
        device=CONFIG["device"],
        score_threshold=TRAIN_CONFIG["score_threshold"],
        iou_threshold=TRAIN_CONFIG["iou_threshold"],
        show_progress=False,
    )


BEST_SOURCE_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "person_source_best.pth"
)

LATEST_SOURCE_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "person_source_latest.pth"
)

print("=" * 70)
print("STEP 24: TRAINING HELPERS READY")
print("=" * 70)

print(
    "Short training updates:",
    (
        TRAIN_CONFIG["short_epochs"]
        * TRAIN_CONFIG["short_steps_per_epoch"]
    )
)

print(
    "Tiny diagnostic AP floor:",
    TRAIN_CONFIG["tiny_ap50_floor"]
)

print(
    "Tiny diagnostic geometry recall floor:",
    TRAIN_CONFIG["tiny_localization_recall_floor"]
)

print("=" * 70)
