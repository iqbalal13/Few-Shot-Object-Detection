# ==========================================================
# STEP 31: Diagnosis Model Tiny — Tanpa Training Ulang
# Jalankan setelah STEP 30 pada runtime yang sama.
# ==========================================================

import json
import numpy as np
import torch

required = (
    "model",
    "tiny_model",
    "train_dataset",
    "tiny_indices",
    "tiny_history",
    "criterion",
    "CONFIG",
    "move_targets_to_device",
    "compute_person_metrics",
    "as_numpy",
    "freeze_backbone_bn_statistics",
)

missing = [
    name
    for name in required
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "State runtime belum tersedia: "
        + ", ".join(missing)
        + ". Sel ini memerlukan runtime hasil STEP 28–30."
    )


def diagnostic_diversity(tensor):
    # Variasi relatif antarposisi/query.
    # Nilai 0 berarti semua posisi/query identik.
    tensor = tensor.detach().float()

    centered = tensor - tensor.mean(
        dim=1,
        keepdim=True,
    )

    return (
        centered.square().mean().sqrt()
        / tensor.square().mean().sqrt().clamp_min(1e-12)
    ).item()


def inspect_existing_model(
    target_model,
    dropout_active=False,
):
    old_device = next(
        target_model.parameters()
    ).device

    old_modes = [
        (module, module.training)
        for module in target_model.modules()
    ]

    hooks = []
    activations = {}
    stage_values = {}

    records = []
    losses = []
    box_spreads = []
    logit_spreads = []

    def capture(name):
        def hook(module, inputs, output):
            tensor = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            activations[name] = diagnostic_diversity(
                tensor
            )

        return hook

    try:
        target_model.to(CONFIG["device"])
        target_model.train(dropout_active)

        # BN tidak memperbarui running statistics.
        freeze_backbone_bn_statistics(
            target_model.backbone
        )

        stages = [
            ("query_tokens", target_model.query_encoder)
        ]

        stages += [
            (f"encoder_{i + 1}", layer)
            for i, layer in enumerate(
                target_model.transformer_encoder.layers
            )
        ]

        stages += [
            (f"decoder_{i + 1}", layer)
            for i, layer in enumerate(
                target_model.transformer_decoder.layers
            )
        ]

        stages.append(
            ("relation", target_model.relation_module)
        )

        for name, module in stages:
            hooks.append(
                module.register_forward_hook(
                    capture(name)
                )
            )

        cuda_devices = list(
            range(torch.cuda.device_count())
        )

        # State RNG dikembalikan setelah pemeriksaan.
        with torch.random.fork_rng(
            devices=cuda_devices
        ):
            torch.manual_seed(CONFIG["seed"])

            with torch.inference_mode():
                for index in tiny_indices:
                    episode = train_dataset[index]

                    support = (
                        episode["support_image"]
                        .unsqueeze(0)
                        .to(CONFIG["device"])
                    )

                    query = (
                        episode["query_image"]
                        .unsqueeze(0)
                        .to(CONFIG["device"])
                    )

                    targets = move_targets_to_device(
                        [episode["query_target"]],
                        CONFIG["device"],
                    )

                    activations.clear()

                    outputs, extras = (
                        target_model.forward_with_features(
                            support,
                            query,
                        )
                    )

                    loss_dict = criterion(
                        outputs,
                        targets,
                    )

                    losses.append({
                        key: value.item()
                        for key, value in loss_dict.items()
                    })

                    for name, value in activations.items():
                        stage_values.setdefault(
                            name, []
                        ).append(value)

                    box_spreads.append(
                        outputs["pred_boxes"]
                        .std(dim=1, unbiased=False)
                        .mean()
                        .item()
                    )

                    logit_spreads.append(
                        outputs["pred_logits"]
                        .std(dim=1, unbiased=False)
                        .mean()
                        .item()
                    )

                    records.append({
                        "scores": as_numpy(
                            outputs["pred_logits"][0, :, 0]
                            .sigmoid()
                        ),
                        "pred_boxes": as_numpy(
                            outputs["pred_boxes"][0]
                        ),
                        "gt_boxes": as_numpy(
                            targets[0]["boxes"]
                        ),
                    })

        metrics = compute_person_metrics(records)

        return {
            "mean_loss": {
                key: float(np.mean([
                    row[key]
                    for row in losses
                ]))
                for key in losses[0]
            },

            "diversity_by_stage": {
                key: float(np.mean(values))
                for key, values in stage_values.items()
            },

            "box_std_across_queries": float(
                np.mean(box_spreads)
            ),

            "logit_std_across_queries": float(
                np.mean(logit_spreads)
            ),

            "AP50_percent": (
                100 * metrics["person_ap50"]
            ),

            "geometry_recall50": metrics[
                "one_to_one_localization_recall50"
            ],

            "mean_best_iou": metrics[
                "mean_best_iou"
            ],
        }

    finally:
        for handle in hooks:
            handle.remove()

        target_model.to(old_device)

        for module, training in old_modes:
            module.training = training


old_epoch = train_dataset.epoch
diagnostic_report = {}

try:
    train_dataset.set_epoch(0)

    for name, target_model, dropout_active in (
        ("initial_eval", model, False),
        ("tiny_eval", tiny_model, False),
        (
            "tiny_dropout_active_bn_frozen",
            tiny_model,
            True,
        ),
    ):
        print(
            "Memeriksa:",
            name,
            flush=True,
        )

        diagnostic_report[name] = inspect_existing_model(
            target_model,
            dropout_active,
        )

finally:
    train_dataset.set_epoch(old_epoch)


diagnostic_report["recorded_training_losses"] = [
    {
        "epoch": row["epoch"],
        **row["train"],
    }
    for row in tiny_history
    if row["epoch"] in (
        1, 20, 40, 60, 80, 100
    )
]

print(
    json.dumps(
        diagnostic_report,
        indent=2,
    )
)

print(
    "STEP 31 selesai. "
    "Tidak ada backward atau optimizer update."
)
