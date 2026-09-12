# ==========================================================
# STEP 33: TINY XAVIER INDEPENDEN + DROPOUT 0
# Tambahkan setelah STEP 32 selesai, dalam runtime yang sama.
# Menguji inisialisasi layer STEP 7-8 pada salinan model awal.
# Tetap: encoder 6, decoder 6, 100 query, 10 episode, 1.000 update.
# ==========================================================

import copy
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

_xi_required = (
    "model", "no_dropout_model", "train_dataset", "CONFIG", "COCO_CONFIG",
    "TRAIN_CONFIG", "CHECKPOINT_DIR", "criterion", "make_episode_loader",
    "make_trial_model", "build_optimizer_and_scheduler", "train_detection_epoch",
    "evaluate_person", "compute_person_metrics", "as_numpy",
    "_nd_fingerprint", "_nd_write_json", "_nd_disable_dropout", "_nd_evaluate",
    "_nd_cpu_tree", "nd_experiment", "nd_summary", "nd_manifest", "ND_INDICES",
)
_xi_missing = [name for name in _xi_required if name not in globals()]
if _xi_missing:
    raise RuntimeError(
        "STEP 33 perlu runtime STEP 32 yang sudah selesai. Belum tersedia: "
        + ", ".join(_xi_missing)
    )
if nd_summary["total_updates"] != 1000 or nd_experiment["effective_dropout"] != 0:
    raise RuntimeError("Pembanding harus STEP 32 dropout 0 dengan 1.000 update.")
if COCO_CONFIG["batch_size"] != 1 or COCO_CONFIG["num_classes"] != 1:
    raise ValueError("Gunakan batch_size=1 dan person-only seperti STEP 32.")
_xi_config = {
    k: str(v) if isinstance(v, torch.device) else v for k, v in CONFIG.items()
}
if _xi_config != nd_experiment["model_config"]:
    raise RuntimeError("CONFIG berubah sejak STEP 32; samakan sebelum perbandingan.")
for _xi_key in (
    "learning_rate", "backbone_learning_rate", "weight_decay", "gradient_clip",
    "score_threshold", "iou_threshold", "tiny_ap50_floor",
    "tiny_localization_recall_floor",
):
    if TRAIN_CONFIG[_xi_key] != nd_experiment["train_config"][_xi_key]:
        raise RuntimeError(f"TRAIN_CONFIG[{_xi_key!r}] berubah sejak STEP 32.")

XI_EPOCHS = 100
XI_INDICES = [int(i) for i in ND_INDICES]
if len(XI_INDICES) != 10 or len(set(XI_INDICES)) != 10:
    raise ValueError("Diperlukan 10 episode berbeda, sama dengan STEP 32.")
_xi_source_hash = _nd_fingerprint(model)
if _xi_source_hash != nd_experiment["source_state_sha256"]:
    raise RuntimeError("model bukan lagi bobot awal yang digunakan STEP 32.")
_xi_baseline_hash = _nd_fingerprint(no_dropout_model)


def _xi_initialize_layers(target_model, reference_model, seed):
    """Xavier hanya untuk matriks di dalam encoder.layers/decoder.layers."""
    stacks = (
        ("transformer_encoder", target_model.transformer_encoder.layers),
        ("transformer_decoder", target_model.transformer_decoder.layers),
    )
    if any(len(layers) != 6 for _, layers in stacks):
        raise RuntimeError("Percobaan ini mempertahankan encoder 6 dan decoder 6.")
    if target_model.transformer_decoder.num_queries != 100:
        raise RuntimeError("Jumlah object query harus tetap 100.")

    expected = []
    # Seed hanya SEKALI. Jangan reset seed di setiap layer.
    # fork_rng memulihkan RNG agar inisialisasi tidak menggeser RNG training.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(int(seed))
        with torch.no_grad():
            for stack_name, layers in stacks:
                for index, layer in enumerate(layers):
                    for name, parameter in layer.named_parameters():
                        if parameter.ndim > 1:
                            nn.init.xavier_uniform_(parameter)
                            expected.append(f"{stack_name}.layers.{index}.{name}")

    reference = reference_model.state_dict()
    current = target_model.state_dict()
    if set(reference) != set(current):
        raise RuntimeError("Struktur state_dict berubah.")
    changed = {
        name for name in current
        if not torch.equal(current[name].detach().cpu(), reference[name].detach().cpu())
    }
    if len(expected) != 60 or changed != set(expected):
        raise RuntimeError(
            "Perubahan bobot di luar rencana: "
            f"expected={len(expected)}, changed={len(changed)}, "
            f"extra={sorted(changed - set(expected))}, "
            f"missing={sorted(set(expected) - changed)}"
        )
    for stack_name, layers in stacks:
        matrices = [dict(layer.named_parameters()) for layer in layers]
        for name, parameter in matrices[0].items():
            if parameter.ndim <= 1:
                continue
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    left, right = matrices[i][name], matrices[j][name]
                    if left.data_ptr() == right.data_ptr() or torch.equal(left, right):
                        raise RuntimeError(f"Layer masih identik/shared: {stack_name}.{name}")
    return {
        "changed_matrices": expected,
        "matrix_count": len(expected),
        "parameter_elements": sum(current[name].numel() for name in expected),
        "outside_scope_unchanged": True,
        "layer_matrices_distinct": True,
    }


def _xi_print(epoch, report, train_stats=None):
    div = report["diversity_by_stage"]
    train_text = "-" if train_stats is None else f'{train_stats["loss_total"]:.4f}'
    print(
        f'Epoch {epoch:3d} | train loss {train_text}'
        f' | eval loss {report["mean_loss"]["loss_total"]:.4f}'
        f' | AP50 {report["AP50_percent"]:.4f}%'
        f' | geometry {report["geometry_recall50"]:.4f}'
        f' | IoU {report["mean_best_iou"]:.4f}'
        f' | enc6 {div["encoder_6"]:.3e} | dec6 {div["decoder_6"]:.3e}'
        f' | box std {report["box_std_across_queries"]:.3e}', flush=True,
    )


def _xi_save_checkpoint(path, target_model, epoch, report, optimizer=None):
    payload = {
        "model": _nd_cpu_tree(target_model.state_dict()),
        "epoch": epoch, "updates": epoch * 10,
        "report": report, "experiment": xi_experiment,
    }
    if optimizer is not None:
        payload["optimizer"] = _nd_cpu_tree(optimizer.state_dict())
        payload["torch_rng_state"] = torch.get_rng_state()
        payload["cuda_rng_states"] = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _xi_collect_predictions(target_model, manifest):
    records, per_image = [], []
    old_modes = [(module, module.training) for module in target_model.modules()]
    try:
        target_model.eval()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            with torch.inference_mode():
                for identity in manifest:
                    item = train_dataset[identity["episode_index"]]
                    output = target_model(
                        item["support_image"].unsqueeze(0).to(CONFIG["device"]),
                        item["query_image"].unsqueeze(0).to(CONFIG["device"]),
                    )
                    scores = as_numpy(output["pred_logits"][0, :, 0].sigmoid())
                    boxes = as_numpy(output["pred_boxes"][0])
                    gt = as_numpy(item["query_target"]["boxes"])
                    if len(scores) != 100 or boxes.shape != (100, 4):
                        raise RuntimeError("Output harus berisi seluruh 100 query.")
                    record = {
                        **identity, "scores": scores.tolist(),
                        "pred_boxes": boxes.tolist(), "gt_boxes": gt.tolist(),
                    }
                    metrics = compute_person_metrics(
                        [record], TRAIN_CONFIG["score_threshold"], TRAIN_CONFIG["iou_threshold"]
                    )
                    per_image.append({
                        **identity, "queries": len(scores), "gt": len(gt),
                        "geometry_hits50": int(round(
                            metrics["one_to_one_localization_recall50"] * len(gt)
                        )),
                        "geometry_recall50": metrics["one_to_one_localization_recall50"],
                        "mean_best_iou": metrics["mean_best_iou"],
                        "max_score": float(scores.max()),
                        "detections_at_threshold": metrics["tp"] + metrics["fp"],
                        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
                        "box_std_across_queries": float(boxes.std(axis=0).mean()),
                    })
                    records.append(record)
    finally:
        for module, training in old_modes:
            module.training = training
    return records, per_image


def _xi_visualize(records, folder, epoch):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for record in records[:3]:
        index = record["episode_index"]
        item = train_dataset[index]
        rgb = item["query_image"].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(
            rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1
        )
        height, width = rgb.shape[:2]
        scores = np.asarray(record["scores"])
        boxes = np.asarray(record["pred_boxes"])
        gt = np.asarray(record["gt_boxes"]).reshape(-1, 4)
        selected = np.argsort(-scores, kind="stable")[:10]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for axis in axes:
            axis.imshow(rgb)
            axis.set_xlim(0, width)
            axis.set_ylim(height, 0)
            axis.axis("off")
        for axis, items, color in ((axes[0], gt, "lime"), (axes[1], boxes[selected], "red")):
            for j, (cx, cy, w, h) in enumerate(items):
                x, y = (cx - w / 2) * width, (cy - h / 2) * height
                axis.add_patch(Rectangle(
                    (x, y), w * width, h * height,
                    fill=False, edgecolor=color, linewidth=1.5,
                ))
                if axis is axes[1]:
                    axis.text(x, y, f"{scores[selected[j]]:.3f}", color="white",
                              backgroundcolor="black", fontsize=8)
        axes[0].set_title(f"Episode {index}: GT ({len(gt)} person)")
        axes[1].set_title(f"Xavier epoch {epoch}: top-10, tanpa filter skor")
        fig.tight_layout()
        fig.savefig(Path(folder) / f"episode_{index}_final.png", dpi=140)
        plt.show()
        plt.close(fig)


XI_RUN_DIR = Path(CHECKPOINT_DIR) / "tiny_xavier_no_dropout" / datetime.now(
    timezone.utc
).strftime("%Y%m%d_%H%M%S_%f")
XI_RUN_DIR.mkdir(parents=True, exist_ok=False)
xi_model, xi_optimizer = None, None
xi_history = []
xi_best_epoch = 0
XI_TINY_GATE_PASSED = False
_xi_old_epoch = train_dataset.epoch

try:
    train_dataset.set_epoch(0)
    xi_manifest = []
    for index in XI_INDICES:
        item = train_dataset[index]
        xi_manifest.append({
            "episode_index": index,
            "support_annotation_id": int(item["support_target"]["annotation_id"].item()),
            "support_image_id": int(item["support_target"]["image_id"].item()),
            "query_image_id": int(item["query_target"]["image_id"].item()),
        })
    if xi_manifest != nd_manifest or xi_manifest != nd_experiment["manifest"]:
        raise RuntimeError("Support/query berbeda dari STEP 32; percobaan dihentikan.")

    # Factory menyalin model AWAL dan mengatur seed training seperti STEP 32.
    xi_model = make_trial_model().cpu()
    if _nd_fingerprint(xi_model) != _xi_source_hash:
        raise RuntimeError("Salinan tidak identik dengan bobot model awal.")
    xi_dropout_changes = _nd_disable_dropout(xi_model)
    xi_init_check = _xi_initialize_layers(xi_model, model, CONFIG["seed"])
    xi_initial_hash = _nd_fingerprint(xi_model)
    xi_model.to(CONFIG["device"])
    xi_optimizer, _xi_unused_scheduler = build_optimizer_and_scheduler(xi_model)
    del _xi_unused_scheduler
    # LR konstan: sama dengan STEP 32; tidak ada scheduler.step().
    xi_loader = make_episode_loader(Subset(train_dataset, XI_INDICES), num_workers=0)
    if len(xi_loader) != 10:
        raise RuntimeError("Loader harus menghasilkan 10 update per epoch.")
    xi_lrs = [group["lr"] for group in xi_optimizer.param_groups]
    if xi_lrs != nd_experiment["learning_rates"]:
        raise RuntimeError("Learning rate berbeda dari STEP 32.")
    xi_experiment = {
        "name": "tiny_xavier_no_dropout", "epochs": XI_EPOCHS,
        "expected_updates": 1000, "dataset_epoch": 0,
        "source_state_sha256": _xi_source_hash,
        "initialized_state_sha256": xi_initial_hash,
        "initialization_seed": int(CONFIG["seed"]),
        "initialization": xi_init_check,
        "model_config": _xi_config, "train_config": copy.deepcopy(TRAIN_CONFIG),
        "effective_dropout": 0.0, "dropout_changes": xi_dropout_changes,
        "learning_rates": xi_lrs, "scheduler_stepped": False,
        "manifest": xi_manifest,
        "step32_directory": str(globals().get("ND_RUN_DIR", "")),
    }
    _nd_write_json(XI_RUN_DIR / "experiment.json", xi_experiment)
    print("Folder hasil:", XI_RUN_DIR)
    print("Manifest sama: PASS | hanya 60 matriks encoder-decoder berubah: PASS")
    print("Matriks antarlayer berbeda: PASS | parameter lainnya identik: PASS")
    print("Dropout 0 pada", len(xi_dropout_changes), "module/attention | LR:", xi_lrs)
    print("Loss awal BOLEH berbeda dari STEP 32 karena inisialisasi berubah.")

    xi_initial_report = _nd_evaluate(xi_model, xi_loader)
    xi_best_report = copy.deepcopy(xi_initial_report)
    xi_history.append({"epoch": 0, "eval": xi_initial_report})
    _xi_print(0, xi_initial_report)
    _xi_save_checkpoint(XI_RUN_DIR / "initial.pth", xi_model, 0, xi_initial_report)
    _xi_save_checkpoint(XI_RUN_DIR / "best.pth", xi_model, 0, xi_initial_report)
    _xi_save_checkpoint(XI_RUN_DIR / "last.pth", xi_model, 0, xi_initial_report, xi_optimizer)

    for xi_epoch in range(1, XI_EPOCHS + 1):
        train_dataset.set_epoch(0)
        xi_train_stats = train_detection_epoch(
            target_model=xi_model, loader=xi_loader, optimizer=xi_optimizer,
            max_steps=len(xi_loader), description=f"Xavier {xi_epoch}/{XI_EPOCHS}",
            show_progress=False,
        )
        if xi_train_stats["updates"] != 10:
            raise RuntimeError("Jumlah update per epoch tidak sesuai.")
        xi_row = {"epoch": xi_epoch, "train": xi_train_stats}
        if xi_epoch in (1, 5) or xi_epoch % 10 == 0:
            xi_report = _nd_evaluate(xi_model, xi_loader)
            xi_row["eval"] = xi_report
            _xi_print(xi_epoch, xi_report, xi_train_stats)
            xi_rank = lambda r: (
                r["AP50_percent"], r["geometry_recall50"], -r["mean_loss"]["loss_total"]
            )
            if xi_rank(xi_report) > xi_rank(xi_best_report):
                xi_best_report = copy.deepcopy(xi_report)
                xi_best_epoch = xi_epoch
                _xi_save_checkpoint(XI_RUN_DIR / "best.pth", xi_model, xi_epoch, xi_report)
            _xi_save_checkpoint(
                XI_RUN_DIR / "last.pth", xi_model, xi_epoch, xi_report, xi_optimizer
            )
            xi_final_report = xi_report
        xi_history.append(xi_row)
        _nd_write_json(XI_RUN_DIR / "history.json", xi_history)

    XI_TINY_GATE_PASSED = bool(
        xi_final_report["mean_loss"]["loss_total"] < xi_initial_report["mean_loss"]["loss_total"]
        and xi_final_report["metrics"]["person_ap50"] >= TRAIN_CONFIG["tiny_ap50_floor"]
        and xi_final_report["geometry_recall50"] >= TRAIN_CONFIG["tiny_localization_recall_floor"]
    )
    xi_summary = {
        "initial_eval": xi_initial_report, "final_eval": xi_final_report,
        "best_epoch": xi_best_epoch, "best_eval": xi_best_report,
        "XI_TINY_GATE_PASSED": XI_TINY_GATE_PASSED,
        "total_updates": sum(row.get("train", {}).get("updates", 0) for row in xi_history),
        "step32_reference": {
            key: nd_summary[key] for key in ("initial_eval", "final_eval", "best_epoch", "best_eval")
        },
        "difference_from_step32": {
            "final_AP50_percentage_points": xi_final_report["AP50_percent"] - nd_summary["final_eval"]["AP50_percent"],
            "best_AP50_percentage_points": xi_best_report["AP50_percent"] - nd_summary["best_eval"]["AP50_percent"],
            "final_geometry_recall50": xi_final_report["geometry_recall50"] - nd_summary["final_eval"]["geometry_recall50"],
            "final_mean_best_iou": xi_final_report["mean_best_iou"] - nd_summary["final_eval"]["mean_best_iou"],
        },
    }
    _nd_write_json(XI_RUN_DIR / "summary.json", xi_summary)
    xi_predictions, xi_per_image = _xi_collect_predictions(xi_model, xi_manifest)
    _nd_write_json(XI_RUN_DIR / "predictions_final.json", {
        "epoch": XI_EPOCHS, "box_format": "normalized_cxcywh",
        "score_threshold": TRAIN_CONFIG["score_threshold"],
        "episodes": xi_predictions,
    })
    xi_summary["per_image_final_all_100_queries"] = xi_per_image
    _nd_write_json(XI_RUN_DIR / "summary.json", xi_summary)
    print("\nHASIL STEP 33")
    print(json.dumps(xi_summary, indent=2, allow_nan=False))
    print("\nPER GAMBAR: seluruh 100 query; geometry memakai pencocokan satu-ke-satu")
    for row in xi_per_image:
        print(
            f'Episode {row["episode_index"]}: GT={row["gt"]}'
            f' | geometry hit={row["geometry_hits50"]}/{row["gt"]}'
            f' | mean IoU={row["mean_best_iou"]:.4f}'
            f' | max score={row["max_score"]:.4f}'
            f' | lolos cutoff={row["detections_at_threshold"]}'
        )
    print("Checkpoint terbaik:", XI_RUN_DIR / "best.pth", "| epoch", xi_best_epoch)
    print("Checkpoint epoch 100 + optimizer:", XI_RUN_DIR / "last.pth")
    print("Prediksi lengkap seluruh query:", XI_RUN_DIR / "predictions_final.json")
    _xi_visualize(xi_predictions, XI_RUN_DIR, XI_EPOCHS)
finally:
    train_dataset.set_epoch(_xi_old_epoch)
    if xi_optimizer is not None:
        xi_optimizer.zero_grad(set_to_none=True)
        xi_optimizer = None
    if xi_model is not None:
        xi_model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _nd_fingerprint(model) != _xi_source_hash:
        raise RuntimeError("Bobot model awal berubah selama percobaan.")
    if _nd_fingerprint(no_dropout_model) != _xi_baseline_hash:
        raise RuntimeError("Bobot pembanding STEP 32 berubah selama percobaan.")

print("STEP 33 selesai. xi_model = epoch 100 di CPU; best.pth = epoch terbaik.")
print("Gate STEP 28/32 tidak diubah. STEP 29 tidak dijalankan otomatis.")
