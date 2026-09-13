# ==========================================================
# STEP 34: ENCODER 3 + DECODER 6, DROPOUT 0
# Tambahkan setelah STEP 33, dalam runtime yang sama.
# Pembanding utama: STEP 32. Salinan berasal dari model AWAL.
# 10 episode tetap, 100 epoch, 1.000 update, tanpa Xavier ulang.
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

_e3_required = (
    "model", "no_dropout_model", "train_dataset", "CONFIG", "COCO_CONFIG",
    "TRAIN_CONFIG", "CHECKPOINT_DIR", "criterion", "make_episode_loader",
    "make_trial_model", "build_optimizer_and_scheduler", "train_detection_epoch",
    "evaluate_person", "compute_person_metrics", "as_numpy",
    "_nd_fingerprint", "_nd_write_json", "_nd_disable_dropout", "_nd_evaluate",
    "_nd_cpu_tree", "_xi_collect_predictions", "nd_experiment", "nd_summary",
    "nd_manifest", "ND_INDICES",
)
_e3_missing = [name for name in _e3_required if name not in globals()]
if _e3_missing:
    raise RuntimeError(
        "Gunakan runtime tempat STEP 32 dan 33 selesai. Belum tersedia: "
        + ", ".join(_e3_missing)
    )
if nd_summary["total_updates"] != 1000 or nd_experiment["effective_dropout"] != 0:
    raise RuntimeError("Pembanding harus STEP 32 dropout 0 dengan 1.000 update.")
if COCO_CONFIG["batch_size"] != 1 or COCO_CONFIG["num_classes"] != 1:
    raise ValueError("Percobaan ini memakai batch_size=1 dan person-only.")
_e3_control_config = {
    k: str(v) if isinstance(v, torch.device) else v for k, v in CONFIG.items()
}
if _e3_control_config != nd_experiment["model_config"]:
    raise RuntimeError("Samakan CONFIG dengan STEP 32. Kedalaman diubah pada salinan model.")
for _e3_key in (
    "learning_rate", "backbone_learning_rate", "weight_decay", "gradient_clip",
    "score_threshold", "iou_threshold", "tiny_ap50_floor",
    "tiny_localization_recall_floor",
):
    if TRAIN_CONFIG[_e3_key] != nd_experiment["train_config"][_e3_key]:
        raise RuntimeError(f"TRAIN_CONFIG[{_e3_key!r}] berubah sejak STEP 32.")

E3_EPOCHS = 100
E3_INDICES = [int(i) for i in ND_INDICES]
if len(E3_INDICES) != 10 or len(set(E3_INDICES)) != 10:
    raise ValueError("Diperlukan 10 indeks episode berbeda seperti STEP 32.")
_e3_source_hash = _nd_fingerprint(model)
if _e3_source_hash != nd_experiment["source_state_sha256"]:
    raise RuntimeError("model bukan lagi bobot awal STEP 32.")
_e3_baseline_hash = _nd_fingerprint(no_dropout_model)


def _e3_reduce_encoder(target_model, reference_model):
    if target_model is reference_model:
        raise RuntimeError("Perubahan kedalaman harus dilakukan pada salinan model.")
    if len(target_model.transformer_encoder.layers) != 6:
        raise RuntimeError("Model awal harus mempunyai 6 layer encoder.")
    if len(target_model.transformer_decoder.layers) != 6:
        raise RuntimeError("Decoder harus tetap 6 layer.")
    if target_model.transformer_decoder.num_queries != 100:
        raise RuntimeError("Jumlah object query harus tetap 100.")

    # Pertahankan bobot awal layer 1-3; lepaskan layer 4-6.
    # Tidak membuat layer baru dan tidak mengacak ulang bobot.
    target_model.transformer_encoder.layers = nn.ModuleList(
        list(target_model.transformer_encoder.layers)[:3]
    )
    removed_prefixes = tuple(f"transformer_encoder.layers.{i}." for i in (3, 4, 5))
    reference = reference_model.state_dict()
    current = target_model.state_dict()
    expected = {name for name in reference if not name.startswith(removed_prefixes)}
    if set(current) != expected:
        raise RuntimeError("Struktur berubah di luar penghapusan encoder layer 4-6.")
    changed = [
        name for name in current
        if not torch.equal(current[name].detach().cpu(), reference[name].detach().cpu())
    ]
    if changed:
        raise RuntimeError("Bobot yang seharusnya dipertahankan berubah: " + ", ".join(changed))
    return {
        "encoder_layers": 3, "decoder_layers": 6, "num_queries": 100,
        "retained_encoder_indices": [0, 1, 2],
        "removed_state_keys": sorted(set(reference) - set(current)),
        "retained_weights_identical": True,
        "original_parameters": sum(p.numel() for p in reference_model.parameters()),
        "trial_parameters": sum(p.numel() for p in target_model.parameters()),
    }


def _e3_print(epoch, report, target_model, train_stats=None):
    enc = f"encoder_{len(target_model.transformer_encoder.layers)}"
    dec = f"decoder_{len(target_model.transformer_decoder.layers)}"
    train_text = "-" if train_stats is None else f'{train_stats["loss_total"]:.4f}'
    print(
        f'Epoch {epoch:3d} | train loss {train_text}'
        f' | eval loss {report["mean_loss"]["loss_total"]:.4f}'
        f' | AP50 {report["AP50_percent"]:.4f}%'
        f' | geometry {report["geometry_recall50"]:.4f}'
        f' | IoU {report["mean_best_iou"]:.4f}'
        f' | {enc} {report["diversity_by_stage"][enc]:.3e}'
        f' | {dec} {report["diversity_by_stage"][dec]:.3e}'
        f' | box std {report["box_std_across_queries"]:.3e}', flush=True,
    )


def _e3_save(path, target_model, epoch, report, optimizer=None):
    payload = {
        "model": _nd_cpu_tree(target_model.state_dict()),
        "epoch": epoch, "updates": epoch * 10,
        "report": report, "experiment": e3_experiment,
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


def _e3_groups(rows):
    result = {}
    for name, selected in (
        ("single_person", [row for row in rows if row["gt"] == 1]),
        ("multiple_people", [row for row in rows if row["gt"] > 1]),
    ):
        total_gt = sum(row["gt"] for row in selected)
        hits = sum(row["geometry_hits50"] for row in selected)
        result[name] = {
            "episodes": len(selected), "total_gt": total_gt,
            "geometry_hits50": hits, "geometry_recall50": hits / max(total_gt, 1),
            "mean_best_iou": sum(row["mean_best_iou"] * row["gt"] for row in selected)
            / max(total_gt, 1),
        }
    return result


def _e3_check_metrics(records, reference_metrics):
    measured = compute_person_metrics(
        records, TRAIN_CONFIG["score_threshold"], TRAIN_CONFIG["iou_threshold"]
    )
    if measured["total_gt"] != reference_metrics["total_gt"]:
        raise RuntimeError("Jumlah GT berbeda dari evaluasi pembanding.")
    for key in ("person_ap50", "one_to_one_localization_recall50", "mean_best_iou"):
        if not np.isclose(measured[key], reference_metrics[key], rtol=1e-3, atol=1e-4):
            raise RuntimeError(f"Evaluasi per gambar tidak cocok untuk {key}.")


def _e3_visualize(records, folder):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected_episodes = {E3_INDICES[i] for i in (0, 1, 2, 5)}
    for record in records:
        index = record["episode_index"]
        if index not in selected_episodes:
            continue
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
        axes[1].set_title("Encoder 3 / Decoder 6, epoch 100: top-10 tanpa cutoff")
        fig.tight_layout()
        fig.savefig(Path(folder) / f"episode_{index}_final.png", dpi=140)
        plt.show()
        plt.close(fig)


E3_RUN_DIR = Path(CHECKPOINT_DIR) / "tiny_encoder3_decoder6" / datetime.now(
    timezone.utc
).strftime("%Y%m%d_%H%M%S_%f")
E3_RUN_DIR.mkdir(parents=True, exist_ok=False)
e3_model, e3_optimizer = None, None
e3_history, e3_best_epoch = [], 0
E3_TINY_GATE_PASSED = False
_e3_old_epoch = train_dataset.epoch

try:
    train_dataset.set_epoch(0)
    e3_manifest = []
    for index in E3_INDICES:
        item = train_dataset[index]
        e3_manifest.append({
            "episode_index": index,
            "support_annotation_id": int(item["support_target"]["annotation_id"].item()),
            "support_image_id": int(item["support_target"]["image_id"].item()),
            "query_image_id": int(item["query_target"]["image_id"].item()),
        })
    if e3_manifest != nd_manifest or e3_manifest != nd_experiment["manifest"]:
        raise RuntimeError("Episode support/query berbeda dari STEP 32.")

    # Evaluasi pembanding per gambar; tidak ada backward atau optimizer update.
    print("Memeriksa pembanding STEP 32 per gambar...", flush=True)
    _e3_baseline_device = next(no_dropout_model.parameters()).device
    try:
        no_dropout_model.to(CONFIG["device"])
        e3_reference_predictions, e3_reference_rows = _xi_collect_predictions(
            no_dropout_model, e3_manifest
        )
    finally:
        no_dropout_model.to(_e3_baseline_device)
    _e3_check_metrics(e3_reference_predictions, nd_summary["final_eval"]["metrics"])
    _nd_write_json(E3_RUN_DIR / "step32_predictions_final.json", {
        "epoch": 100, "box_format": "normalized_cxcywh",
        "score_threshold": TRAIN_CONFIG["score_threshold"],
        "episodes": e3_reference_predictions,
    })

    e3_model = make_trial_model().cpu()
    if _nd_fingerprint(e3_model) != _e3_source_hash:
        raise RuntimeError("Salinan berbeda dari bobot model awal.")
    e3_structure = _e3_reduce_encoder(e3_model, model)
    e3_dropout_changes = _nd_disable_dropout(e3_model)
    e3_model.to(CONFIG["device"])
    e3_optimizer, _e3_unused_scheduler = build_optimizer_and_scheduler(e3_model)
    del _e3_unused_scheduler
    # LR konstan, sama dengan STEP 32.
    e3_loader = make_episode_loader(Subset(train_dataset, E3_INDICES), num_workers=0)
    if len(e3_loader) != 10:
        raise RuntimeError("Loader harus menghasilkan 10 update per epoch.")
    e3_lrs = [group["lr"] for group in e3_optimizer.param_groups]
    if e3_lrs != nd_experiment["learning_rates"]:
        raise RuntimeError("Learning rate tidak sama dengan STEP 32.")
    e3_experiment = {
        "name": "tiny_encoder3_decoder6", "epochs": E3_EPOCHS,
        "expected_updates": 1000, "dataset_epoch": 0,
        "source_state_sha256": _e3_source_hash,
        "structure": e3_structure, "reinitialized_weights": False,
        "control_model_config": _e3_control_config,
        "model_config": {**_e3_control_config, "num_encoder_layers": 3, "dropout": 0.0},
        "train_config": copy.deepcopy(TRAIN_CONFIG),
        "effective_dropout": 0.0, "dropout_changes": e3_dropout_changes,
        "learning_rates": e3_lrs, "scheduler_stepped": False, "manifest": e3_manifest,
    }
    _nd_write_json(E3_RUN_DIR / "experiment.json", e3_experiment)
    print("Folder hasil:", E3_RUN_DIR)
    print("Encoder 3 | decoder 6 | query 100 | bobot yang dipertahankan identik: PASS")
    print("Parameter:", e3_structure["original_parameters"], "->", e3_structure["trial_parameters"])
    print("Dropout 0 pada", len(e3_dropout_changes), "module/attention | LR:", e3_lrs)
    print("Loss awal boleh berbeda karena kedalaman encoder berubah.")

    e3_initial_report = _nd_evaluate(e3_model, e3_loader)
    e3_best_report = copy.deepcopy(e3_initial_report)
    e3_history.append({"epoch": 0, "eval": e3_initial_report})
    _e3_print(0, e3_initial_report, e3_model)
    _e3_save(E3_RUN_DIR / "initial.pth", e3_model, 0, e3_initial_report)
    _e3_save(E3_RUN_DIR / "best.pth", e3_model, 0, e3_initial_report)
    _e3_save(E3_RUN_DIR / "last.pth", e3_model, 0, e3_initial_report, e3_optimizer)

    for e3_epoch in range(1, E3_EPOCHS + 1):
        train_dataset.set_epoch(0)
        e3_train = train_detection_epoch(
            target_model=e3_model, loader=e3_loader, optimizer=e3_optimizer,
            max_steps=len(e3_loader), description=f"Encoder3 {e3_epoch}/{E3_EPOCHS}",
            show_progress=False,
        )
        if e3_train["updates"] != 10:
            raise RuntimeError("Jumlah update per epoch tidak sesuai.")
        e3_row = {"epoch": e3_epoch, "train": e3_train}
        if e3_epoch in (1, 5) or e3_epoch % 10 == 0:
            e3_report = _nd_evaluate(e3_model, e3_loader)
            e3_row["eval"] = e3_report
            _e3_print(e3_epoch, e3_report, e3_model, e3_train)
            e3_rank = lambda r: (
                r["AP50_percent"], r["geometry_recall50"], -r["mean_loss"]["loss_total"]
            )
            if e3_rank(e3_report) > e3_rank(e3_best_report):
                e3_best_report = copy.deepcopy(e3_report)
                e3_best_epoch = e3_epoch
                _e3_save(E3_RUN_DIR / "best.pth", e3_model, e3_epoch, e3_report)
            _e3_save(E3_RUN_DIR / "last.pth", e3_model, e3_epoch, e3_report, e3_optimizer)
            e3_final_report = e3_report
        e3_history.append(e3_row)
        _nd_write_json(E3_RUN_DIR / "history.json", e3_history)

    E3_TINY_GATE_PASSED = bool(
        e3_final_report["mean_loss"]["loss_total"] < e3_initial_report["mean_loss"]["loss_total"]
        and e3_final_report["metrics"]["person_ap50"] >= TRAIN_CONFIG["tiny_ap50_floor"]
        and e3_final_report["geometry_recall50"] >= TRAIN_CONFIG["tiny_localization_recall_floor"]
    )
    e3_summary = {
        "initial_eval": e3_initial_report, "final_eval": e3_final_report,
        "best_epoch": e3_best_epoch, "best_eval": e3_best_report,
        "E3_TINY_GATE_PASSED": E3_TINY_GATE_PASSED,
        "total_updates": sum(row.get("train", {}).get("updates", 0) for row in e3_history),
        "step32_reference": {
            key: nd_summary[key] for key in ("final_eval", "best_epoch", "best_eval")
        },
        "step32_per_image_final": e3_reference_rows,
        "step32_groups_final": _e3_groups(e3_reference_rows),
    }
    _nd_write_json(E3_RUN_DIR / "summary.json", e3_summary)
    e3_predictions, e3_per_image = _xi_collect_predictions(e3_model, e3_manifest)
    _e3_check_metrics(e3_predictions, e3_final_report["metrics"])
    _nd_write_json(E3_RUN_DIR / "predictions_final.json", {
        "epoch": E3_EPOCHS, "box_format": "normalized_cxcywh",
        "score_threshold": TRAIN_CONFIG["score_threshold"], "episodes": e3_predictions,
    })
    e3_summary["per_image_final_all_100_queries"] = e3_per_image
    e3_summary["groups_final"] = _e3_groups(e3_per_image)
    e3_summary["difference_from_step32"] = {
        "final_AP50_percentage_points": e3_final_report["AP50_percent"] - nd_summary["final_eval"]["AP50_percent"],
        "best_AP50_percentage_points": e3_best_report["AP50_percent"] - nd_summary["best_eval"]["AP50_percent"],
        "final_geometry_recall50": e3_final_report["geometry_recall50"] - nd_summary["final_eval"]["geometry_recall50"],
        "final_mean_best_iou": e3_final_report["mean_best_iou"] - nd_summary["final_eval"]["mean_best_iou"],
    }
    _nd_write_json(E3_RUN_DIR / "summary.json", e3_summary)
    print("\nHASIL STEP 34")
    print(json.dumps(e3_summary, indent=2, allow_nan=False))
    print("\nGEOMETRY HIT PER GAMBAR, seluruh 100 query, STEP32 -> STEP34")
    for old, new in zip(e3_reference_rows, e3_per_image):
        print(
            f'Episode {new["episode_index"]}: GT={new["gt"]}'
            f' | hit {old["geometry_hits50"]} -> {new["geometry_hits50"]}'
            f' | mean IoU {old["mean_best_iou"]:.4f} -> {new["mean_best_iou"]:.4f}'
            f' | max score {old["max_score"]:.4f} -> {new["max_score"]:.4f}'
        )
    print("Checkpoint terbaik:", E3_RUN_DIR / "best.pth", "| epoch", e3_best_epoch)
    print("Checkpoint epoch 100 + optimizer:", E3_RUN_DIR / "last.pth")
    print("Prediksi lengkap:", E3_RUN_DIR / "predictions_final.json")
    _e3_visualize(e3_predictions, E3_RUN_DIR)
finally:
    train_dataset.set_epoch(_e3_old_epoch)
    if e3_optimizer is not None:
        e3_optimizer.zero_grad(set_to_none=True)
        e3_optimizer = None
    if e3_model is not None:
        e3_model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _nd_fingerprint(model) != _e3_source_hash:
        raise RuntimeError("Bobot model awal berubah selama percobaan.")
    if _nd_fingerprint(no_dropout_model) != _e3_baseline_hash:
        raise RuntimeError("Bobot pembanding STEP 32 berubah selama percobaan.")

print("STEP 34 selesai. e3_model = epoch 100 di CPU; best.pth = epoch terbaik.")
print("Gate lama tidak diubah. STEP 29 tidak dijalankan otomatis.")
