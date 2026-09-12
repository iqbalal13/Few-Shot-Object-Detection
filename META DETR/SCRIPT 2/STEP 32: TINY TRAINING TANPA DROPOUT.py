# ==========================================================
# STEP 32: TINY TRAINING TANPA DROPOUT
# Satu percobaan: dropout 0, bobot awal dan data tetap sama.
# Jalankan setelah STEP 31. Runtime baru: jalankan STEP 1-26.
# ==========================================================

import copy
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

_nd_required = (
    "model", "train_dataset", "CONFIG", "COCO_CONFIG", "TRAIN_CONFIG",
    "CHECKPOINT_DIR", "LOG_DIR", "criterion", "make_episode_loader",
    "make_trial_model", "build_optimizer_and_scheduler",
    "train_detection_epoch", "evaluate_person", "as_numpy",
)
_nd_missing = [name for name in _nd_required if name not in globals()]
if _nd_missing:
    raise RuntimeError(
        "Dependency belum tersedia: " + ", ".join(_nd_missing)
        + ". Pada runtime baru, jalankan STEP 1-26 terlebih dahulu."
    )
if COCO_CONFIG["batch_size"] != 1 or COCO_CONFIG["num_classes"] != 1:
    raise ValueError("Percobaan ini memakai batch_size=1 dan person-only.")
if CONFIG["image_size"] != 640 or TRAIN_CONFIG["iou_threshold"] != 0.5:
    raise ValueError("Gunakan resolusi 640 dan IoU 0.5 seperti tiny sebelumnya.")

ND_EPOCHS = 100
ND_INDICES = [int(i) for i in globals().get("tiny_indices", range(10))]
if len(ND_INDICES) != 10 or len(set(ND_INDICES)) != 10:
    raise ValueError("Diperlukan tepat 10 indeks episode tiny yang berbeda.")
if min(ND_INDICES) < 0 or max(ND_INDICES) >= len(train_dataset):
    raise ValueError("Indeks tiny tidak sesuai dengan train_dataset.")


def _nd_fingerprint(target_model):
    digest = hashlib.sha256()
    for name, tensor in target_model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _nd_write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def _nd_disable_dropout(target_model):
    changes = []
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout,
    )
    for name, module in target_model.named_modules():
        if isinstance(module, dropout_types):
            changes.append({"module": name, "attribute": "p", "old": float(module.p)})
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            changes.append({"module": name, "attribute": "dropout", "old": float(module.dropout)})
            module.dropout = 0.0
    if not changes or not any(row["old"] > 0 for row in changes):
        raise RuntimeError("Model awal tidak memiliki dropout aktif untuk dibandingkan.")
    return changes


def _nd_evaluate(target_model, loader):
    stage_values, loss_values = {}, {}
    box_spreads, logit_spreads, handles = [], [], []
    old_modes = [(module, module.training) for module in target_model.modules()]

    def stage_hook(name):
        def capture(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            x = x.detach().float()
            centered = x - x.mean(dim=1, keepdim=True)
            diversity = (
                centered.square().mean().sqrt()
                / x.square().mean().sqrt().clamp_min(1e-12)
            ).item()
            stage_values.setdefault(name, []).append(diversity)
        return capture

    def prediction_hook(module, inputs, output):
        box_spreads.append(output["pred_boxes"].std(dim=1, unbiased=False).mean().item())
        logit_spreads.append(output["pred_logits"].std(dim=1, unbiased=False).mean().item())

    def loss_hook(module, inputs, output):
        for name, value in output.items():
            loss_values.setdefault(name, []).append(value.item())

    stages = [("query_tokens", target_model.query_encoder)]
    stages += [(f"encoder_{i + 1}", layer) for i, layer in enumerate(target_model.transformer_encoder.layers)]
    stages += [(f"decoder_{i + 1}", layer) for i, layer in enumerate(target_model.transformer_decoder.layers)]
    stages.append(("relation", target_model.relation_module))
    try:
        for name, module in stages:
            handles.append(module.register_forward_hook(stage_hook(name)))
        handles.append(target_model.register_forward_hook(prediction_hook))
        handles.append(criterion.register_forward_hook(loss_hook))
        # Evaluasi tidak menggeser RNG training, termasuk seed iterator DataLoader.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            metrics = evaluate_person(target_model, loader)
        return {
            "mean_loss": {k: float(np.mean(v)) for k, v in loss_values.items()},
            "diversity_by_stage": {k: float(np.mean(v)) for k, v in stage_values.items()},
            "box_std_across_queries": float(np.mean(box_spreads)),
            "logit_std_across_queries": float(np.mean(logit_spreads)),
            "AP50_percent": 100.0 * metrics["person_ap50"],
            "geometry_recall50": metrics["one_to_one_localization_recall50"],
            "mean_best_iou": metrics["mean_best_iou"],
            "metrics": metrics,
        }
    finally:
        for handle in handles:
            handle.remove()
        for module, training in old_modes:
            module.training = training


def _nd_print(epoch, report, train_stats=None):
    div = report["diversity_by_stage"]
    enc_key = f"encoder_{len(no_dropout_model.transformer_encoder.layers)}"
    dec_key = f"decoder_{len(no_dropout_model.transformer_decoder.layers)}"
    train_text = "-" if train_stats is None else f'{train_stats["loss_total"]:.4f}'
    print(
        f'Epoch {epoch:3d} | train loss {train_text}'
        f' | eval loss {report["mean_loss"]["loss_total"]:.4f}'
        f' | AP50 {report["AP50_percent"]:.4f}%'
        f' | geometry {report["geometry_recall50"]:.4f}'
        f' | IoU {report["mean_best_iou"]:.4f}'
        f' | enc {div[enc_key]:.3e} | dec {div[dec_key]:.3e}'
        f' | box std {report["box_std_across_queries"]:.3e}',
        flush=True,
    )


def _nd_cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _nd_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nd_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_nd_cpu_tree(item) for item in value)
    return value


def _nd_save_checkpoint(filename, epoch, report, optimizer=None):
    payload = {
        "model": _nd_cpu_tree(no_dropout_model.state_dict()),
        "epoch": epoch, "updates": epoch * len(nd_loader),
        "report": report, "experiment": nd_experiment,
    }
    if optimizer is not None:
        payload["optimizer"] = _nd_cpu_tree(optimizer.state_dict())
        payload["torch_rng_state"] = torch.get_rng_state()
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    path = ND_RUN_DIR / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _nd_visualize(target_model, indices):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    target_model.eval()
    for index in indices:
        episode = train_dataset[index]
        with torch.inference_mode():
            output = target_model(
                episode["support_image"].unsqueeze(0).to(CONFIG["device"]),
                episode["query_image"].unsqueeze(0).to(CONFIG["device"]),
            )
        rgb = episode["query_image"].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        height, width = rgb.shape[:2]
        gt = as_numpy(episode["query_target"]["boxes"])
        boxes = as_numpy(output["pred_boxes"][0])
        scores = as_numpy(output["pred_logits"][0, :, 0].sigmoid())
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
                axis.add_patch(Rectangle((x, y), w * width, h * height, fill=False, edgecolor=color, linewidth=1.5))
                if axis is axes[1]:
                    axis.text(x, y, f"{scores[selected[j]]:.3f}", color="white", backgroundcolor="black", fontsize=8)
        axes[0].set_title(f"Episode {index}: GT ({len(gt)} person)")
        axes[1].set_title("Epoch 100: top-10 prediksi, tanpa filter skor")
        fig.tight_layout()
        fig.savefig(ND_RUN_DIR / f"episode_{index}_final.png", dpi=140)
        plt.show()
        plt.close(fig)


# Manifest diperiksa sebelum training agar support/query tidak berganti.
_nd_old_epoch = train_dataset.epoch
try:
    train_dataset.set_epoch(0)
    nd_manifest = []
    for index in ND_INDICES:
        item = train_dataset[index]
        nd_manifest.append({
            "episode_index": index,
            "support_annotation_id": int(item["support_target"]["annotation_id"].item()),
            "support_image_id": int(item["support_target"]["image_id"].item()),
            "query_image_id": int(item["query_target"]["image_id"].item()),
        })
finally:
    train_dataset.set_epoch(_nd_old_epoch)

_nd_previous_manifest = globals().get("tiny_manifest")
_nd_manifest_path = Path(LOG_DIR) / "tiny_person_manifest.json"
if _nd_previous_manifest is None and _nd_manifest_path.exists():
    _nd_previous_manifest = json.loads(_nd_manifest_path.read_text())
if _nd_previous_manifest is not None and nd_manifest != _nd_previous_manifest:
    raise RuntimeError("Episode berbeda dari tiny sebelumnya. Periksa seed dan dataset sebelum melanjutkan.")

_nd_source_hash = _nd_fingerprint(model)
no_dropout_model = make_trial_model()
nd_dropout_changes = _nd_disable_dropout(no_dropout_model)
if _nd_fingerprint(no_dropout_model) != _nd_source_hash:
    raise RuntimeError("Bobot salinan berbeda dari model awal.")
nd_optimizer, _nd_unused_scheduler = build_optimizer_and_scheduler(no_dropout_model)
del _nd_unused_scheduler
# STEP 28 lama tidak menjalankan scheduler.step(); LR tetap di percobaan ini.
nd_loader = make_episode_loader(Subset(train_dataset, ND_INDICES), num_workers=0)
if len(nd_loader) != 10:
    raise RuntimeError("Tiny loader harus menghasilkan tepat 10 update per epoch.")

ND_RUN_DIR = Path(CHECKPOINT_DIR) / "tiny_no_dropout" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
ND_RUN_DIR.mkdir(parents=True, exist_ok=False)
nd_experiment = {
    "name": "tiny_no_dropout", "epochs": ND_EPOCHS, "expected_updates": 1000,
    "dataset_epoch": 0, "source_state_sha256": _nd_source_hash,
    "model_config": {k: str(v) if isinstance(v, torch.device) else v for k, v in CONFIG.items()},
    "train_config": copy.deepcopy(TRAIN_CONFIG),
    "effective_dropout": 0.0, "dropout_changes": nd_dropout_changes,
    "scheduler_stepped": False,
    "learning_rates": [group["lr"] for group in nd_optimizer.param_groups],
    "manifest": nd_manifest,
}
nd_history = []
nd_best_report = None
nd_best_epoch = 0
nd_final_report = None
ND_TINY_GATE_PASSED = False
print("Folder hasil:", ND_RUN_DIR)
print("Dropout dinonaktifkan pada", len(nd_dropout_changes), "module/attention.")
print("Bobot awal identik: PASS | 10 episode x 100 epoch = 1.000 update")
print("Learning rates:", nd_experiment["learning_rates"], "| scheduler tidak dijalankan")
_nd_write_json(ND_RUN_DIR / "experiment.json", nd_experiment)

try:
    train_dataset.set_epoch(0)
    nd_initial_report = _nd_evaluate(no_dropout_model, nd_loader)
    # Jika tersedia, cocokkan loss awal dengan hasil runtime sebelumnya.
    _nd_previous_initial = globals().get("tiny_initial_metrics")
    if _nd_previous_initial is not None and not np.isclose(
        nd_initial_report["mean_loss"]["loss_total"],
        _nd_previous_initial["loss"], rtol=1e-4, atol=1e-5,
    ):
        raise RuntimeError("Loss awal berbeda dari tiny lama. Bobot/data/konfigurasi perlu diperiksa.")
    nd_history.append({"epoch": 0, "eval": nd_initial_report})
    nd_best_report = copy.deepcopy(nd_initial_report)
    _nd_print(0, nd_initial_report)
    _nd_save_checkpoint("best.pth", 0, nd_initial_report)
    _nd_save_checkpoint("last.pth", 0, nd_initial_report, nd_optimizer)

    for epoch in range(1, ND_EPOCHS + 1):
        train_dataset.set_epoch(0)
        train_stats = train_detection_epoch(
            target_model=no_dropout_model, loader=nd_loader,
            optimizer=nd_optimizer, max_steps=len(nd_loader),
            description=f"No dropout {epoch}/{ND_EPOCHS}", show_progress=False,
        )
        if train_stats["updates"] != 10:
            raise RuntimeError("Jumlah update per epoch tidak sesuai.")
        row = {"epoch": epoch, "train": train_stats}
        if epoch in (1, 5) or epoch % 10 == 0:
            report = _nd_evaluate(no_dropout_model, nd_loader)
            row["eval"] = report
            _nd_print(epoch, report, train_stats)
            rank = lambda result: (result["AP50_percent"], result["geometry_recall50"], -result["mean_loss"]["loss_total"])
            if rank(report) > rank(nd_best_report):
                nd_best_report = copy.deepcopy(report)
                nd_best_epoch = epoch
                _nd_save_checkpoint("best.pth", epoch, report)
            _nd_save_checkpoint("last.pth", epoch, report, nd_optimizer)
            nd_final_report = report
        nd_history.append(row)
        _nd_write_json(ND_RUN_DIR / "history.json", nd_history)

    ND_TINY_GATE_PASSED = bool(
        nd_final_report["mean_loss"]["loss_total"] < nd_initial_report["mean_loss"]["loss_total"]
        and nd_final_report["metrics"]["person_ap50"] >= TRAIN_CONFIG["tiny_ap50_floor"]
        and nd_final_report["geometry_recall50"] >= TRAIN_CONFIG["tiny_localization_recall_floor"]
    )
    nd_summary = {
        "initial_eval": nd_initial_report,
        "final_eval": nd_final_report,
        "best_epoch": nd_best_epoch, "best_eval": nd_best_report,
        "ND_TINY_GATE_PASSED": ND_TINY_GATE_PASSED,
        "total_updates": sum(row.get("train", {}).get("updates", 0) for row in nd_history),
        "previous_step31": globals().get("diagnostic_report"),
    }
    _nd_write_json(ND_RUN_DIR / "summary.json", nd_summary)
    print("\nHASIL STEP 32")
    print(json.dumps(nd_summary, indent=2, allow_nan=False))
    print("Checkpoint terbaik:", ND_RUN_DIR / "best.pth")
    print("Checkpoint terakhir + optimizer:", ND_RUN_DIR / "last.pth")
    print("STEP 29 tidak dijalankan otomatis; nilai gate lama tidak diubah.")
    _nd_visualize(no_dropout_model, ND_INDICES[:3])
finally:
    train_dataset.set_epoch(_nd_old_epoch)
    nd_optimizer.zero_grad(set_to_none=True)
    no_dropout_model.cpu()
    del nd_optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _nd_fingerprint(model) != _nd_source_hash:
        raise RuntimeError("Bobot model awal berubah selama percobaan.")

print("STEP 32 selesai. no_dropout_model berisi bobot epoch 100 di CPU.")
