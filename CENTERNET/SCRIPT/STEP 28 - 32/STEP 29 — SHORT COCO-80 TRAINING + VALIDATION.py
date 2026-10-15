
# ============================================================
# STEP 29 — BF16 TRAINING + SINGLE-IOU-MATRIX VALIDATION
# Supports original and pre-encoded DataLoader batches.
# ============================================================

import os
import math
import torch
from torchvision.ops import box_iou

GRAD_CLIP_NORM = CONFIG["grad_clip_norm"]

RUN_ENGINEERING_SANITY = bool(
    CONFIG.get("run_engineering_sanity", False)
)

DEBUG_FINITE_CHECKS = bool(
    CONFIG.get("debug_finite_checks", False)
)

PROGRESS_EVERY = int(
    CONFIG.get("progress_every", 100)
)


def assert_outputs_finite(outputs, context=""):
    for name, tensor in outputs.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"Non-finite {name}: {context}"
            )


def _unpack_centernet_batch(batch):
    if len(batch) == 2:
        images, targets = batch
        return images, targets, None

    if len(batch) == 3:
        return batch

    raise ValueError(
        f"Unexpected batch tuple size: {len(batch)}"
    )


def _targets_gpu(raw_targets, encoded_cpu):
    if encoded_cpu is None:
        return build_batch_centernet_targets(
            raw_targets,
            device=device
        )

    return {
        key: value.to(device, non_blocking=True)
        for key, value in encoded_cpu.items()
    }


def _prepare_match_data(
    prediction,
    target,
    class_filter=None
):
    pb = prediction["boxes"].detach().cpu()
    ps = prediction["scores"].detach().cpu()
    pl = prediction["labels"].detach().cpu()

    gb = target["boxes"].detach().cpu()
    gl = target["labels"].detach().cpu()

    if class_filter is not None:
        pm = pl == class_filter
        gm = gl == class_filter

        pb, ps, pl = pb[pm], ps[pm], pl[pm]
        gb, gl = gb[gm], gl[gm]

    # Compute IoU only once per image
    if pb.numel() and gb.numel():
        matrix = box_iou(pb, gb)
    else:
        matrix = torch.empty(
            (len(pb), len(gb))
        )

    return ps, pl, gl, matrix


def _greedy_from_cached(
    ps,
    pl,
    gl,
    matrix,
    score_threshold,
    iou_threshold
):
    selected = torch.where(
        ps >= score_threshold
    )[0]

    # Preserve descending confidence matching
    if selected.numel():
        selected = selected[
            torch.argsort(
                ps[selected],
                descending=True
            )
        ]

    used = set()
    tp = 0
    fp = 0

    for pred_idx in selected.tolist():

        candidate = [
            j
            for j in range(len(gl))
            if j not in used
            and int(gl[j]) == int(pl[pred_idx])
        ]

        if not candidate:
            fp += 1
            continue

        best_value, local_idx = torch.max(
            matrix[pred_idx, candidate],
            dim=0
        )

        if float(best_value) >= iou_threshold:
            used.add(
                candidate[int(local_idx)]
            )
            tp += 1
        else:
            fp += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": len(gl) - tp,
        "gt": len(gl)
    }


def greedy_detection_counts(
    prediction,
    target,
    score_threshold=0.50,
    iou_threshold=0.50,
    class_filter=None
):
    """
    Backward-compatible evaluator API.
    """
    cache = _prepare_match_data(
        prediction,
        target,
        class_filter
    )

    return _greedy_from_cached(
        *cache,
        score_threshold,
        iou_threshold
    )


def evaluate_centernet(
    model,
    loader,
    max_batches=None,
    score_threshold=0.50,
    iou_threshold=0.50,
    class_filter=None,
    compute_loss=True
):
    model.eval()

    total_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float32
    )

    loss_batches = 0

    tp = 0
    fp = 0
    fn = 0

    geometry_matches = 0
    geometry_gt = 0

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            images, targets, encoded_cpu = (
                _unpack_centernet_batch(batch)
            )

            images = images.to(
                device,
                non_blocking=True
            )

            if DEBUG_FINITE_CHECKS:
                if not bool(torch.isfinite(images).all()):
                    raise FloatingPointError(
                        "Invalid validation input"
                    )

            # BF16 forward
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                outputs = model(images)

            if DEBUG_FINITE_CHECKS:
                assert_outputs_finite(
                    outputs,
                    f"validation batch {batch_index}"
                )

            # FP32 loss
            if compute_loss:

                encoded = _targets_gpu(
                    targets,
                    encoded_cpu
                )

                loss = criterion(
                    outputs,
                    encoded
                )["loss_total"]

                if DEBUG_FINITE_CHECKS:
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            "Invalid validation loss"
                        )

                total_loss += loss.detach().float()
                loss_batches += 1

            # Decode in FP32
            decode_outputs = {
                name: tensor.float()
                for name, tensor in outputs.items()
            }

            predictions = decode_centernet(
                decode_outputs,
                K=100,
                score_threshold=0.0
            )

            # Metrics
            for prediction, target in zip(
                predictions,
                targets
            ):
                cached = _prepare_match_data(
                    prediction,
                    target,
                    class_filter
                )

                primary = _greedy_from_cached(
                    *cached,
                    score_threshold,
                    iou_threshold
                )

                geometry = _greedy_from_cached(
                    *cached,
                    0.0,
                    iou_threshold
                )

                tp += primary["tp"]
                fp += primary["fp"]
                fn += primary["fn"]

                geometry_matches += geometry["tp"]
                geometry_gt += geometry["gt"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    geometry50 = (
        geometry_matches
        / max(geometry_gt, 1)
    )

    if compute_loss:
        mean_loss = float(
            (
                total_loss / max(loss_batches, 1)
            ).detach().cpu()
        )
    else:
        mean_loss = None

    return {
        "loss": mean_loss,
        "precision": precision,
        "recall": recall,
        "geometry50": geometry50,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": geometry_gt
    }


def train_centernet_epoch(
    model,
    loader,
    optimizer,
    max_batches=None
):
    model.train()
    set_backbone_bn_eval(model)

    metric_sums = torch.zeros(
        4,
        device=device,
        dtype=torch.float32
    )

    batches = 0

    for batch_index, batch in enumerate(loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images, targets, encoded_cpu = (
            _unpack_centernet_batch(batch)
        )

        images = images.to(
            device,
            non_blocking=True
        )

        encoded_targets = _targets_gpu(
            targets,
            encoded_cpu
        )

        if DEBUG_FINITE_CHECKS:
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError(
                    "Invalid train input"
                )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            outputs = model(images)

        if DEBUG_FINITE_CHECKS:
            assert_outputs_finite(
                outputs,
                f"train batch {batch_index}"
            )

        losses = criterion(
            outputs,
            encoded_targets
        )

        total = losses["loss_total"]

        if DEBUG_FINITE_CHECKS:
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(
                    "Invalid train loss"
                )

        total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
            error_if_nonfinite=True
        )

        optimizer.step()

        metric_sums += torch.stack([
            total.detach().float(),
            losses["loss_heatmap"].detach().float(),
            losses["loss_wh"].detach().float(),
            losses["loss_offset"].detach().float()
        ])

        batches += 1

        if (
            batches == 1
            or batches % PROGRESS_EVERY == 0
        ):
            print(
                f"  batch {batches:05d}/{len(loader):05d} "
                f"| loss {float(total.detach()):.4f} "
                f"| grad_norm {float(grad_norm.detach()):.4f}"
            )

    if batches == 0:
        raise RuntimeError(
            "No training batch processed"
        )

    averages = (
        metric_sums / batches
    ).detach().cpu().tolist()

    return {
        "loss": averages[0],
        "heatmap": averages[1],
        "wh": averages[2],
        "offset": averages[3],
        "batches": batches
    }


# ============================================================
# OPTIONAL SHORT TRAINING
# ============================================================

if not RUN_ENGINEERING_SANITY:
    print("RUN_ENGINEERING_SANITY = False")
    print("Short COCO-80 training/validation SKIPPED.")
    print("Evaluator and trainer are ready.")
    print("STEP 29 PASSED")

else:
    SHORT_EPOCHS = 2
    SHORT_TRAIN_BATCHES = 300
    SHORT_VAL_BATCHES = 100

    short_model = build_fresh_source_model()

    short_optimizer = build_source_optimizer(
        short_model
    )

    initial_val = evaluate_centernet(
        short_model,
        val_loader,
        max_batches=SHORT_VAL_BATCHES,
        score_threshold=CONFIG["score_threshold"],
        iou_threshold=CONFIG["iou_threshold"]
    )

    print(
        "Initial Geometry50:",
        initial_val["geometry50"]
    )

    best_short_geometry = -1.0

    SHORT_CHECKPOINT = os.path.join(
        CONFIG["checkpoint_root"],
        "centernet_coco80_short_best_fast_repro_v2_bs8.pth"
    )

    for epoch in range(1, SHORT_EPOCHS + 1):

        train_generator.manual_seed(
            SEED + 2900 + epoch
        )

        train_stats = train_centernet_epoch(
            short_model,
            train_loader,
            short_optimizer,
            max_batches=SHORT_TRAIN_BATCHES
        )

        val_stats = evaluate_centernet(
            short_model,
            val_loader,
            max_batches=SHORT_VAL_BATCHES,
            score_threshold=CONFIG["score_threshold"],
            iou_threshold=CONFIG["iou_threshold"]
        )

        print(
            "Short epoch:",
            epoch,
            "train loss:",
            train_stats["loss"],
            "Geometry50:",
            val_stats["geometry50"]
        )

        if val_stats["geometry50"] > best_short_geometry:

            best_short_geometry = val_stats["geometry50"]

            torch.save(
                {
                    "model_state_dict": short_model.state_dict(),
                    "epoch": epoch,
                    "val_stats": val_stats,
                    "precision_mode": "bfloat16",
                    "experiment_name": CONFIG["experiment_name"]
                },
                SHORT_CHECKPOINT
            )

    print("STEP 29 SANITY PASSED")
