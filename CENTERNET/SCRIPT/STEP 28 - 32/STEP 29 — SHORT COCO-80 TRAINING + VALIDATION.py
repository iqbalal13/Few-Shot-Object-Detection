
# ============================================================
# STEP 29 — BF16 TRAINING + SINGLE-IOU-MATRIX VALIDATION
# Supports original and pre-encoded DataLoader batches.
# ============================================================

import os
import math
import io
import contextlib
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
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
