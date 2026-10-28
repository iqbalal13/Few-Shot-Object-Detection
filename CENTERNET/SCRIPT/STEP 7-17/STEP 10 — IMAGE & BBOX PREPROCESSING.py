
# ============================================================
# STEP 10 — ORIGINAL LETTERBOX +
# TRAIN-ONLY GEOMETRIC AUGMENTATION
# ============================================================

from PIL import Image, ImageEnhance
import random
import numpy as np
import torch

INPUT_SIZE = CONFIG["input_size"]

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def _normalize_pil(img):
    arr = np.asarray(
        img, dtype=np.float32
    ) / 255.0

    arr = (
        arr - IMAGENET_MEAN
    ) / IMAGENET_STD

    return torch.from_numpy(
        arr
    ).permute(2, 0, 1).float()


def preprocess_image_and_boxes(
    image,
    boxes,
    input_size=INPUT_SIZE
):
    # Original deterministic validation geometry.
    original_w, original_h = image.size

    scale = min(
        input_size / original_w,
        input_size / original_h
    )

    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    resized = image.resize(
        (resized_w, resized_h),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (input_size, input_size),
        color=(114, 114, 114)
    )

    pad_x = (input_size - resized_w) // 2
    pad_y = (input_size - resized_h) // 2

    canvas.paste(
        resized, (pad_x, pad_y)
    )

    boxes = np.asarray(
        boxes, dtype=np.float32
    ).reshape(-1, 4)

    new = boxes.copy()

    if len(new):
        new[:, [0, 2]] = np.clip(
            new[:, [0, 2]] * scale + pad_x,
            0, input_size - 1
        )

        new[:, [1, 3]] = np.clip(
            new[:, [1, 3]] * scale + pad_y,
            0, input_size - 1
        )

    meta = {
        "original_size": (original_h, original_w),
        "resized_size": (resized_h, resized_w),
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }

    return _normalize_pil(canvas), new, meta


def preprocess_train_augmented(
    image,
    boxes,
    input_size=INPUT_SIZE
):
    # Scale + translation are applied to both image and GT.
    original_w, original_h = image.size

    base_scale = min(
        input_size / original_w,
        input_size / original_h
    )

    scale = base_scale * random.uniform(
        CONFIG["train_scale_min"],
        CONFIG["train_scale_max"]
    )

    rw = max(
        1, int(round(original_w * scale))
    )

    rh = max(
        1, int(round(original_h * scale))
    )

    resized = image.resize(
        (rw, rh),
        Image.Resampling.BILINEAR
    )

    shift = round(
        input_size * CONFIG["train_translate_frac"]
    )

    px = (
        (input_size - rw) // 2
        + random.randint(-shift, shift)
    )

    py = (
        (input_size - rh) // 2
        + random.randint(-shift, shift)
    )

    canvas = Image.new(
        "RGB",
        (input_size, input_size),
        color=(114, 114, 114)
    )

    # PIL clips image parts outside canvas.
    canvas.paste(resized, (px, py))

    arr = np.asarray(
        boxes, dtype=np.float32
    ).reshape(-1, 4).copy()

    if len(arr):
        arr[:, [0, 2]] = (
            arr[:, [0, 2]] * scale + px
        )

        arr[:, [1, 3]] = (
            arr[:, [1, 3]] * scale + py
        )

        uncut = (
            np.maximum(0, arr[:, 2] - arr[:, 0])
            * np.maximum(0, arr[:, 3] - arr[:, 1])
        )

        arr[:, [0, 2]] = np.clip(
            arr[:, [0, 2]],
            0, input_size - 1
        )

        arr[:, [1, 3]] = np.clip(
            arr[:, [1, 3]],
            0, input_size - 1
        )

        visible_area = (
            np.maximum(0, arr[:, 2] - arr[:, 0])
            * np.maximum(0, arr[:, 3] - arr[:, 1])
        )

        visible_fraction = (
            visible_area / np.maximum(uncut, 1e-7)
        )

        keep = (
            (visible_fraction >= CONFIG["train_visible_box_min"])
            & ((arr[:, 2] - arr[:, 0]) > 1)
            & ((arr[:, 3] - arr[:, 1]) > 1)
        )
    else:
        keep = np.zeros(
            (0,), dtype=bool
        )

    meta = {
        "original_size": (original_h, original_w),
        "resized_size": (rh, rw),
        "scale": scale,
        "pad_x": px,
        "pad_y": py,
    }

    return (
        _normalize_pil(canvas),
        arr,
        meta,
        keep
    )


def denormalize_image(image_tensor):
    arr = (
        image_tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    return np.clip(
        arr * IMAGENET_STD + IMAGENET_MEAN,
        0.0, 1.0
    )


print(
    "STEP 10 PASS — validation unchanged; "
    "jointly transformed training geometry"
)
