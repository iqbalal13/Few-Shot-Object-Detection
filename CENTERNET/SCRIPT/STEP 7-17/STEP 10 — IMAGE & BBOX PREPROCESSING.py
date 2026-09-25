# ============================================================
# STEP 10 — IMAGE & BBOX PREPROCESSING
# ============================================================

from PIL import Image
import numpy as np
import torch

print("=" * 70)
print("STEP 10 — PREPROCESSING SETUP")
print("=" * 70)

INPUT_SIZE = CONFIG["input_size"]

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image_and_boxes(image, boxes, input_size=INPUT_SIZE):
    """
    image:
        PIL RGB image

    boxes:
        numpy array [N, 4]
        format xyxy in original image coordinates

    Returns:
        image_tensor : [3, input_size, input_size]
        boxes_new     : [N, 4] in resized/padded coordinates
        meta          : resize information
    """

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
        resized,
        (pad_x, pad_y)
    )

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    ).reshape(-1, 4)

    boxes_new = boxes.copy()

    if len(boxes_new) > 0:

        boxes_new[:, [0, 2]] *= scale
        boxes_new[:, [1, 3]] *= scale

        boxes_new[:, [0, 2]] += pad_x
        boxes_new[:, [1, 3]] += pad_y

        boxes_new[:, [0, 2]] = np.clip(
            boxes_new[:, [0, 2]],
            0,
            input_size - 1
        )

        boxes_new[:, [1, 3]] = np.clip(
            boxes_new[:, [1, 3]],
            0,
            input_size - 1
        )

    image_np = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    image_np = (
        image_np - IMAGENET_MEAN
    ) / IMAGENET_STD

    image_tensor = torch.from_numpy(
        image_np
    ).permute(2, 0, 1).float()

    meta = {
        "original_size": (original_h, original_w),
        "resized_size": (resized_h, resized_w),
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }

    return image_tensor, boxes_new, meta


def denormalize_image(image_tensor):

    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()

    image = (
        image * IMAGENET_STD
        + IMAGENET_MEAN
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


print("Input size     :", INPUT_SIZE)
print("Resize method  : letterbox")
print("Aspect ratio   : preserved")
print("Normalization  : ImageNet")
print("Output tensor  : [3, 640, 640]")

print("\nSTEP 10 PASSED")
