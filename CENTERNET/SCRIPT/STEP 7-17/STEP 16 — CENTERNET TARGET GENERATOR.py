# ============================================================
# STEP 16 — CENTERNET TARGET GENERATOR
# ============================================================

print("=" * 70)
print("STEP 16 — CENTERNET TARGET GENERATOR")
print("=" * 70)

CONFIG["max_objects"] = 128

MAX_OBJECTS = CONFIG["max_objects"]
NUM_CLASSES = CONFIG["num_classes"]


def build_centernet_target(
    target,
    num_classes=NUM_CLASSES,
    output_height=OUTPUT_HEIGHT,
    output_width=OUTPUT_WIDTH,
    output_stride=OUTPUT_STRIDE,
    max_objects=MAX_OBJECTS
):

    heatmap = torch.zeros(
        (
            num_classes,
            output_height,
            output_width
        ),
        dtype=torch.float32
    )

    wh_target = torch.zeros(
        (max_objects, 2),
        dtype=torch.float32
    )

    offset_target = torch.zeros(
        (max_objects, 2),
        dtype=torch.float32
    )

    indices = torch.zeros(
        (max_objects,),
        dtype=torch.long
    )

    mask = torch.zeros(
        (max_objects,),
        dtype=torch.bool
    )

    labels = torch.full(
        (max_objects,),
        -1,
        dtype=torch.long
    )

    boxes = target["boxes"]
    class_labels = target["labels"]

    object_index = 0

    for box, class_id in zip(
        boxes,
        class_labels
    ):

        if object_index >= max_objects:
            break

        x1, y1, x2, y2 = box.tolist()

        width = x2 - x1
        height = y2 - y1

        if width <= 1 or height <= 1:
            continue

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        center_x /= output_stride
        center_y /= output_stride

        width /= output_stride
        height /= output_stride

        if (
            center_x < 0
            or center_x >= output_width
            or center_y < 0
            or center_y >= output_height
        ):
            continue

        center_int_x = int(
            center_x
        )

        center_int_y = int(
            center_y
        )

        center_int_x = min(
            max(center_int_x, 0),
            output_width - 1
        )

        center_int_y = min(
            max(center_int_y, 0),
            output_height - 1
        )

        radius = gaussian_radius(
            height,
            width,
            min_overlap=0.7
        )

        radius = max(
            0,
            int(radius)
        )

        class_id = int(
            class_id.item()
        )

        draw_gaussian(
            heatmap[class_id],
            (
                center_int_x,
                center_int_y
            ),
            radius
        )

        wh_target[
            object_index
        ] = torch.tensor(
            [width, height],
            dtype=torch.float32
        )

        offset_target[
            object_index
        ] = torch.tensor(
            [
                center_x - center_int_x,
                center_y - center_int_y
            ],
            dtype=torch.float32
        )

        indices[
            object_index
        ] = (
            center_int_y
            * output_width
            + center_int_x
        )

        mask[
            object_index
        ] = True

        labels[
            object_index
        ] = class_id

        object_index += 1

    return {
        "heatmap": heatmap,
        "wh": wh_target,
        "offset": offset_target,
        "indices": indices,
        "mask": mask,
        "labels": labels,
        "num_objects": object_index,
    }


# Test target generator
sample_image, sample_target = train_dataset[0]

centernet_target = build_centernet_target(
    sample_target
)

print(
    "Heatmap :",
    centernet_target["heatmap"].shape
)

print(
    "WH      :",
    centernet_target["wh"].shape
)

print(
    "Offset  :",
    centernet_target["offset"].shape
)

print(
    "Indices :",
    centernet_target["indices"].shape
)

print(
    "Mask    :",
    centernet_target["mask"].shape
)

print(
    "Objects :",
    centernet_target["num_objects"]
)

print(
    "Heatmap max :",
    centernet_target["heatmap"].max().item()
)

assert centernet_target["heatmap"].shape == (
    80,
    160,
    160
)

assert centernet_target["wh"].shape == (
    MAX_OBJECTS,
    2
)

assert centernet_target["offset"].shape == (
    MAX_OBJECTS,
    2
)

print("\nSTEP 16 PASSED")
