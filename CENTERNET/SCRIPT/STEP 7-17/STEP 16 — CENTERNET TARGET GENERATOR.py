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
    valid_candidate_objects = 0

    for box, class_id in zip(
        boxes,
        class_labels
    ):

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
