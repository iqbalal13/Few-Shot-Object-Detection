# ==========================================================
# STEP 19 : Bounding-Box Geometry Utilities
# ==========================================================

def box_cxcywh_to_xyxy(
    boxes
):

    if (
        boxes.shape[-1]
        !=
        4
    ):
        raise ValueError(
            "boxes last dimension "
            "must be 4."
        )

    cx, cy, w, h = (
        boxes.unbind(
            -1
        )
    )

    return torch.stack(
        (
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ),
        dim=-1,
    )


def box_xyxy_area(
    boxes
):

    wh = (
        boxes[..., 2:]
        -
        boxes[..., :2]
    ).clamp(
        min=0
    )

    return (
        wh[..., 0]
        *
        wh[..., 1]
    )


def generalized_box_iou(
    boxes1,
    boxes2
):

    if (
        boxes1.ndim != 2
        or
        boxes2.ndim != 2
        or
        boxes1.shape[-1] != 4
        or
        boxes2.shape[-1] != 4
    ):
        raise ValueError(
            "GIoU expects "
            "[N,4] and [M,4]."
        )

    area1 = (
        box_xyxy_area(
            boxes1
        )
    )

    area2 = (
        box_xyxy_area(
            boxes2
        )
    )

    lt = torch.maximum(
        boxes1[
            :, None, :2
        ],
        boxes2[
            None, :, :2
        ],
    )

    rb = torch.minimum(
        boxes1[
            :, None, 2:
        ],
        boxes2[
            None, :, 2:
        ],
    )

    wh = (
        rb - lt
    ).clamp(
        min=0
    )

    intersection = (
        wh[..., 0]
        *
        wh[..., 1]
    )

    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection
    )

    iou = (
        intersection
        /
        union.clamp(
            min=1e-7
        )
    )

    enclosing_lt = (
        torch.minimum(
            boxes1[
                :, None, :2
            ],
            boxes2[
                None, :, :2
            ],
        )
    )

    enclosing_rb = (
        torch.maximum(
            boxes1[
                :, None, 2:
            ],
            boxes2[
                None, :, 2:
            ],
        )
    )

    enclosing_wh = (
        enclosing_rb
        -
        enclosing_lt
    ).clamp(
        min=0
    )

    enclosing_area = (
        enclosing_wh[..., 0]
        *
        enclosing_wh[..., 1]
    )

    return (
        iou
        -
        (
            enclosing_area
            -
            union
        )
        /
        enclosing_area.clamp(
            min=1e-7
        )
    )


print("=" * 70)
print("STEP 19 : BBOX GEOMETRY UTILITIES READY")
print("=" * 70)
