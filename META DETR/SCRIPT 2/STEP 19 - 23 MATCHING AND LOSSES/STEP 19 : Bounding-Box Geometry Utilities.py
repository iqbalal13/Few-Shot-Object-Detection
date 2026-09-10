# ==========================================================
# STEP 19 : Bounding-Box Geometry Utilities
# ==========================================================


def box_cxcywh_to_xyxy(
    boxes
):

    cx, cy, w, h = (
        boxes.unbind(-1)
    )


    return torch.stack(

        (
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h
        ),

        dim=-1
    )


def box_iou(
    boxes1,
    boxes2
):

    area1 = (

        (
            boxes1[:, 2]
            -
            boxes1[:, 0]
        ).clamp(min=0)

        *

        (
            boxes1[:, 3]
            -
            boxes1[:, 1]
        ).clamp(min=0)
    )


    area2 = (

        (
            boxes2[:, 2]
            -
            boxes2[:, 0]
        ).clamp(min=0)

        *

        (
            boxes2[:, 3]
            -
            boxes2[:, 1]
        ).clamp(min=0)
    )


    lt = torch.max(

        boxes1[
            :,
            None,
            :2
        ],

        boxes2[
            :,
            :2
        ]
    )


    rb = torch.min(

        boxes1[
            :,
            None,
            2:
        ],

        boxes2[
            :,
            2:
        ]
    )


    wh = (
        rb - lt
    ).clamp(
        min=0
    )


    intersection = (

        wh[:, :, 0]
        *
        wh[:, :, 1]
    )


    union = (

        area1[:, None]
        +
        area2
        -
        intersection
    )


    iou = (

        intersection

        /

        union.clamp(
            min=1e-6
        )
    )


    return (
        iou,
        union
    )


def generalized_box_iou(
    boxes1,
    boxes2
):

    iou, union = box_iou(
        boxes1,
        boxes2
    )


    lt = torch.min(

        boxes1[
            :,
            None,
            :2
        ],

        boxes2[
            :,
            :2
        ]
    )


    rb = torch.max(

        boxes1[
            :,
            None,
            2:
        ],

        boxes2[
            :,
            2:
        ]
    )


    wh = (
        rb - lt
    ).clamp(
        min=0
    )


    enclosing_area = (

        wh[:, :, 0]
        *
        wh[:, :, 1]
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
            min=1e-6
        )
    )


# ==========================================================
# QUICK SANITY
# ==========================================================

test_box = torch.tensor(
    [[0.5, 0.5, 0.2, 0.2]]
)


test_xyxy = (
    box_cxcywh_to_xyxy(
        test_box
    )
)


test_iou, _ = box_iou(
    test_xyxy,
    test_xyxy
)


test_giou = generalized_box_iou(
    test_xyxy,
    test_xyxy
)


assert torch.allclose(

    test_iou,

    torch.ones_like(
        test_iou
    )
)


assert torch.allclose(

    test_giou,

    torch.ones_like(
        test_giou
    )
)


print("=" * 70)
print("✓ STEP 19 BOX GEOMETRY PASSED")
print("=" * 70)
