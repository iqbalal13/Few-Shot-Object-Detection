# ==========================================================
# STEP 26 : Pairwise IoU (FINAL)
# ==========================================================

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

        boxes1[:, None, :2],

        boxes2[:, :2]
    )


    rb = torch.min(

        boxes1[:, None, 2:],

        boxes2[:, 2:]
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


    return iou, union


print("=" * 70)
print("STEP 26 : IoU READY")
print("=" * 70)
