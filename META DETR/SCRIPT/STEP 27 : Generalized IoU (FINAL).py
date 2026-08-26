# ==========================================================
# STEP 27 : Generalized IoU (FINAL)
# ==========================================================

def generalized_box_iou(
    boxes1,
    boxes2
):

    iou, union = box_iou(
        boxes1,
        boxes2
    )


    lt = torch.min(

        boxes1[:, None, :2],

        boxes2[:, :2]
    )


    rb = torch.max(

        boxes1[:, None, 2:],

        boxes2[:, 2:]
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


    giou = (

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


    return giou


print("=" * 70)
print("STEP 27 : GIoU READY")
print("=" * 70)
