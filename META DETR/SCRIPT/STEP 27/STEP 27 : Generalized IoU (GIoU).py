# ==========================================================
# STEP 27 : Generalized IoU (GIoU)
# ==========================================================

def generalized_box_iou(boxes1, boxes2):
    """
    Compute pairwise Generalized IoU (GIoU).
    Boxes must be in (x1, y1, x2, y2) format.
    """

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / (area + 1e-6)

    return giou


print("=" * 60)
print("Generalized IoU Ready")
print("=" * 60)
