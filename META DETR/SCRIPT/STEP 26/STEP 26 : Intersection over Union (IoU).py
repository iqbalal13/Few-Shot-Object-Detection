# ==========================================================
# STEP 26 : Intersection over Union (IoU)
# ==========================================================

def box_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes.
    Boxes must be in (x1, y1, x2, y2) format.
    """

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou, union


print("=" * 60)
print("IoU Function Ready")
print("=" * 60)
