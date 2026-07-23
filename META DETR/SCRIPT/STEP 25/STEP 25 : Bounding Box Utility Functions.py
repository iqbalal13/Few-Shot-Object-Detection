# ==========================================================
# STEP 25 : Bounding Box Utility Functions
# ==========================================================

def box_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from
    (cx, cy, w, h) -> (x1, y1, x2, y2)
    """

    cx, cy, w, h = boxes.unbind(-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack((x1, y1, x2, y2), dim=-1)


print("=" * 60)
print("Bounding Box Utilities Ready")
print("=" * 60)
