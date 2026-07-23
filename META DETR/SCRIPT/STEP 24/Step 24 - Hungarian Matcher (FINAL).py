# ============================================================
# Step 24 - Hungarian Matcher (FINAL)
# ============================================================

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    """
    Hungarian Matcher for Simplified Meta-DETR

    Matching Cost:
        1. Classification Cost
        2. Bounding Box L1 Cost
        3. Generalized IoU Cost
    """

    def __init__(
        self,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    ):

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]

        pred_logits = outputs["pred_logits"].softmax(-1)
        pred_boxes = outputs["pred_boxes"]

        indices = []

        for b in range(bs):

            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]

            # Jika tidak ada objek pada gambar
            if len(tgt_labels) == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # -------------------------------------------------
            # Classification Cost
            # -------------------------------------------------

            cost_class = -pred_logits[b][:, tgt_labels]

            # -------------------------------------------------
            # Bounding Box L1 Cost
            # -------------------------------------------------

            cost_bbox = torch.cdist(
                pred_boxes[b],
                tgt_boxes,
                p=1
            )

            # -------------------------------------------------
            # Generalized IoU Cost
            # -------------------------------------------------

            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

            cost_giou = -generalized_box_iou(
                pred_xyxy,
                tgt_xyxy
            )

            # -------------------------------------------------
            # Final Matching Cost
            # -------------------------------------------------

            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )

            C = C.cpu()

            src_idx, tgt_idx = linear_sum_assignment(C)

            indices.append((
                torch.as_tensor(src_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64)
            ))

        return indices


print("=" * 60)
print("HungarianMatcher (FINAL) Ready")
print("=" * 60)
