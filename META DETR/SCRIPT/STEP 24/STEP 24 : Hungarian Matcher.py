# ==========================================================
# STEP 24 : Hungarian Matcher
# ==========================================================

class HungarianMatcher:
    """
    Matches predicted objects with ground truth objects
    using the Hungarian Algorithm.
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

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        batch_size = pred_logits.shape[0]

        indices = []

        for b in range(batch_size):

            out_prob = pred_logits[b].softmax(-1)
            out_bbox = pred_boxes[b]

            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            cost_class = -out_prob[:, tgt_ids]

            cost_bbox = torch.cdist(
                out_bbox,
                tgt_bbox,
                p=1
            )

            cost = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox
            )

            cost = cost.cpu()

            row_ind, col_ind = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

        return indices


print("=" * 60)
print("Hungarian Matcher Ready")
print("=" * 60)
