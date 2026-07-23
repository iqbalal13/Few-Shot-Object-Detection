# ==========================================================
# STEP 28 : SetCriterion (FINAL)
# ==========================================================

class SetCriterion:
    """
    Detection Loss for Simplified Meta-DETR

    Losses:
        - Classification Loss
        - Bounding Box L1 Loss
        - Generalized IoU Loss
    """

    def __init__(self, matcher, weight_dict=None):

        self.matcher = matcher

        if weight_dict is None:
            weight_dict = {
                "loss_ce": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }

        self.weight_dict = weight_dict

    def __call__(self, outputs, targets):

        indices = self.matcher(outputs, targets)

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        total_ce = 0.0
        total_bbox = 0.0
        total_giou = 0.0

        matched_images = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):

            if len(src_idx) == 0:
                continue

            matched_images += 1

            # -------------------------------------------------
            # Classification Loss
            # -------------------------------------------------

            src_logits = pred_logits[b][src_idx]
            tgt_labels = targets[b]["labels"][tgt_idx]

            loss_ce = F.cross_entropy(
                src_logits,
                tgt_labels
            )

            # -------------------------------------------------
            # Bounding Box Loss (L1)
            # -------------------------------------------------

            src_boxes = pred_boxes[b][src_idx]
            tgt_boxes = targets[b]["boxes"][tgt_idx]

            loss_bbox = F.l1_loss(
                src_boxes,
                tgt_boxes,
                reduction="mean"
            )

            # -------------------------------------------------
            # Generalized IoU Loss
            # -------------------------------------------------

            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

            giou = generalized_box_iou(
                src_xyxy,
                tgt_xyxy
            )

            loss_giou = (
                1 - torch.diag(giou)
            ).mean()

            total_ce += loss_ce
            total_bbox += loss_bbox
            total_giou += loss_giou

        if matched_images > 0:

            total_ce /= matched_images
            total_bbox /= matched_images
            total_giou /= matched_images

        else:

            device = pred_logits.device

            total_ce = torch.tensor(
                0.0,
                device=device
            )

            total_bbox = torch.tensor(
                0.0,
                device=device
            )

            total_giou = torch.tensor(
                0.0,
                device=device
            )

        losses = {
            "loss_ce": total_ce * self.weight_dict["loss_ce"],
            "loss_bbox": total_bbox * self.weight_dict["loss_bbox"],
            "loss_giou": total_giou * self.weight_dict["loss_giou"]
        }

        return losses


print("=" * 60)
print("STEP 28 : SetCriterion (FINAL) Ready")
print("=" * 60)
