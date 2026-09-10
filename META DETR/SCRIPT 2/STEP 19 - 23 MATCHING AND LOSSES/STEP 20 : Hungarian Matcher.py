# ==========================================================
# STEP 20 : Hungarian Matcher
# Episodic Binary Detection
# ==========================================================

from scipy.optimize import (
    linear_sum_assignment
)


class HungarianMatcher:

    def __init__(
        self,
        cost_class=
            CONFIG[
                "matcher_class_cost"
            ],
        cost_bbox=
            CONFIG[
                "matcher_bbox_cost"
            ],
        cost_giou=
            CONFIG[
                "matcher_giou_cost"
            ]
    ):

        if (
            cost_class == 0
            and
            cost_bbox == 0
            and
            cost_giou == 0
        ):

            raise ValueError(
                "All matcher costs cannot be zero."
            )


        self.cost_class = float(
            cost_class
        )

        self.cost_bbox = float(
            cost_bbox
        )

        self.cost_giou = float(
            cost_giou
        )


    @torch.no_grad()
    def __call__(
        self,
        outputs,
        targets
    ):

        pred_logits = (
            outputs[
                "pred_logits"
            ]
        )

        pred_boxes = (
            outputs[
                "pred_boxes"
            ]
        )


        if (
            pred_logits.ndim != 3
            or
            pred_logits.shape[-1] != 1
        ):

            raise ValueError(
                "Matcher expects pred_logits [B,Q,1]."
            )


        if (
            pred_boxes.ndim != 3
            or
            pred_boxes.shape[-1] != 4
        ):

            raise ValueError(
                "Matcher expects pred_boxes [B,Q,4]."
            )


        foreground_prob = (

            pred_logits
            .sigmoid()
            .squeeze(-1)
        )


        indices = []


        for b in range(
            pred_logits.shape[0]
        ):

            target_boxes = (
                targets[b][
                    "boxes"
                ]
            )


            target_labels = (
                targets[b][
                    "labels"
                ]
            )


            if (
                len(target_boxes)
                !=
                len(target_labels)
            ):

                raise ValueError(
                    "Target boxes/labels length mismatch."
                )


            if len(
                target_boxes
            ) == 0:

                empty = torch.empty(

                    0,

                    dtype=torch.long,

                    device=
                        pred_logits.device
                )


                indices.append(
                    (
                        empty,
                        empty.clone()
                    )
                )

                continue


            num_targets = (
                target_boxes.shape[0]
            )


            # ------------------------------------------------
            # Binary episodic class cost.
            #
            # Every GT is foreground relative to support.
            # ------------------------------------------------

            cost_class = -(

                foreground_prob[b]

                .unsqueeze(1)

                .expand(
                    -1,
                    num_targets
                )
            )


            # ------------------------------------------------
            # L1 bbox
            # ------------------------------------------------

            cost_bbox = torch.cdist(

                pred_boxes[b],

                target_boxes,

                p=1
            )


            # ------------------------------------------------
            # GIoU
            # ------------------------------------------------

            pred_xyxy = (
                box_cxcywh_to_xyxy(
                    pred_boxes[b]
                )
            )


            target_xyxy = (
                box_cxcywh_to_xyxy(
                    target_boxes
                )
            )


            cost_giou = -(

                generalized_box_iou(

                    pred_xyxy,

                    target_xyxy
                )
            )


            final_cost = (

                self.cost_class
                *
                cost_class

                +

                self.cost_bbox
                *
                cost_bbox

                +

                self.cost_giou
                *
                cost_giou
            )


            if not torch.isfinite(
                final_cost
            ).all():

                raise RuntimeError(
                    "Matcher cost contains NaN/Inf."
                )


            (
                src_idx,
                tgt_idx
            ) = linear_sum_assignment(

                final_cost
                .detach()
                .cpu()
                .numpy()
            )


            device = (
                pred_logits.device
            )


            indices.append((

                torch.as_tensor(

                    src_idx,

                    dtype=torch.long,

                    device=device
                ),

                torch.as_tensor(

                    tgt_idx,

                    dtype=torch.long,

                    device=device
                )
            ))


        return indices


matcher = HungarianMatcher()


print("=" * 70)
print("STEP 20 : HUNGARIAN MATCHER READY")
print("=" * 70)

print(
    "Class Cost:",
    matcher.cost_class
)

print(
    "BBox Cost :",
    matcher.cost_bbox
)

print(
    "GIoU Cost :",
    matcher.cost_giou
)

print("=" * 70)
