# ==========================================================
# STEP 20 : Hungarian Matcher
#
# Episodic binary detection.
#
# Every GT in the query belongs to the support category.
# ==========================================================

from scipy.optimize import (
    linear_sum_assignment
)


class HungarianMatcher:

    def __init__(
        self,
        cost_class=CONFIG[
            "matcher_class_cost"
        ],
        cost_bbox=CONFIG[
            "matcher_bbox_cost"
        ],
        cost_giou=CONFIG[
            "matcher_giou_cost"
        ],
    ):

        if (
            cost_class == 0
            and
            cost_bbox == 0
            and
            cost_giou == 0
        ):
            raise ValueError(
                "All matcher costs "
                "cannot be zero."
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
                "Matcher expects "
                "pred_logits [B,Q,1]."
            )

        if (
            pred_boxes.ndim != 3
            or
            pred_boxes.shape[-1] != 4
        ):
            raise ValueError(
                "Matcher expects "
                "pred_boxes [B,Q,4]."
            )

        if (
            pred_logits.shape[0]
            !=
            len(
                targets
            )
        ):
            raise ValueError(
                "Output/target batch mismatch."
            )

        foreground_prob = (
            pred_logits
            .sigmoid()
            .squeeze(
                -1
            )
        )

        indices = []

        for batch_index in range(
            pred_logits.shape[0]
        ):

            target_boxes = (
                targets[
                    batch_index
                ][
                    "boxes"
                ]
            )

            if len(
                target_boxes
            ) == 0:

                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=
                        pred_logits.device,
                )

                indices.append(
                    (
                        empty,
                        empty.clone(),
                    )
                )

                continue

            num_targets = (
                target_boxes.shape[0]
            )

            # Binary support-match class cost.
            cost_class = -(
                foreground_prob[
                    batch_index
                ]
                .unsqueeze(
                    1
                )
                .expand(
                    -1,
                    num_targets
                )
            )

            cost_bbox = torch.cdist(

                pred_boxes[
                    batch_index
                ],

                target_boxes,

                p=1,
            )

            pred_xyxy = (
                box_cxcywh_to_xyxy(
                    pred_boxes[
                        batch_index
                    ]
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
                    target_xyxy,
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
                    "Matcher cost contains "
                    "NaN/Inf."
                )

            (
                source_index,
                target_index

            ) = linear_sum_assignment(

                final_cost
                .detach()
                .cpu()
                .numpy()
            )

            device = (
                pred_logits.device
            )

            indices.append(
                (

                    torch.as_tensor(
                        source_index,
                        dtype=torch.long,
                        device=device,
                    ),

                    torch.as_tensor(
                        target_index,
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )

        return indices


matcher = HungarianMatcher()


print("=" * 70)
print("STEP 20 : HUNGARIAN MATCHER READY")
print("=" * 70)

print("Class cost :", matcher.cost_class)
print("BBox cost  :", matcher.cost_bbox)
print("GIoU cost  :", matcher.cost_giou)

print("=" * 70)
