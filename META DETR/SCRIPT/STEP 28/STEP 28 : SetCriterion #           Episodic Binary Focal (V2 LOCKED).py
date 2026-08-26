# ==========================================================
# STEP 28 : SetCriterion
#           Episodic Binary Focal (V2 LOCKED)
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import (
    sigmoid_focal_loss
)


class SetCriterion(nn.Module):

    def __init__(
        self,
        matcher,
        focal_alpha=
            CONFIG["focal_alpha"],
        focal_gamma=
            CONFIG["focal_gamma"],
        loss_cls_weight=1.0,
        loss_bbox_weight=5.0,
        loss_giou_weight=2.0
    ):

        super().__init__()


        self.matcher = (
            matcher
        )


        self.focal_alpha = (
            focal_alpha
        )


        self.focal_gamma = (
            focal_gamma
        )


        self.loss_cls_weight = (
            loss_cls_weight
        )


        self.loss_bbox_weight = (
            loss_bbox_weight
        )


        self.loss_giou_weight = (
            loss_giou_weight
        )


    def forward(
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
            pred_logits.ndim
            !=
            3

            or

            pred_logits.shape[-1]
            !=
            1
        ):

            raise ValueError(
                "V2 criterion expects "
                "pred_logits [B,Q,1]."
            )


        # ==================================================
        # Matching
        # ==================================================

        indices = (
            self.matcher(
                outputs,
                targets
            )
        )


        # ==================================================
        # Binary targets
        #
        # unmatched -> 0
        # matched   -> 1
        # ==================================================

        target_foreground = (
            torch.zeros_like(
                pred_logits
            )
        )


        for batch_idx, (
            src_idx,
            tgt_idx
        ) in enumerate(
            indices
        ):

            if (
                len(
                    src_idx
                )
                ==
                0
            ):

                continue


            target_foreground[
                batch_idx,
                src_idx,
                0
            ] = 1.0


        # ==================================================
        # Number GT objects
        # ==================================================

        num_target_boxes = sum(

            len(
                target[
                    "boxes"
                ]
            )

            for target
            in targets
        )


        num_target_boxes = max(

            int(
                num_target_boxes
            ),

            1
        )


        # ==================================================
        # Sigmoid Focal Classification
        # ==================================================

        loss_cls = (

            sigmoid_focal_loss(

                inputs=
                    pred_logits,

                targets=
                    target_foreground,

                alpha=
                    self.focal_alpha,

                gamma=
                    self.focal_gamma,

                reduction=
                    "sum"
            )

            /

            float(
                num_target_boxes
            )
        )


        # ==================================================
        # Matched Boxes
        # ==================================================

        matched_pred_boxes = []

        matched_target_boxes = []


        for batch_idx, (
            src_idx,
            tgt_idx
        ) in enumerate(
            indices
        ):

            if (
                len(
                    src_idx
                )
                ==
                0
            ):

                continue


            matched_pred_boxes.append(

                pred_boxes[
                    batch_idx,
                    src_idx
                ]
            )


            matched_target_boxes.append(

                targets[
                    batch_idx
                ][
                    "boxes"
                ][
                    tgt_idx
                ]
            )


        if (
            len(
                matched_pred_boxes
            )
            >
            0
        ):

            src_boxes = torch.cat(

                matched_pred_boxes,

                dim=0
            )


            target_boxes = torch.cat(

                matched_target_boxes,

                dim=0
            )


            # ==============================================
            # L1
            # ==============================================

            loss_bbox = (

                F.l1_loss(

                    src_boxes,

                    target_boxes,

                    reduction=
                        "none"
                ).sum()

                /

                float(
                    num_target_boxes
                )
            )


            # ==============================================
            # GIoU
            # ==============================================

            src_xyxy = (
                box_cxcywh_to_xyxy(
                    src_boxes
                )
            )


            target_xyxy = (
                box_cxcywh_to_xyxy(
                    target_boxes
                )
            )


            giou_matrix = (
                generalized_box_iou(

                    src_xyxy,

                    target_xyxy
                )
            )


            matched_giou = (
                torch.diag(
                    giou_matrix
                )
            )


            loss_giou = (

                (
                    1.0
                    -
                    matched_giou
                ).sum()

                /

                float(
                    num_target_boxes
                )
            )


        else:

            loss_bbox = (

                pred_boxes.sum()
                *
                0.0
            )


            loss_giou = (

                pred_boxes.sum()
                *
                0.0
            )


        # ==================================================
        # Total
        # ==================================================

        loss_total = (

            self.loss_cls_weight
            *
            loss_cls

            +

            self.loss_bbox_weight
            *
            loss_bbox

            +

            self.loss_giou_weight
            *
            loss_giou
        )


        return {

            "loss_cls":
                loss_cls,

            "loss_bbox":
                loss_bbox,

            "loss_giou":
                loss_giou,

            "loss_total":
                loss_total
        }


# ==========================================================
# Initialize
# ==========================================================

criterion = SetCriterion(

    matcher=
        matcher,

    focal_alpha=
        CONFIG[
            "focal_alpha"
        ],

    focal_gamma=
        CONFIG[
            "focal_gamma"
        ]

).to(
    CONFIG["device"]
)


print("=" * 70)

print(
    "STEP 28 : EPISODIC BINARY "
    "CRITERION READY"
)

print("=" * 70)


print(
    "Classification mode:",
    CONFIG[
        "classification_mode"
    ]
)


print(
    "Detection logits   :",
    CONFIG[
        "num_detection_logits"
    ]
)


print(
    "Focal alpha        :",
    CONFIG[
        "focal_alpha"
    ]
)


print(
    "Focal gamma        :",
    CONFIG[
        "focal_gamma"
    ]
)


print("=" * 70)
