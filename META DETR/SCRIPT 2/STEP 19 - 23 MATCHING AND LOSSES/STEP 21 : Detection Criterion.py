# ==========================================================
# STEP 21 : Detection Criterion
#
# Ldet =
#   focal
#   + 5 * L1
#   + 2 * GIoU
# ==========================================================

from torchvision.ops import (
    sigmoid_focal_loss
)


class DetectionCriterion(
    nn.Module
):

    def __init__(
        self,
        matcher,
        focal_alpha=
            CONFIG[
                "focal_alpha"
            ],
        focal_gamma=
            CONFIG[
                "focal_gamma"
            ],
        bbox_weight=
            CONFIG[
                "loss_bbox_weight"
            ],
        giou_weight=
            CONFIG[
                "loss_giou_weight"
            ]
    ):
        super().__init__()


        self.matcher = matcher

        self.focal_alpha = float(
            focal_alpha
        )

        self.focal_gamma = float(
            focal_gamma
        )

        self.bbox_weight = float(
            bbox_weight
        )

        self.giou_weight = float(
            giou_weight
        )


    def forward(
        self,
        outputs,
        targets,
        indices=None
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
                "Criterion expects logits [B,Q,1]."
            )


        # --------------------------------------------------
        # Match only once if caller already has indices.
        # --------------------------------------------------

        if indices is None:

            indices = self.matcher(
                outputs,
                targets
            )


        # --------------------------------------------------
        # Binary foreground target:
        # unmatched = 0
        # matched   = 1
        # --------------------------------------------------

        target_foreground = (
            torch.zeros_like(
                pred_logits
            )
        )


        for b, (
            src_idx,
            _
        ) in enumerate(
            indices
        ):

            if len(src_idx) > 0:

                target_foreground[
                    b,
                    src_idx,
                    0
                ] = 1.0


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
            int(num_target_boxes),
            1
        )


        # --------------------------------------------------
        # Focal
        # --------------------------------------------------

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

                reduction="sum"
            )

            /
            float(
                num_target_boxes
            )
        )


        matched_pred_boxes = []
        matched_target_boxes = []


        for b, (
            src_idx,
            tgt_idx
        ) in enumerate(
            indices
        ):

            if len(src_idx) == 0:
                continue


            matched_pred_boxes.append(

                pred_boxes[
                    b,
                    src_idx
                ]
            )


            matched_target_boxes.append(

                targets[b][
                    "boxes"
                ][
                    tgt_idx
                ]
            )


        if matched_pred_boxes:

            src_boxes = torch.cat(

                matched_pred_boxes,

                dim=0
            )


            target_boxes = torch.cat(

                matched_target_boxes,

                dim=0
            )


            # ----------------------------------------------
            # L1
            # ----------------------------------------------

            loss_bbox = (

                F.l1_loss(

                    src_boxes,

                    target_boxes,

                    reduction="none"
                ).sum()

                /
                float(
                    num_target_boxes
                )
            )


            # ----------------------------------------------
            # GIoU
            # ----------------------------------------------

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


            matched_giou = torch.diag(
                giou_matrix
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

            zero = (
                pred_boxes.sum()
                *
                0.0
            )

            loss_bbox = zero

            loss_giou = zero


        loss_total = (

            loss_cls

            +

            self.bbox_weight
            *
            loss_bbox

            +

            self.giou_weight
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


criterion = DetectionCriterion(

    matcher=matcher

).to(
    CONFIG["device"]
)


print("=" * 70)
print("STEP 21 : DETECTION CRITERION READY")
print("=" * 70)

print(
    "Focal alpha:",
    CONFIG["focal_alpha"]
)

print(
    "Focal gamma:",
    CONFIG["focal_gamma"]
)

print(
    "BBox weight:",
    CONFIG["loss_bbox_weight"]
)

print(
    "GIoU weight:",
    CONFIG["loss_giou_weight"]
)

print("=" * 70)
