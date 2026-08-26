# ==========================================================
# STEP 24 : Hungarian Matcher
#           Episodic Binary (V2 LOCKED)
# ==========================================================

import torch

from scipy.optimize import (
    linear_sum_assignment
)


class HungarianMatcher:

    def __init__(
        self,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    ):

        if (
            cost_class == 0
            and
            cost_bbox == 0
            and
            cost_giou == 0
        ):

            raise ValueError(
                "All matching costs "
                "cannot be zero."
            )


        self.cost_class = (
            cost_class
        )

        self.cost_bbox = (
            cost_bbox
        )

        self.cost_giou = (
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


        # ==================================================
        # Shape Validation
        # ==================================================

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
                "V2 matcher expects "
                "pred_logits [B,Q,1]."
            )


        if (
            pred_boxes.ndim
            !=
            3

            or

            pred_boxes.shape[-1]
            !=
            4
        ):

            raise ValueError(
                "pred_boxes must "
                "have shape [B,Q,4]."
            )


        batch_size = (
            pred_logits.shape[0]
        )


        # ==================================================
        # Foreground Probability
        #
        # [B,Q,1]
        # ->
        # [B,Q]
        # ==================================================

        foreground_prob = (

            pred_logits
            .sigmoid()
            .squeeze(-1)
        )


        indices = []


        # ==================================================
        # Per Batch Matching
        # ==================================================

        for b in range(
            batch_size
        ):

            target_boxes = (
                targets[
                    b
                ][
                    "boxes"
                ]
            )


            target_labels = (
                targets[
                    b
                ][
                    "labels"
                ]
            )


            # ==============================================
            # Target Consistency
            # ==============================================

            if (
                len(
                    target_boxes
                )
                !=
                len(
                    target_labels
                )
            ):

                raise ValueError(
                    "Target boxes and labels "
                    "have different lengths."
                )


            # ==============================================
            # Empty Target
            # ==============================================

            if (
                len(
                    target_boxes
                )
                ==
                0
            ):

                device = (
                    pred_logits.device
                )


                indices.append((

                    torch.empty(
                        0,
                        dtype=torch.long,
                        device=device
                    ),

                    torch.empty(
                        0,
                        dtype=torch.long,
                        device=device
                    )
                ))


                continue


            num_targets = (
                target_boxes.shape[0]
            )


            # ==============================================
            # Classification Cost
            #
            # Semua GT dalam 1-way episode
            # adalah foreground relatif terhadap
            # support class.
            # ==============================================

            cost_class = -(

                foreground_prob[
                    b
                ]
                .unsqueeze(1)
                .expand(
                    -1,
                    num_targets
                )
            )


            # ==============================================
            # Bounding Box L1 Cost
            # ==============================================

            cost_bbox = (
                torch.cdist(

                    pred_boxes[
                        b
                    ],

                    target_boxes,

                    p=1
                )
            )


            # ==============================================
            # GIoU Cost
            #
            # Fungsi ini baru akan tersedia
            # setelah STEP 25-27 dijalankan.
            #
            # Itu tidak masalah karena fungsi
            # hanya dipanggil ketika matcher()
            # dieksekusi.
            # ==============================================

            pred_xyxy = (
                box_cxcywh_to_xyxy(

                    pred_boxes[
                        b
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

                    target_xyxy
                )
            )


            # ==============================================
            # Final Hungarian Cost
            # ==============================================

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


            # ==============================================
            # Numerical Safety
            # ==============================================

            if not torch.isfinite(
                final_cost
            ).all():

                raise RuntimeError(
                    "Hungarian matching "
                    "cost contains NaN/Inf."
                )


            # ==============================================
            # Hungarian Assignment
            # ==============================================

            src_idx, tgt_idx = (
                linear_sum_assignment(

                    final_cost
                    .detach()
                    .cpu()
                    .numpy()
                )
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


# ==========================================================
# Initialize Matcher
#
# IMPORTANT:
# Jangan panggil matcher() di STEP 24.
#
# box_cxcywh_to_xyxy dan generalized_box_iou
# baru tersedia setelah STEP 25-27.
# ==========================================================

matcher = HungarianMatcher(

    cost_class=1.0,

    cost_bbox=5.0,

    cost_giou=2.0
)


print("=" * 70)

print(
    "STEP 24 : EPISODIC BINARY "
    "HUNGARIAN MATCHER DEFINED"
)

print("=" * 70)

print(
    "Matcher definition : READY"
)

print(
    "Sanity test        : RUN AFTER STEP 27"
)

print("=" * 70)

