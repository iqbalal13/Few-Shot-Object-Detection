# ============================================================
# STEP 24 — CENTERNET LOSS
# ============================================================

print("=" * 70)
print("STEP 24 — BUILD CENTERNET LOSS")
print("=" * 70)

CONFIG["hm_loss_weight"] = 1.0
CONFIG["wh_loss_weight"] = 0.1
CONFIG["offset_loss_weight"] = 1.0


def centernet_focal_loss(
    prediction_logits,
    target
):

    prediction = torch.sigmoid(
        prediction_logits
    )

    prediction = torch.clamp(
        prediction,
        min=1e-4,
        max=1 - 1e-4
    )

    positive_mask = target.eq(
        1.0
    ).float()

    negative_mask = target.lt(
        1.0
    ).float()

    negative_weights = torch.pow(
        1 - target,
        4
    )

    positive_loss = (
        torch.log(prediction)
        * torch.pow(
            1 - prediction,
            2
        )
        * positive_mask
    )

    negative_loss = (
        torch.log(
            1 - prediction
        )
        * torch.pow(
            prediction,
            2
        )
        * negative_weights
        * negative_mask
    )

    num_positive = positive_mask.sum()

    positive_loss = positive_loss.sum()
    negative_loss = negative_loss.sum()

    if num_positive > 0:

        loss = -(
            positive_loss
            + negative_loss
        ) / num_positive

    else:

        loss = -negative_loss

    return loss


def transpose_and_gather_feature(
    feature,
    indices
):
    """
    feature:
        [B, C, H, W]

    indices:
        [B, max_objects]

    return:
        [B, max_objects, C]
    """

    batch, channels, height, width = \
        feature.shape

    feature = feature.permute(
        0,
        2,
        3,
        1
    ).contiguous()

    feature = feature.view(
        batch,
        -1,
        channels
    )

    indices = indices.unsqueeze(
        -1
    ).expand(
        -1,
        -1,
        channels
    )

    gathered = feature.gather(
        1,
        indices
    )

    return gathered


def masked_l1_loss(
    prediction,
    target,
    mask
):

    mask = mask.unsqueeze(
        -1
    ).expand_as(
        prediction
    ).float()

    loss = F.l1_loss(
        prediction * mask,
        target * mask,
        reduction="sum"
    )

    denominator = (
        mask.sum()
        + 1e-4
    )

    return loss / denominator


def build_batch_centernet_targets(
    targets,
    device
):

    encoded = [
        build_centernet_target(
            target
        )
        for target in targets
    ]

    heatmap = torch.stack(
        [
            item["heatmap"]
            for item in encoded
        ]
    ).to(device)

    wh = torch.stack(
        [
            item["wh"]
            for item in encoded
        ]
    ).to(device)

    offset = torch.stack(
        [
            item["offset"]
            for item in encoded
        ]
    ).to(device)

    indices = torch.stack(
        [
            item["indices"]
            for item in encoded
        ]
    ).to(device)

    mask = torch.stack(
        [
            item["mask"]
            for item in encoded
        ]
    ).to(device)

    return {
        "heatmap": heatmap,
        "wh": wh,
        "offset": offset,
        "indices": indices,
        "mask": mask
    }


class CenterNetLoss(nn.Module):

    def __init__(
        self,
        hm_weight=1.0,
        wh_weight=0.1,
        offset_weight=1.0
    ):
        super().__init__()

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offset_weight

    def forward(
        self,
        outputs,
        targets
    ):

        heatmap_loss = centernet_focal_loss(
            outputs["heatmap"],
            targets["heatmap"]
        )

        predicted_wh = \
            transpose_and_gather_feature(
                outputs["wh"],
                targets["indices"]
            )

        predicted_offset = \
            transpose_and_gather_feature(
                outputs["offset"],
                targets["indices"]
            )

        wh_loss = masked_l1_loss(
            predicted_wh,
            targets["wh"],
            targets["mask"]
        )

        offset_loss = masked_l1_loss(
            predicted_offset,
            targets["offset"],
            targets["mask"]
        )

        total_loss = (
            self.hm_weight
            * heatmap_loss

            + self.wh_weight
            * wh_loss

            + self.offset_weight
            * offset_loss
        )

        return {
            "loss_total": total_loss,
            "loss_heatmap": heatmap_loss,
            "loss_wh": wh_loss,
            "loss_offset": offset_loss
        }


criterion = CenterNetLoss(
    hm_weight=CONFIG["hm_loss_weight"],
    wh_weight=CONFIG["wh_loss_weight"],
    offset_weight=CONFIG["offset_loss_weight"]
).to(device)

print("Heatmap loss weight :", CONFIG["hm_loss_weight"])
print("WH loss weight      :", CONFIG["wh_loss_weight"])
print("Offset loss weight  :", CONFIG["offset_loss_weight"])

print("\nSTEP 24 PASSED")
