
# ============================================================
# STEP 24 — EXPERIMENT B1 LOSS
# HEATMAP: ORIGINAL CENTERNET FOCAL LOSS
# WH: BASELINE MASKED L1
# OFFSET: BASELINE MASKED L1
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG["hm_loss_weight"] = 1.0
CONFIG["wh_loss_weight"] = 0.1
CONFIG["offset_loss_weight"] = 1.0


# ============================================================
# 1. ORIGINAL CENTERNET FOCAL LOSS
# ============================================================

def centernet_focal_loss(prediction_logits, target):
    with torch.amp.autocast(
        device_type=prediction_logits.device.type,
        enabled=False
    ):
        logits = prediction_logits.float()
        target = target.float()

        probability = torch.sigmoid(logits)

        positive_mask = target.eq(1.0).float()
        negative_mask = target.lt(1.0).float()

        negative_weights = torch.pow(
            1.0 - target, 4
        )

        positive_loss = (
            F.logsigmoid(logits)
            * torch.pow(1.0 - probability, 2)
            * positive_mask
        )

        negative_loss = (
            F.logsigmoid(-logits)
            * torch.pow(probability, 2)
            * negative_weights
            * negative_mask
        )

        num_positive = positive_mask.sum()

        loss = torch.where(
            num_positive > 0,
            -(
                positive_loss.sum()
                + negative_loss.sum()
            ) / num_positive.clamp_min(1.0),
            -negative_loss.sum()
        )

        return loss


# ============================================================
# 2. ORIGINAL FEATURE GATHERING
# ============================================================

def transpose_and_gather_feature(feature, indices):
    batch, channels, height, width = feature.shape

    feature = feature.permute(
        0, 2, 3, 1
    ).contiguous()

    feature = feature.view(
        batch, -1, channels
    )

    indices = indices.unsqueeze(-1).expand(
        -1, -1, channels
    )

    return feature.gather(1, indices)


# ============================================================
# 3. BASELINE MASKED L1
# ============================================================

def masked_l1_loss(prediction, target, mask):
    with torch.amp.autocast(
        device_type=prediction.device.type,
        enabled=False
    ):
        prediction = prediction.float()
        target = target.float()

        expanded_mask = (
            mask.unsqueeze(-1)
            .expand_as(prediction)
            .float()
        )

        loss = F.l1_loss(
            prediction * expanded_mask,
            target * expanded_mask,
            reduction="sum"
        )

        return loss / (
            expanded_mask.sum() + 1e-4
        )


# ============================================================
# 4. BATCH TARGET BUILDER
# ============================================================

def build_batch_centernet_targets(targets, device):
    encoded = [
        build_centernet_target(target)
        for target in targets
    ]

    names = (
        "heatmap",
        "wh",
        "offset",
        "indices",
        "mask"
    )

    return {
        name: torch.stack([
            item[name]
            for item in encoded
        ]).to(device, non_blocking=True)
        for name in names
    }


# ============================================================
# 5. B1 CENTERNET LOSS
# ============================================================

class CenterNetLoss(nn.Module):
    def __init__(
        self,
        hm_weight=1.0,
        wh_weight=0.1,
        offset_weight=1.0,
        wh_loss_type="masked_l1"
    ):
        super().__init__()

        self.hm_weight = float(hm_weight)
        self.wh_weight = float(wh_weight)
        self.offset_weight = float(offset_weight)

        self.wh_loss_type = str(wh_loss_type)

        assert self.wh_loss_type == "masked_l1"

    def forward(self, outputs, targets):
        heatmap_loss = centernet_focal_loss(
            outputs["heatmap"],
            targets["heatmap"]
        )

        predicted_wh = transpose_and_gather_feature(
            outputs["wh"],
            targets["indices"]
        )

        predicted_offset = transpose_and_gather_feature(
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
            self.hm_weight * heatmap_loss
            + self.wh_weight * wh_loss
            + self.offset_weight * offset_loss
        )

        return {
            "loss_total": total_loss,
            "loss_heatmap": heatmap_loss,
            "loss_wh": wh_loss,
            "loss_offset": offset_loss
        }


# ============================================================
# 6. CRITERION
# ============================================================

criterion = CenterNetLoss(
    hm_weight=CONFIG["hm_loss_weight"],
    wh_weight=CONFIG["wh_loss_weight"],
    offset_weight=CONFIG["offset_loss_weight"],
    wh_loss_type=CONFIG["wh_loss_type"]
).to(device)


# ============================================================
# 7. MASKED L1 GRADIENT SANITY
# ============================================================

test_prediction = torch.tensor(
    [[[3.0, 2.0], [100000.0, 100000.0]]],
    device=device,
    requires_grad=True
)

test_target = torch.tensor(
    [[[2.0, 2.0], [0.0, 0.0]]],
    device=device
)

test_mask = torch.tensor(
    [[1, 0]],
    device=device
)

test_loss = masked_l1_loss(
    test_prediction,
    test_target,
    test_mask
)

assert torch.isfinite(test_loss)

test_loss.backward()

assert torch.isfinite(test_prediction.grad).all()

assert (
    test_prediction.grad[0, 0].abs() > 0
).any()

assert (
    test_prediction.grad[0, 1].abs().sum().item()
    == 0.0
)

assert criterion.wh_loss_type == "masked_l1"

print("=" * 70)
print("STEP 24 — B1 LOSS")
print("=" * 70)
print("Heatmap loss: CenterNet Focal")
print("WH loss     : Masked L1")
print("Offset loss : Masked L1")
print("HM weight   :", criterion.hm_weight)
print("WH weight   :", criterion.wh_weight)
print("Offset weight:", criterion.offset_weight)
print("Gradient test: PASSED")
print("STEP 24 PASSED — NO TRAINING")
