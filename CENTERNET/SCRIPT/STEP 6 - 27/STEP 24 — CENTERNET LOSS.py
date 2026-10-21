
# ============================================================
# STEP 24 — B1 ON A2: HYBRID WH LOSS
# Gaussian V1.3: handled by Step 15-16
# WH: A2 HYBRID ABSOLUTE + RELATIVE
# HEATMAP AND OFFSET LOSS: UNCHANGED
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG["hm_loss_weight"] = 1.0
CONFIG["wh_loss_weight"] = 0.1
CONFIG["offset_loss_weight"] = 1.0

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"


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
# 2. FEATURE GATHERING
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
# 3. MASKED ABSOLUTE L1
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
# 5. RELATIVE SMOOTH L1 — FROM A1/A2
# ============================================================

def masked_relative_smooth_l1_loss(
    prediction,
    target,
    mask,
    eps=2.0,
    beta=0.20,
    scale=10.0
):
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

        denominator_scale = target.clamp_min(
            float(eps)
        )

        relative_error = (
            prediction - target
        ) / denominator_scale

        per_coordinate_loss = F.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            beta=float(beta),
            reduction="none"
        )

        masked_loss = (
            per_coordinate_loss * expanded_mask
        ).sum()

        valid_coordinates = (
            expanded_mask.sum().clamp_min(1.0)
        )

        return (
            float(scale)
            * masked_loss
            / valid_coordinates
        )


# ============================================================
# 6. CENTERNET LOSS — B1 ON A2
# ============================================================

class CenterNetLoss(nn.Module):

    def __init__(
        self,
        hm_weight=1.0,
        wh_weight=0.1,
        offset_weight=1.0,
        wh_loss_type="hybrid_abs_relative",
        wh_absolute_fraction=0.75,
        wh_relative_fraction=0.25,
        wh_relative_eps=2.0,
        wh_relative_beta=0.20,
        wh_relative_scale=10.0
    ):
        super().__init__()

        self.hm_weight = float(hm_weight)
        self.wh_weight = float(wh_weight)
        self.offset_weight = float(offset_weight)

        self.wh_loss_type = str(wh_loss_type)

        self.wh_absolute_fraction = float(
            wh_absolute_fraction
        )

        self.wh_relative_fraction = float(
            wh_relative_fraction
        )

        self.wh_relative_eps = float(
            wh_relative_eps
        )

        self.wh_relative_beta = float(
            wh_relative_beta
        )

        self.wh_relative_scale = float(
            wh_relative_scale
        )

        assert self.wh_loss_type == "hybrid_abs_relative"

        assert self.wh_absolute_fraction > 0
        assert self.wh_relative_fraction > 0

        assert abs(
            self.wh_absolute_fraction
            + self.wh_relative_fraction - 1.0
        ) < 1e-12

    def forward(self, outputs, targets):

        # ----------------------------------------------------
        # HEATMAP — UNCHANGED CENTERNET FOCAL LOSS
        # ----------------------------------------------------

        heatmap_loss = centernet_focal_loss(
            outputs["heatmap"],
            targets["heatmap"]
        )

        # ----------------------------------------------------
        # GATHER WH AND OFFSET AT GT CENTERS
        # ----------------------------------------------------

        predicted_wh = transpose_and_gather_feature(
            outputs["wh"],
            targets["indices"]
        )

        predicted_offset = transpose_and_gather_feature(
            outputs["offset"],
            targets["indices"]
        )

        # ----------------------------------------------------
        # ABSOLUTE WH LOSS — BASELINE COMPONENT
        # ----------------------------------------------------

        wh_abs = masked_l1_loss(
            predicted_wh,
            targets["wh"],
            targets["mask"]
        )

        # ----------------------------------------------------
        # RELATIVE WH LOSS — A1 COMPONENT
        # ----------------------------------------------------

        wh_rel = masked_relative_smooth_l1_loss(
            predicted_wh,
            targets["wh"],
            targets["mask"],
            eps=self.wh_relative_eps,
            beta=self.wh_relative_beta,
            scale=self.wh_relative_scale
        )

        # ----------------------------------------------------
        # A2 HYBRID — 75% ABSOLUTE + 25% RELATIVE
        # ----------------------------------------------------

        wh_loss = (
            self.wh_absolute_fraction * wh_abs
            + self.wh_relative_fraction * wh_rel
        )

        # ----------------------------------------------------
        # OFFSET — UNCHANGED MASKED L1
        # ----------------------------------------------------

        offset_loss = masked_l1_loss(
            predicted_offset,
            targets["offset"],
            targets["mask"]
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

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
# 7. INITIALIZE CRITERION
# ============================================================

criterion = CenterNetLoss(
    hm_weight=CONFIG["hm_loss_weight"],
    wh_weight=CONFIG["wh_loss_weight"],
    offset_weight=CONFIG["offset_loss_weight"],
    wh_loss_type=CONFIG["wh_loss_type"],
    wh_absolute_fraction=CONFIG["wh_absolute_fraction"],
    wh_relative_fraction=CONFIG["wh_relative_fraction"],
    wh_relative_eps=CONFIG["wh_relative_eps"],
    wh_relative_beta=CONFIG["wh_relative_beta"],
    wh_relative_scale=CONFIG["wh_relative_scale"]
).to(device)


# ============================================================
# 8. GRADIENT SANITY TEST
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

test_abs = masked_l1_loss(
    test_prediction,
    test_target,
    test_mask
)

test_rel = masked_relative_smooth_l1_loss(
    test_prediction,
    test_target,
    test_mask,
    eps=CONFIG["wh_relative_eps"],
    beta=CONFIG["wh_relative_beta"],
    scale=CONFIG["wh_relative_scale"]
)

test_hybrid = (
    CONFIG["wh_absolute_fraction"] * test_abs
    + CONFIG["wh_relative_fraction"] * test_rel
)

assert torch.isfinite(test_hybrid)

test_hybrid.backward()

assert torch.isfinite(
    test_prediction.grad
).all()

assert (
    test_prediction.grad[0, 0].abs() > 0
).any()

assert (
    test_prediction.grad[0, 1].abs().sum().item()
    == 0.0
)


# ============================================================
# 9. CONFIGURATION VERIFICATION
# ============================================================

assert criterion.wh_loss_type == "hybrid_abs_relative"

assert criterion.wh_absolute_fraction == 0.75
assert criterion.wh_relative_fraction == 0.25

assert criterion.wh_relative_eps == 2.0
assert criterion.wh_relative_beta == 0.20
assert criterion.wh_relative_scale == 10.0

assert criterion.hm_weight == 1.0
assert criterion.wh_weight == 0.1
assert criterion.offset_weight == 1.0

print("=" * 70)
print("STEP 24 — B1 ON A2 HYBRID WH LOSS")
print("=" * 70)

print("Gaussian mode :", CONFIG["gaussian_radius_mode"])
print("WH loss       :", criterion.wh_loss_type)
print("Absolute frac :", criterion.wh_absolute_fraction)
print("Relative frac :", criterion.wh_relative_fraction)
print("Relative eps  :", criterion.wh_relative_eps)
print("Relative beta :", criterion.wh_relative_beta)
print("Relative scale:", criterion.wh_relative_scale)

print("Gradient sanity: PASSED")
print("STEP 24 PASSED — NO TRAINING")
