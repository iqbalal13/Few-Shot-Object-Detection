
# ============================================================
# STEP 24 — B1: ORIGINAL A2 HYBRID WH, HEATMAP V1.3 TARGET
# Only Step 15/16 changes Gaussian; the A2 loss is unchanged.
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG['hm_loss_weight'] = 1.0
CONFIG['wh_loss_weight'] = 0.1
CONFIG['offset_loss_weight'] = 1.0
assert CONFIG['gaussian_radius_mode'] == 'v1_3_corrected'
assert CONFIG['wh_loss_type'] == 'hybrid_abs_relative'

def centernet_focal_loss(prediction_logits, target):
    with torch.amp.autocast(device_type=prediction_logits.device.type, enabled=False):
        logits = prediction_logits.float()
        target = target.float()
        p = torch.sigmoid(logits)
        positives = target.eq(1.0).float()
        negatives = target.lt(1.0).float()
        neg_weights = (1.0 - target).pow(4)
        pos_loss = F.logsigmoid(logits) * (1.0 - p).pow(2) * positives
        neg_loss = F.logsigmoid(-logits) * p.pow(2) * neg_weights * negatives
        npos = positives.sum()
        return torch.where(npos > 0,
                           -(pos_loss.sum() + neg_loss.sum()) / npos.clamp_min(1.0),
                           -neg_loss.sum())

def transpose_and_gather_feature(feature, indices):
    b, c, h, w = feature.shape
    feature = feature.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)
    return feature.gather(1, indices.unsqueeze(-1).expand(-1, -1, c))

def masked_l1_loss(prediction, target, mask):
    with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        expanded = mask.unsqueeze(-1).expand_as(prediction).float()
        error = F.l1_loss(prediction * expanded, target * expanded, reduction='sum')
        return error / (expanded.sum() + 1e-4)

def masked_relative_smooth_l1_loss(prediction, target, mask,
                                   eps=2.0, beta=0.20, scale=10.0):
    with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        expanded = mask.unsqueeze(-1).expand_as(prediction).float()
        relative = (prediction - target) / target.clamp_min(float(eps))
        per_coord = F.smooth_l1_loss(relative, torch.zeros_like(relative),
                                     beta=float(beta), reduction='none')
        return float(scale) * (per_coord * expanded).sum() / expanded.sum().clamp_min(1.0)

def build_batch_centernet_targets(targets, device):
    encoded = [build_centernet_target(t) for t in targets]
    keys = ('heatmap', 'wh', 'offset', 'indices', 'mask')
    return {key: torch.stack([e[key] for e in encoded]).to(device, non_blocking=True)
            for key in keys}

class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight=1.0, wh_weight=0.1, offset_weight=1.0,
                 wh_loss_type='hybrid_abs_relative', wh_absolute_fraction=0.75,
                 wh_relative_fraction=0.25, wh_relative_eps=2.0,
                 wh_relative_beta=0.20, wh_relative_scale=10.0):
        super().__init__()
        self.hm_weight, self.wh_weight = float(hm_weight), float(wh_weight)
        self.offset_weight = float(offset_weight)
        self.wh_loss_type = wh_loss_type
        self.wh_absolute_fraction = float(wh_absolute_fraction)
        self.wh_relative_fraction = float(wh_relative_fraction)
        self.wh_relative_eps = float(wh_relative_eps)
        self.wh_relative_beta = float(wh_relative_beta)
        self.wh_relative_scale = float(wh_relative_scale)
        assert self.wh_loss_type == 'hybrid_abs_relative'
        assert self.wh_absolute_fraction > 0 and self.wh_relative_fraction > 0
        assert abs(self.wh_absolute_fraction + self.wh_relative_fraction - 1.0) < 1e-12

    def forward(self, outputs, targets):
        hm = centernet_focal_loss(outputs['heatmap'], targets['heatmap'])
        wh_pred = transpose_and_gather_feature(outputs['wh'], targets['indices'])
        off_pred = transpose_and_gather_feature(outputs['offset'], targets['indices'])
        wh_abs = masked_l1_loss(wh_pred, targets['wh'], targets['mask'])
        wh_rel = masked_relative_smooth_l1_loss(
            wh_pred, targets['wh'], targets['mask'],
            self.wh_relative_eps, self.wh_relative_beta, self.wh_relative_scale)
        wh = self.wh_absolute_fraction * wh_abs + self.wh_relative_fraction * wh_rel
        off = masked_l1_loss(off_pred, targets['offset'], targets['mask'])
        total = self.hm_weight*hm + self.wh_weight*wh + self.offset_weight*off
        return {'loss_total': total, 'loss_heatmap': hm,
                'loss_wh': wh, 'loss_offset': off}

criterion = CenterNetLoss(
    hm_weight=CONFIG['hm_loss_weight'],
    wh_weight=CONFIG['wh_loss_weight'],
    offset_weight=CONFIG['offset_loss_weight'],
    wh_loss_type=CONFIG['wh_loss_type'],
    wh_absolute_fraction=CONFIG['wh_absolute_fraction'],
    wh_relative_fraction=CONFIG['wh_relative_fraction'],
    wh_relative_eps=CONFIG['wh_relative_eps'],
    wh_relative_beta=CONFIG['wh_relative_beta'],
    wh_relative_scale=CONFIG['wh_relative_scale'],
).to(device)

# No-training sanity: masked coordinates have exactly zero gradient.
pred = torch.tensor([[[3., 2.], [100000., 100000.]]], device=device,
                    requires_grad=True)
true = torch.tensor([[[2., 2.], [0., 0.]]], device=device)
valid = torch.tensor([[1, 0]], device=device)
loss = (0.75 * masked_l1_loss(pred, true, valid)
        + 0.25 * masked_relative_smooth_l1_loss(
            pred, true, valid, eps=2.0, beta=0.20, scale=10.0))
assert torch.isfinite(loss)
loss.backward()
assert torch.isfinite(pred.grad).all()
assert pred.grad[0, 0].abs().sum() > 0
assert pred.grad[0, 1].abs().sum() == 0
assert criterion.wh_loss_type == CONFIG['wh_loss_type']
print('STEP 24 PASS | HM focal | WH hybrid A2 (75/25) | Offset masked L1')
