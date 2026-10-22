
# STEP 24 — A3 SWITCHABLE WH LOSS
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG["hm_loss_weight"] = 1.0
CONFIG["wh_loss_weight"] = 0.1
CONFIG["offset_loss_weight"] = 1.0

assert CONFIG["gaussian_radius_mode"]=="v1_2_original"

def centernet_focal_loss(logits,target):
    with torch.amp.autocast(
        device_type=logits.device.type,
        enabled=False
    ):
        logits,target = logits.float(),target.float()
        p = logits.sigmoid()
        positive = target.eq(1.).float()
        negative = target.lt(1.).float()

        pos = (
            F.logsigmoid(logits)
            *(1.-p).square()*positive
        )
        neg = (
            F.logsigmoid(-logits)
            *p.square()
            *(1.-target).pow(4)
            *negative
        )

        n = positive.sum()

        return torch.where(
            n>0,
            -(pos.sum()+neg.sum())/n.clamp_min(1.),
            -neg.sum()
        )

def transpose_and_gather_feature(feature,indices):
    b,c,h,w = feature.shape

    return feature.permute(
        0,2,3,1
    ).contiguous().view(
        b,h*w,c
    ).gather(
        1,indices.unsqueeze(-1).expand(-1,-1,c)
    )

def masked_l1_loss(pred,gt,mask):
    with torch.amp.autocast(
        device_type=pred.device.type,
        enabled=False
    ):
        pred,gt = pred.float(),gt.float()
        valid = mask.unsqueeze(-1).expand_as(pred).float()

        return (
            ((pred-gt).abs()*valid).sum()
            /(valid.sum()+1e-4)
        )

def masked_relative_smooth_l1_loss(
    pred,gt,mask,eps=2.,beta=.2,scale=10.
):
    with torch.amp.autocast(
        device_type=pred.device.type,
        enabled=False
    ):
        pred,gt = pred.float(),gt.float()
        valid = mask.unsqueeze(-1).expand_as(pred).float()
        relative = (pred-gt)/gt.clamp_min(float(eps))

        loss = F.smooth_l1_loss(
            relative,torch.zeros_like(relative),
            beta=float(beta),reduction="none"
        )

        return (
            float(scale)*(loss*valid).sum()
            /valid.sum().clamp_min(1.)
        )

def build_batch_centernet_targets(raw_targets,device):
    encoded = [build_centernet_target(t) for t in raw_targets]

    keys = (
        "heatmap","wh","offset",
        "indices","mask",
        "neighbor_indices","neighbor_mask"
    )

    return {
        k:torch.stack([e[k] for e in encoded]).to(
            device,non_blocking=True
        )
        for k in keys
    }

class CenterNetLoss(nn.Module):
    def __init__(
        self,hm_weight=1.,wh_weight=.1,offset_weight=1.,
        wh_loss_type="hybrid_abs_relative",
        wh_absolute_fraction=.75,
        wh_relative_fraction=.25,
        wh_relative_eps=2.,
        wh_relative_beta=.2,
        wh_relative_scale=10.,
        neighbor_lambda=0.
    ):
        super().__init__()
        self.hm_weight=float(hm_weight)
        self.wh_weight=float(wh_weight)
        self.offset_weight=float(offset_weight)
        self.wh_loss_type=wh_loss_type
        self.wh_absolute_fraction=float(wh_absolute_fraction)
        self.wh_relative_fraction=float(wh_relative_fraction)
        self.wh_relative_eps=float(wh_relative_eps)
        self.wh_relative_beta=float(wh_relative_beta)
        self.wh_relative_scale=float(wh_relative_scale)
        self.neighbor_lambda=float(neighbor_lambda)

        assert self.wh_loss_type=="hybrid_abs_relative"
        assert abs(
            self.wh_absolute_fraction
            +self.wh_relative_fraction-1.
        )<1e-12

    def forward(self,outputs,targets):
        hm = centernet_focal_loss(
            outputs["heatmap"],targets["heatmap"]
        )

        wp = transpose_and_gather_feature(
            outputs["wh"],targets["indices"]
        )
        op = transpose_and_gather_feature(
            outputs["offset"],targets["indices"]
        )

        wh_abs = masked_l1_loss(
            wp,targets["wh"],targets["mask"]
        )
        wh_rel = masked_relative_smooth_l1_loss(
            wp,targets["wh"],targets["mask"],
            self.wh_relative_eps,
            self.wh_relative_beta,
            self.wh_relative_scale
        )

        wh_center = (
            self.wh_absolute_fraction*wh_abs
            +self.wh_relative_fraction*wh_rel
        )

        off = masked_l1_loss(
            op,targets["offset"],targets["mask"]
        )

        neighbor_loss = wp.sum()*0.

        if self.neighbor_lambda>0.:
            ix = targets["neighbor_indices"]
            nb_mask = (
                targets["neighbor_mask"]
                & targets["mask"].unsqueeze(-1)
            )

            nb_pred = transpose_and_gather_feature(
                outputs["wh"],ix.reshape(ix.shape[0],-1)
            ).float().reshape(*ix.shape,2)

            nb_true = targets["wh"].float().unsqueeze(2)
            nb_true = nb_true.expand_as(nb_pred)

            abs_per_neighbor = (
                nb_pred-nb_true
            ).abs().mean(-1)

            relative_error = (
                (nb_pred-nb_true)
                /nb_true.clamp_min(self.wh_relative_eps)
            )

            rel_per_neighbor = (
                self.wh_relative_scale
                *F.smooth_l1_loss(
                    relative_error,
                    torch.zeros_like(relative_error),
                    beta=self.wh_relative_beta,
                    reduction="none"
                ).mean(-1)
            )

            hybrid = (
                self.wh_absolute_fraction*abs_per_neighbor
                +self.wh_relative_fraction*rel_per_neighbor
            )

            valid_count = nb_mask.sum(-1)

            per_object = (
                (hybrid*nb_mask.float()).sum(-1)
                /valid_count.clamp_min(1)
            )

            neighbor_loss = (
                (per_object*targets["mask"].float()).sum()
                /targets["mask"].sum().clamp_min(1)
            )

        wh = wh_center+self.neighbor_lambda*neighbor_loss

        total = (
            self.hm_weight*hm
            +self.wh_weight*wh
            +self.offset_weight*off
        )

        return {
            "loss_total":total,
            "loss_heatmap":hm,
            "loss_wh":wh,
            "loss_offset":off,
            "loss_wh_center":wh_center,
            "loss_wh_neighbor":neighbor_loss
        }

criterion = CenterNetLoss(
    hm_weight=CONFIG["hm_loss_weight"],
    wh_weight=CONFIG["wh_loss_weight"],
    offset_weight=CONFIG["offset_loss_weight"],
    wh_loss_type=CONFIG["wh_loss_type"],
    wh_absolute_fraction=CONFIG["wh_absolute_fraction"],
    wh_relative_fraction=CONFIG["wh_relative_fraction"],
    wh_relative_eps=CONFIG["wh_relative_eps"],
    wh_relative_beta=CONFIG["wh_relative_beta"],
    wh_relative_scale=CONFIG["wh_relative_scale"],
    neighbor_lambda=0.
).to(device)

print("STEP 24 PASS — A2/A3 LOSS READY")
