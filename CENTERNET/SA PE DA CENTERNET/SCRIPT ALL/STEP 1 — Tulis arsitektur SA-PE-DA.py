%%writefile sape_da/model.py
"""LOCKED COCO-80 SA-PE-DA CenterNet: CenterNet ResDCN-101 + proposed Fusion and FiLM.

Original checkpoint: Xingyi Zhou et al., CenterNet Objects as Points, ctdet_coco_resdcn101.
NO silent missing checkpoint tensors: loading requires all backbone, neck and head keys.
"""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.ops import deform_conv2d


class ModulatedDCN(nn.Module):
    """Parameter/state_dict compatible with original CenterNet DCNv2.DCN.

    Original DCNv2 uses 2 groups of offset channels (o1, o2) and one mask;
    torchvision deform_conv2d supports the same modulated forward with mask.
    This implementation avoids compiling the legacy custom extension.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.conv_offset_mask = nn.Conv2d(in_channels, 27, 3, padding=1)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = out.chunk(3, dim=1)
        return deform_conv2d(x, torch.cat([o1, o2], dim=1), self.weight,
                             bias=self.bias, padding=1, mask=mask.sigmoid())


class CenterNetResDCN101(nn.Module):
    """Matches key names and tensor shapes of the original resdcn_101 implementation.

    C2=layer1 (stride4, 256c), C3=layer2 (stride8, 512c),
    C4=layer3 (stride16, 1024c), C5=layer4 (stride32, 2048c).
    Deconvolution channels: 256 -> 128 -> 64, stride4.
    """
    def __init__(self):
        super().__init__()
        backbone = resnet101(weights=None)
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(self, name, getattr(backbone, name))
        self.deconv_layers = nn.Sequential(
            ModulatedDCN(2048, 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ModulatedDCN(256, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ModulatedDCN(128, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        def head(ch):
            return nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, ch, 1))
        self.hm = head(80)  # official COCO contiguous categories (person index 0)
        self.wh = head(2)
        self.reg = head(2)

    def feature_maps(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        base = self.deconv_layers(c5)
        return (c2, c3, c4), base

    def heads(self, feat):
        return {'hm': self.hm(feat), 'wh': self.wh(feat), 'reg': self.reg(feat)}

    def forward(self, x):
        _, feat = self.feature_maps(x)
        return self.heads(feat)


def load_original_coco80_checkpoint(base: CenterNetResDCN101, checkpoint_path: str):
    """Require exact tensors from original 80-class checkpoint; KEEP all 80 channels.

    Never accepts backbone-only ImageNet weights as a replacement.
    The original official CenterNet checkpoint has keys 'state_dict', 'hm.2.weight', etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Official COCO checkpoint not found: {checkpoint_path}')
    # Checkpoint downloaded exclusively from official CenterNet model zoo; loading pickle file is trusted source.
    try:
        obj = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError('Checkpoint incompatible or unreadable; re-download official COCO resdcn101.') from e
    raw = obj.get('state_dict', obj)
    raw = {k.removeprefix('module.'): v for k,v in raw.items()}
    for head in ('hm.2.weight', 'hm.2.bias'):
        if head not in raw or raw[head].shape[0] != 80:
            raise ValueError(f'Checkpoint is not original COCO-80: {head} has invalid shape')
    person = dict(raw)  # retain ALL 80 pretrained heatmap channels
    expected = base.state_dict()
    # Historical PyTorch 0.4 checkpoints may omit BatchNorm counters.
    # Permit ONLY these non-learned buffers; all actual weights remain strict.
    missing_before = set(expected) - set(person)
    legacy_bn = [k for k in missing_before if k.endswith('.num_batches_tracked')]
    for k in legacy_bn:
        person[k] = expected[k].clone()
    if legacy_bn:
        print(f'Filled {len(legacy_bn)} legacy BatchNorm counters (non-learned buffers).')
    missing = set(expected) - set(person)
    extra = set(person) - set(expected)
    wrong = {k:(tuple(expected[k].shape), tuple(person[k].shape)) for k in expected.keys() & person.keys()
             if expected[k].shape != person[k].shape}
    if missing or extra or wrong:
        raise RuntimeError(f'CHECKPOINT STRICT VALIDATION FAILED. Missing={sorted(missing)[:12]}; '
                           f'Extra={sorted(extra)[:12]}; shape mismatch={list(wrong.items())[:10]}. '
                           'Do not train until architecture and checkpoint agree.')
    base.load_state_dict(person, strict=True)
    print(f'Loaded verified COCO-80 CenterNet ResDCN101 checkpoint; '
          f'ALL 80 classes retained; {len(expected)} tensors matched.')


class AdaptiveSpatialFusion(nn.Module):
    """C2/C3/C4 => stride-4; shared 64 channels; spatial softmax over scales.

    Unlike global scale attention, alpha_i is [B,1,H/4,W/4].
    Residual gate starts at zero for exact pretrained baseline at initialization.
    """
    def __init__(self, fusion_channels=64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, fusion_channels, 1, bias=True) for ch in (256,512,1024)])
        self.logit_heads = nn.ModuleList([
            nn.Conv2d(fusion_channels, 1, 1) for _ in range(3)])
        self.output = nn.Conv2d(fusion_channels, 64, 1)
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, c2c3c4, reference):
        spatial = reference.shape[-2:]
        feats = [F.interpolate(proj(c), size=spatial, mode='bilinear', align_corners=False)
                 for proj,c in zip(self.projections, c2c3c4)]
        logits = torch.stack([score(f) for score,f in zip(self.logit_heads, feats)], dim=1)
        weights = logits.softmax(dim=1)
        mixed = sum(weights[:,i] * feats[i] for i in range(3))
        residual = self.output(mixed)
        return reference + self.eta * residual


class FiLMAdapter(nn.Module):
    """Spatially shared per-channel affine correction; exactly identity at initialization."""
    def __init__(self, channels=64):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        return x * (1 + self.g) + self.b
    def regularizer(self):
        return self.g.square().mean() + self.b.square().mean()


class SAPEDACenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CenterNetResDCN101()
        self.fusion = AdaptiveSpatialFusion()
        self.adapter = FiLMAdapter(64)

    def forward(self, image):
        c, base_feature = self.base.feature_maps(image)
        fs = self.fusion(c, base_feature)
        ft = self.adapter(fs)
        return self.base.heads(ft)

    def freeze_phase(self, phase):
        if phase not in ('source','target','eval'):
            raise ValueError('phase must be source, target, or eval')
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 'source':
            for p in self.fusion.parameters(): p.requires_grad_(True)
            for head in (self.base.hm, self.base.wh, self.base.reg):
                for p in head.parameters(): p.requires_grad_(True)
        if phase == 'target':
            for p in self.adapter.parameters(): p.requires_grad_(True)
        # keep all BatchNorm statistics fixed in BOTH training phases
        self.eval()
        if phase == 'source': self.fusion.train()
        if phase == 'target': self.adapter.train()
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Phase={phase}, trainable params={n:,}')
        return n

    def train(self, mode=True):
        super().train(mode)
        # enforce frozen BatchNorm / feature extractor inference stats on every train() call
        self.base.eval()
        # fusion contains no BN; use train/eval only if its params are trainable
        return self


@torch.no_grad()
def decode_detections(outputs, image_height, image_width, score_threshold=0.1, k=100):
    """CenterNet local-max Top-K, no IoU-NMS. Coordinates in resized/letterboxed INPUT space.
    Returns list[Tensor[N,5]] (xyxy + score). Unpadding handled by external preprocess metadata.
    """
    heat = outputs['hm'][:, :1].sigmoid()  # target CCTV person channel only
    pooled = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (pooled == heat)
    b,_,h,w = heat.shape
    if b == 0: return []
    scores, indices = heat.flatten(1).topk(min(k,h*w), dim=1)
    ys = torch.div(indices, w, rounding_mode='floor')
    xs = indices % w
    hw = outputs['wh'].permute(0,2,3,1).reshape(b,h*w,2)
    reg = outputs['reg'].permute(0,2,3,1).reshape(b,h*w,2)
    wh = torch.gather(hw, 1, indices[...,None].expand(-1,-1,2)).clamp(min=0)
    offsets = torch.gather(reg,1,indices[...,None].expand(-1,-1,2))
    stride_x, stride_y = image_width/w, image_height/h
    cx = (xs+offsets[...,0])*stride_x
    cy = (ys+offsets[...,1])*stride_y
    bw,bh = wh[...,0]*stride_x,wh[...,1]*stride_y
    boxes = torch.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2,scores],dim=-1)
    boxes[...,0] = boxes[...,0].clamp(0,image_width)
    boxes[...,2] = boxes[...,2].clamp(0,image_width)
    boxes[...,1] = boxes[...,1].clamp(0,image_height)
    boxes[...,3] = boxes[...,3].clamp(0,image_height)
    return [bb[bb[:,4]>=score_threshold] for bb in boxes]


@torch.no_grad()
def decode_coco80(outputs, image_height, image_width, score_threshold=0.001, k=100):
    """All-class local-peak Top-K decoder, Nx6: xyxy, score, contiguous class idx.

    Top-K across ALL 80 class/position pairs, not per-class Top-K 100 (which
    can generate 8000 detections per frame and break COCO maxDets=100).
    """
    heat = outputs['hm'].sigmoid()
    if heat.shape[1] != 80:
        raise AssertionError(f'Source decoder requires exactly 80 heatmap classes: {heat.shape}')
    peak = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (peak == heat)
    batch, ncls, height, width = heat.shape
    scores, inds = heat.reshape(batch, -1).topk(min(k, ncls*height*width), dim=1)
    class_idx = torch.div(inds, height*width, rounding_mode='floor')
    pos = inds % (height*width)
    ys = torch.div(pos, width, rounding_mode='floor')
    xs = pos % width
    wh = outputs['wh'].permute(0,2,3,1).reshape(batch,height*width,2)
    off = outputs['reg'].permute(0,2,3,1).reshape(batch,height*width,2)
    wh = torch.gather(wh,1,pos[...,None].expand(-1,-1,2)).clamp(min=0)
    off = torch.gather(off,1,pos[...,None].expand(-1,-1,2))
    sx,sy=image_width/width,image_height/height
    cx=(xs+off[...,0])*sx;cy=(ys+off[...,1])*sy
    bw=wh[...,0]*sx;bh=wh[...,1]*sy
    boxes=torch.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2,scores,class_idx.float()],dim=-1)
    boxes[...,0]=boxes[...,0].clamp(0,image_width)
    boxes[...,2]=boxes[...,2].clamp(0,image_width)
    boxes[...,1]=boxes[...,1].clamp(0,image_height)
    boxes[...,3]=boxes[...,3].clamp(0,image_height)
    return [d[d[:,4]>=score_threshold] for d in boxes]
