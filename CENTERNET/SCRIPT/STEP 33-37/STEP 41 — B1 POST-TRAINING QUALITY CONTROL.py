
# ============================================================
# STEP 41 — B1 vs A2 / COCO-VAL PAIRED QC (NO TRAINING)
# 1000 images, threshold .05, all/small/medium/large/person.
# ============================================================
import os, json, gc
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

QC41_PATHS = {
    'A2_V12_160K': os.path.join(
        '/content/drive/MyDrive',
        'CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8',
        'epoch20_160k.pth'),
    'B1_V13_160K': milestone30(20)
}
QC41_SCORE = 0.05
QC41_IMAGE_COUNT = 1000
QC41_GROUPS = ('all', 'small', 'medium', 'large', 'person')
PERSON41 = int(NAME_TO_CONTIGUOUS['person'])
for label, path in QC41_PATHS.items():
    assert os.path.isfile(path), f'Missing {label}: {path}'


def qc41_size_iou(wh, true_wh):
    pw, ph = [max(0., float(a)) for a in wh]
    tw, th = [float(a) for a in true_wh]
    inter = min(pw, tw) * min(ph, th)
    return inter / max(pw * ph + tw * th - inter, 1e-9)


def qc41_loader():
    return DataLoader(Subset(val_dataset, range(QC41_IMAGE_COUNT)),
                      batch_size=8, shuffle=False, num_workers=2,
                      pin_memory=True, collate_fn=centernet_collate_fn)


def qc41_run_model(path, name):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert int(ckpt['epoch']) == 20
    if name.startswith('B1'):
        assert ckpt['config']['gaussian_radius_mode'] == 'v1_3_corrected'
        assert ckpt['config']['wh_loss_type'] == 'hybrid_abs_relative'
    else:
        assert ckpt['config']['gaussian_radius_mode'] == 'v1_2_original'
        assert ckpt['config']['wh_loss_type'] == 'hybrid_abs_relative'
    net = CenterNetResNet101(num_classes=80, pretrained_backbone=False).to(device)
    net.load_state_dict(ckpt['model_state_dict'], strict=True)
    net.eval()
    del ckpt
    sums = {g: defaultdict(float) for g in QC41_GROUPS}
    seen = 0

    with torch.inference_mode():
        for images, raw_targets in qc41_loader():
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = net(images)
            logits = out['heatmap'].float()
            local = heatmap_local_nms(logits.sigmoid())
            scores, _, classes, ys, xs = topk_centers(local, K=100)
            scores = scores.detach().cpu().numpy()
            classes = classes.detach().cpu().numpy()
            ys = ys.detach().cpu().numpy()
            xs = xs.detach().cpu().numpy()

            for b, raw_target in enumerate(raw_targets):
                # The WH/offset targets are identical for V1.2 and V1.3.
                enc = build_centernet_target(raw_target)
                valid_idx = torch.nonzero(enc['mask']).flatten().tolist()
                scale = float(raw_target['scale'].item())
                peak_cache = {}
                gt_entries = []

                for k in valid_idx:
                    cls = int(enc['labels'][k])
                    cy, cx = divmod(int(enc['indices'][k]), OUTPUT_WIDTH)
                    gt_wh = enc['wh'][k].tolist()
                    area = gt_wh[0] * gt_wh[1] * (OUTPUT_STRIDE / scale) ** 2
                    size = 'small' if area < 32**2 else ('medium' if area < 96**2 else 'large')
                    groups = ['all', size] + (['person'] if cls == PERSON41 else [])

                    # Find nearest class-correct local maximum with score >= 0.05.
                    if cls not in peak_cache:
                        peak_cache[cls] = torch.nonzero(
                            local[b, cls] >= QC41_SCORE, as_tuple=False
                        ).detach().cpu().numpy()
                    coords = peak_cache[cls]
                    if len(coords):
                        dist = np.maximum(abs(coords[:, 0] - cy),
                                          abs(coords[:, 1] - cx))
                        nearest_idx = int(np.argmin(dist))
                        distance = float(dist[nearest_idx])
                        py, px = map(int, coords[nearest_idx])
                    else:
                        distance, py, px = float('inf'), -1, -1

                    center_class_ok = int(int(logits[b, :, cy, cx].argmax()) == cls)
                    exact = int(float(local[b, cls, cy, cx]) >= QC41_SCORE)
                    wh_gtcenter = out['wh'][b, :, cy, cx].float().tolist()
                    iou_gtcenter = qc41_size_iou(wh_gtcenter, gt_wh)
                    topmask = ((classes[b] == cls) & (scores[b] >= QC41_SCORE)
                               & (np.maximum(abs(ys[b] - cy), abs(xs[b] - cx)) <= 1))
                    top100 = int(bool(topmask.any()))

                    for group in groups:
                        st = sums[group]
                        st['n'] += 1
                        st['class_center'] += center_class_ok
                        st['exact'] += exact
                        st['peak1'] += int(distance <= 1)
                        st['peak2'] += int(distance <= 2)
                        st['peak4'] += int(distance <= 4)
                        st['top100'] += top100
                        st['wh_center_sum'] += iou_gtcenter
                        if distance <= 4:
                            wh_atpeak = out['wh'][b, :, py, px].float().tolist()
                            st['pair_n'] += 1
                            st['pair_center_sum'] += iou_gtcenter
                            st['pair_peak_sum'] += qc41_size_iou(wh_atpeak, gt_wh)
                    gt_entries.append((cls, cy, cx, groups))

                # One-to-one top100 matching within +/-1 Chebyshev cell.
                candidates = []
                for gi, (cls, cy, cx, groups) in enumerate(gt_entries):
                    for pi in range(len(scores[b])):
                        if classes[b, pi] != cls or scores[b, pi] < QC41_SCORE:
                            continue
                        dist = max(abs(int(ys[b, pi]) - cy),
                                   abs(int(xs[b, pi]) - cx))
                        if dist <= 1:
                            candidates.append((-float(scores[b, pi]), float(dist), gi, pi))
                candidates.sort()
                used_gt, used_peak = set(), set()
                for _, _, gi, pi in candidates:
                    if gi in used_gt or pi in used_peak:
                        continue
                    used_gt.add(gi)
                    used_peak.add(pi)
                    for group in gt_entries[gi][3]:
                        sums[group]['top100_unique'] += 1

            seen += len(raw_targets)

    assert seen == QC41_IMAGE_COUNT
    result = {'model': name, 'images': seen, 'groups': {}}
    for group in QC41_GROUPS:
        st = sums[group]
        n, paired = int(st['n']), int(st['pair_n'])
        result['groups'][group] = {
            'gt': n, 'class_correct_center_pct': 100*st['class_center']/max(n, 1),
            'exact_peak_pct': 100*st['exact']/max(n, 1),
            'peak1_pct': 100*st['peak1']/max(n, 1),
            'peak2_pct': 100*st['peak2']/max(n, 1),
            'peak4_pct': 100*st['peak4']/max(n, 1),
            'top100_near1_pct': 100*st['top100']/max(n, 1),
            'top100_one_to_one_near1_pct': 100*st['top100_unique']/max(n, 1),
            'wh_center_iou': st['wh_center_sum']/max(n, 1),
            'matched_peak4_n': paired,
            'wh_center_iou_matched_peak4': st['pair_center_sum']/max(paired, 1),
            'wh_peak_iou_matched_peak4': st['pair_peak_sum']/max(paired, 1),
        }
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return result


qc41_results = {}
for label, path in QC41_PATHS.items():
    print('QC41 evaluating:', label)
    qc41_results[label] = qc41_run_model(path, label)
for group in QC41_GROUPS:
    print('\nGROUP:', group)
    for label, res in qc41_results.items():
        x = res['groups'][group]
        print(f'{label:16} GT={x["gt"]:5d} '
              f'peak±1={x["peak1_pct"]:6.2f}% '
              f'peak±2={x["peak2_pct"]:6.2f}% '
              f'peak±4={x["peak4_pct"]:6.2f}% '
              f'Top100±1={x["top100_near1_pct"]:6.2f}% '
              f'1-to-1={x["top100_one_to_one_near1_pct"]:6.2f}% '
              f'WH@GT={x["wh_center_iou"]:.4f}')

save41 = os.path.join(FULL_CHECKPOINT_DIR, 'b1_qc41_comparison.json')
with open(save41, 'w', encoding='utf-8') as f:
    json.dump(qc41_results, f, indent=2, allow_nan=False)
print('STEP 41 PASS | saved:', save41)
print('Note: QC41 local peaks use score >= 0.05; not identical to old QC36.')
