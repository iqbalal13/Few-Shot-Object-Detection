
# ============================================================
# STEP 34 — PAIRED 1000-IMAGE QC INFERENCE CACHE
# Runs inference once per arm. No training.
# ============================================================
from collections import Counter
import torch.nn.functional as F

QC34_RECORDS = {}
QC34_COUNTS = {}

QC34_RADIUS_DIRECTIONS = [
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dx != 0 or dy != 0
]

for arm in A3_ARMS:
    net = a3_load_net(arm)
    records = []
    seen_images = set()

    with torch.inference_mode():
        for images, raw_targets in a3_qc_loader():
            with torch.amp.autocast(
                'cuda', dtype=torch.bfloat16
            ):
                raw = net(
                    images.to(
                        device, non_blocking=True
                    )
                )

            hm = raw['heatmap'].float().sigmoid()
            wh = raw['wh'].float()
            off = raw['offset'].float()

            peaks = heatmap_local_nms(hm)

            tk_score, tk_idx, tk_cls, tk_y, tk_x = (
                topk_centers(peaks, K=100)
            )

            for b, target in enumerate(raw_targets):
                image_id = int(target['image_id'])

                assert image_id not in seen_images
                seen_images.add(image_id)

                enc = build_centernet_target(
                    target,
                    gaussian_radius_mode='v1_2_original'
                )

                scale = float(target['scale'])
                gt_boxes = (
                    target['boxes'].cpu().numpy()
                )

                active = torch.nonzero(
                    enc['mask']
                ).flatten().tolist()

                # Reconstruct the same accepted GT order
                # as target generation in Step 16.
                boxes_accepted = []

                for bx in gt_boxes:
                    x1, y1, x2, y2 = map(float, bx)
                    bw, bh = x2 - x1, y2 - y1

                    cx = (
                        x1 + x2
                    ) / (2 * OUTPUT_STRIDE)

                    cy = (
                        y1 + y2
                    ) / (2 * OUTPUT_STRIDE)

                    if bw <= 1 or bh <= 1:
                        continue

                    if not (
                        0 <= cx < OUTPUT_WIDTH
                        and 0 <= cy < OUTPUT_HEIGHT
                    ):
                        continue

                    if len(boxes_accepted) >= CONFIG['max_objects']:
                        break

                    boxes_accepted.append(bx.copy())

                assert len(boxes_accepted) == len(active)

                score_b = (
                    tk_score[b].detach().cpu().numpy()
                )
                cls_b = (
                    tk_cls[b].detach().cpu().numpy()
                )
                x_b = (
                    tk_x[b].detach().cpu().numpy().astype(int)
                )
                y_b = (
                    tk_y[b].detach().cpu().numpy().astype(int)
                )

                # Copy once per image to avoid many GPU syncs.
                hm_b = hm[b].detach().cpu().numpy()
                wh_b = wh[b].detach().cpu().numpy()
                off_b = off[b].detach().cpu().numpy()
                peaks_b = peaks[b].detach().cpu().numpy()

                for k in active:
                    ind = int(enc['indices'][k])
                    gy, gx = divmod(ind, OUTPUT_WIDTH)

                    label = int(enc['labels'][k])
                    true_wh = enc['wh'][k].numpy()
                    bx = np.asarray(
                        boxes_accepted[k], dtype=float
                    )
                    true_off = enc['offset'][k].numpy()

                    center_wh = wh_b[:, gy, gx]
                    center_off = off_b[:, gy, gx]

                    radius = int(
                        gaussian_radius(
                            float(true_wh[1]),
                            float(true_wh[0]),
                            min_overlap=0.7,
                            mode='v1_2_original'
                        )
                    )

                    area = (
                        float(true_wh.prod())
                        * (OUTPUT_STRIDE / scale) ** 2
                    )

                    group = (
                        'small' if area < 1024
                        else 'medium' if area < 9216
                        else 'large'
                    )

                    # Class-correct top100 nearest peak.
                    # Non-exclusive diagnostic matching.
                    candidates = [
                        (
                            max(
                                abs(int(xx) - gx),
                                abs(int(yy) - gy)
                            ),
                            -float(ss),
                            int(xx),
                            int(yy),
                            float(ss)
                        )
                        for xx, yy, cl, ss
                        in zip(x_b, y_b, cls_b, score_b)
                        if int(cl) == label and ss >= 0.05
                    ]

                    candidates.sort()
                    near = candidates[0] if candidates else None

                    if near is not None and near[0] > 4:
                        near = None

                    top_dist = (
                        int(near[0]) if near else None
                    )

                    matched_box_iou = None
                    matched_gtwh_iou = None
                    matched_whoiou = None
                    matched_offset_mae = None

                    if near is not None:
                        _, _, px, py, _ = near

                        whpk = wh_b[:, py, px]
                        ofpk = off_b[:, py, px]

                        matched_whoiou = a3_wh_iou(
                            whpk, true_wh
                        )

                        matched_offset_mae = float(
                            np.abs(
                                ofpk - true_off
                            ).mean()
                        )

                        matched_box = a3_pred_box(
                            (px, py), ofpk, whpk
                        )

                        matched_box_iou = a3_xyxy_iou(
                            matched_box, bx
                        )

                        matched_gtwh_iou = a3_xyxy_iou(
                            a3_pred_box(
                                (px, py), ofpk, true_wh
                            ),
                            bx
                        )

                    neighbor_indices = (
                        enc['neighbor_indices'][k][
                            enc['neighbor_mask'][k]
                        ].tolist()
                    )

                    nb_ious = []

                    for ix in neighbor_indices:
                        py, px = divmod(
                            int(ix), OUTPUT_WIDTH
                        )

                        nb_ious.append(
                            a3_wh_iou(
                                wh_b[:, py, px],
                                true_wh
                            )
                        )

                    radii = {}

                    for rad in (1, 2, 4):
                        all_iou = []
                        inside_iou = []

                        for dx, dy in QC34_RADIUS_DIRECTIONS:
                            px = gx + rad * dx
                            py = gy + rad * dy

                            if not (
                                0 <= px < OUTPUT_WIDTH
                                and 0 <= py < OUTPUT_HEIGHT
                            ):
                                continue

                            v = a3_wh_iou(
                                wh_b[:, py, px],
                                true_wh
                            )

                            all_iou.append(v)

                            if (
                                bx[0] <= OUTPUT_STRIDE * (px + 0.5) <= bx[2]
                                and
                                bx[1] <= OUTPUT_STRIDE * (py + 0.5) <= bx[3]
                            ):
                                inside_iou.append(v)

                        radii[f'r{rad}_all'] = (
                            float(np.mean(all_iou))
                            if all_iou else None
                        )

                        radii[f'r{rad}_inside'] = (
                            float(np.mean(inside_iou))
                            if inside_iou else None
                        )

                        radii[f'r{rad}_inside_count'] = (
                            len(inside_iou)
                        )

                    # Local NMS peak within 3x3 around GT.
                    loc = [
                        float(peaks_b[label, py, px])
                        for dx, dy in (
                            QC34_RADIUS_DIRECTIONS
                            + [(0, 0)]
                        )
                        for px, py in [
                            (gx + dx, gy + dy)
                        ]
                        if (
                            0 <= px < OUTPUT_WIDTH
                            and 0 <= py < OUTPUT_HEIGHT
                        )
                    ]

                    records.append({
                        'id': f'{image_id}:{k}',
                        'image_id': image_id,
                        'gt_index': k,
                        'class': label,
                        'person': label == 0,
                        'group': group,
                        'radius_v12': radius,
                        'radius_zero': radius == 0,
                        'gt_area': area,
                        'neighbor_n': len(nb_ious),

                        'center_iou': a3_wh_iou(
                            center_wh, true_wh
                        ),

                        'neighbor_iou': (
                            float(np.mean(nb_ious))
                            if nb_ious else None
                        ),

                        'center_wh_mae_cells': float(
                            np.abs(
                                center_wh - true_wh
                            ).mean()
                        ),

                        'center_wh_bias_w': float(
                            (center_wh[0] - true_wh[0])
                            / max(true_wh[0], 2)
                        ),

                        'center_wh_bias_h': float(
                            (center_wh[1] - true_wh[1])
                            / max(true_wh[1], 2)
                        ),

                        'center_offset_mae_cells': float(
                            np.abs(
                                center_off - true_off
                            ).mean()
                        ),

                        'center_box_iou': a3_xyxy_iou(
                            a3_pred_box(
                                (gx, gy),
                                center_off,
                                center_wh
                            ),
                            bx
                        ),

                        'center_wh_oracle_box_iou': a3_xyxy_iou(
                            a3_pred_box(
                                (gx, gy),
                                center_off,
                                true_wh
                            ),
                            bx
                        ),

                        'center_offset_oracle_box_iou': a3_xyxy_iou(
                            a3_pred_box(
                                (gx, gy),
                                true_off,
                                center_wh
                            ),
                            bx
                        ),

                        'gt_center_score': float(
                            hm_b[label, gy, gx]
                        ),

                        'local_peak_r1_ge005': any(
                            s >= 0.05 for s in loc
                        ),

                        'top100_r1_ge005': (
                            top_dist is not None
                            and top_dist <= 1
                        ),

                        'top100_r2_ge005': (
                            top_dist is not None
                            and top_dist <= 2
                        ),

                        'top100_r4_ge005': (
                            top_dist is not None
                            and top_dist <= 4
                        ),

                        'top100_nearest_r4_dist': top_dist,

                        'matched_peak_wh_iou': matched_whoiou,
                        'matched_peak_box_iou': matched_box_iou,
                        'matched_peak_perfect_wh_box_iou': matched_gtwh_iou,
                        'matched_peak_offset_mae_to_gt_center': matched_offset_mae,

                        **radii,
                    })

    assert len(seen_images) == A3_QC_IMAGE_COUNT
    assert len(records) > 0

    assert len({
        r['id'] for r in records
    }) == len(records)

    QC34_RECORDS[arm] = records

    QC34_COUNTS[arm] = {
        'images': len(seen_images),
        'gt': len(records),
        'eligible_neighbors_gt': sum(
            r['neighbor_n'] > 0
            for r in records
        ),
    }

    print(arm, QC34_COUNTS[arm])

    del net
    gc.collect()
    torch.cuda.empty_cache()

assert (
    [r['id'] for r in QC34_RECORDS['A2_control']]
    ==
    [r['id'] for r in QC34_RECORDS['A3_neighbor_R1']]
)

a3_save_json(
    'qc34_paired_gt_records.json',
    QC34_RECORDS
)

a3_save_json(
    'qc34_counts.json',
    QC34_COUNTS
)

print(
    'STEP 34 PASS — paired GT records, '
    'A2 continuation vs A3'
)
