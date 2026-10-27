
# STEP 34 — Paired GT diagnostic, one-to-one top100
import numpy as np
import torch
import gc

QC_RECORDS = {}
QC_COUNTS = {}

DIRECTIONS = [
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dx or dy
]

for arm in paths:
    net = qc_model(arm)
    rows = []
    seen = set()

    with torch.inference_mode():
        for images, targets in qc_loader():
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                outputs = net(
                    images.to(
                        device, non_blocking=True
                    )
                )

            p = heatmap_local_nms(
                outputs["heatmap"].float().sigmoid()
            )

            scores, _, classes, ys, xs = topk_centers(
                p, K=100
            )

            whs = outputs["wh"].float().cpu().numpy()
            offs = outputs["offset"].float().cpu().numpy()

            sc = scores.cpu().numpy()
            cl = classes.cpu().numpy()
            xx = xs.cpu().numpy().astype(int)
            yy = ys.cpu().numpy().astype(int)

            for b, t in enumerate(targets):
                imgid = int(t["image_id"])
                assert imgid not in seen
                seen.add(imgid)

                enc = build_centernet_target(
                    t,
                    gaussian_radius_mode="v1_2_original"
                )

                act = torch.nonzero(
                    enc["mask"]
                ).flatten().tolist()

                candidates = []
                centers = []

                for k in act:
                    gy, gx = divmod(
                        int(enc["indices"][k]),
                        OUTPUT_WIDTH
                    )

                    gt_wh = (
                        enc["wh"][k]
                        .numpy()
                        .astype(float)
                    )
                    gt_off = (
                        enc["offset"][k]
                        .numpy()
                        .astype(float)
                    )

                    label = int(enc["labels"][k])

                    true_box = qc_box(
                        gx + gt_off[0],
                        gy + gt_off[1],
                        *gt_wh
                    )

                    centers.append((
                        k, gx, gy, label,
                        gt_wh, gt_off, true_box
                    ))

                # Greedy, one-to-one class-correct matches
                for i, (
                    _, gx, gy, label, _, _, _
                ) in enumerate(centers):

                    for j in range(len(sc[b])):
                        if (
                            int(cl[b, j]) != label
                            or float(sc[b, j]) < 0.05
                        ):
                            continue

                        d = max(
                            abs(gx - int(xx[b, j])),
                            abs(gy - int(yy[b, j]))
                        )

                        if d <= 4:
                            candidates.append((
                                d,
                                -float(sc[b, j]),
                                i,
                                j
                            ))

                matches = {}
                used_p = set()
                used_gt = set()

                for d, negscore, i, j in sorted(candidates):
                    if i not in used_gt and j not in used_p:
                        matches[i] = (d, j)
                        used_gt.add(i)
                        used_p.add(j)

                scale = float(t["scale"])

                for i, (
                    k, gx, gy, label,
                    gt_wh, gt_off, gt_box
                ) in enumerate(centers):

                    cw = whs[b, :, gy, gx]
                    co = offs[b, :, gy, gx]

                    row = {
                        "id": f"{imgid}:{k}",
                        "img": imgid,
                        "class": label,
                        "person": label == 0,

                        "center_wh_iou": qc_wh_iou(
                            cw, gt_wh
                        ),

                        "center_box_iou": qc_box_iou(
                            qc_box(
                                gx + co[0],
                                gy + co[1],
                                *cw
                            ),
                            gt_box
                        ),

                        "center_wh_mae": float(
                            np.abs(cw - gt_wh).mean()
                        ),

                        "center_offset_mae": float(
                            np.abs(co - gt_off).mean()
                        ),

                        "wh_bias_w": float(
                            (cw[0] - gt_wh[0])
                            / max(2.0, gt_wh[0])
                        ),

                        "wh_bias_h": float(
                            (cw[1] - gt_wh[1])
                            / max(2.0, gt_wh[1])
                        )
                    }

                    area = (
                        float(gt_wh.prod())
                        * (OUTPUT_STRIDE / scale) ** 2
                    )

                    row["group"] = (
                        "small" if area < 1024
                        else "medium" if area < 9216
                        else "large"
                    )

                    row["radius_v12"] = int(
                        gaussian_radius(
                            gt_wh[1],
                            gt_wh[0],
                            min_overlap=0.7,
                            mode="v1_2_original"
                        )
                    )

                    row["matched"] = i in matches
                    row["peak_r1"] = False
                    row["peak_r2"] = False
                    row["peak_r4"] = False
                    row["peak_wh_iou"] = None
                    row["peak_box_iou"] = None
                    row["peak_distance"] = None

                    if i in matches:
                        d, j = matches[i]

                        px = int(xx[b, j])
                        py = int(yy[b, j])

                        pw = whs[b, :, py, px]
                        po = offs[b, :, py, px]

                        row["peak_distance"] = int(d)
                        row["peak_r1"] = d <= 1
                        row["peak_r2"] = d <= 2
                        row["peak_r4"] = True

                        row["peak_wh_iou"] = qc_wh_iou(
                            pw, gt_wh
                        )

                        row["peak_box_iou"] = qc_box_iou(
                            qc_box(
                                px + po[0],
                                py + po[1],
                                *pw
                            ),
                            gt_box
                        )

                    for radius in (1, 2, 4):
                        all_values = []
                        inside_values = []

                        for dx, dy in DIRECTIONS:
                            px = gx + radius * dx
                            py = gy + radius * dy

                            if (
                                0 <= px < OUTPUT_WIDTH
                                and 0 <= py < OUTPUT_HEIGHT
                            ):
                                v = qc_wh_iou(
                                    whs[b, :, py, px],
                                    gt_wh
                                )

                                all_values.append(v)

                                if (
                                    gt_box[0] <= px + 0.5 <= gt_box[2]
                                    and gt_box[1] <= py + 0.5 <= gt_box[3]
                                ):
                                    inside_values.append(v)

                        row[f"r{radius}_all"] = (
                            float(np.mean(all_values))
                            if all_values else None
                        )

                        row[f"r{radius}_inside"] = (
                            float(np.mean(inside_values))
                            if inside_values else None
                        )

                    rows.append(row)

    assert len(seen) == 1000
    assert len(rows) > 0

    QC_RECORDS[arm] = rows

    QC_COUNTS[arm] = {
        "images": len(seen),
        "GT": len(rows)
    }

    print(arm, QC_COUNTS[arm])

    del net
    gc.collect()
    torch.cuda.empty_cache()

assert (
    [r["id"] for r in QC_RECORDS["A2_control"]]
    ==
    [r["id"] for r in QC_RECORDS[B2_ARM]]
)

qc_save(
    "qc34_paired_records.json",
    QC_RECORDS
)

qc_save(
    "qc34_counts.json",
    QC_COUNTS
)

print("STEP 34 PASS — GT aligned, one-to-one peak matching")
