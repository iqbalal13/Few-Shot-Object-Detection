# ==========================================================
# STEP 25: Person AP50 and Evaluation
#
# Satu definisi AP dipakai pada seluruh diagnosis.
# AP: menggunakan semua skor.
# Precision/recall: menggunakan score threshold.
#
# Ini custom episodic AP, bukan official COCOeval.
# ==========================================================

from scipy.optimize import linear_sum_assignment


def as_numpy(value):
    if hasattr(value, "detach"):
        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value,
        dtype=np.float64
    )


def numpy_box_iou(
    boxes_a,
    boxes_b,
):
    a = as_numpy(
        boxes_a
    ).reshape(-1, 4)

    b = as_numpy(
        boxes_b
    ).reshape(-1, 4)

    # Normalized cxcywh -> xyxy
    a = np.concatenate(
        (
            a[:, :2] - a[:, 2:] / 2,
            a[:, :2] + a[:, 2:] / 2,
        ),
        axis=1,
    )

    b = np.concatenate(
        (
            b[:, :2] - b[:, 2:] / 2,
            b[:, :2] + b[:, 2:] / 2,
        ),
        axis=1,
    )

    wh = np.maximum(
        0,
        (
            np.minimum(
                a[:, None, 2:],
                b[None, :, 2:]
            )
            -
            np.maximum(
                a[:, None, :2],
                b[None, :, :2]
            )
        )
    )

    intersection = wh.prod(
        axis=-1
    )

    area_a = np.maximum(
        0,
        a[:, 2:] - a[:, :2]
    ).prod(
        axis=-1
    )

    area_b = np.maximum(
        0,
        b[:, 2:] - b[:, :2]
    ).prod(
        axis=-1
    )

    union = (
        area_a[:, None]
        + area_b[None, :]
        - intersection
    )

    return intersection / np.maximum(
        union,
        1e-6
    )


def greedy_match_flags(
    scores,
    pred_boxes,
    gt_boxes,
    iou_threshold,
):
    scores = as_numpy(
        scores
    ).reshape(-1)

    ious = numpy_box_iou(
        pred_boxes,
        gt_boxes
    )

    used = set()

    flags = np.zeros(
        len(scores),
        dtype=np.float64
    )

    for pred_id in np.argsort(
        -scores,
        kind="stable"
    ):
        for gt_id in np.argsort(
            -ious[pred_id],
            kind="stable"
        ):
            if (
                gt_id not in used
                and ious[pred_id, gt_id] >= iou_threshold
            ):
                used.add(
                    int(gt_id)
                )

                flags[pred_id] = 1.0
                break

    return flags


def compute_person_metrics(
    episodes,
    score_threshold=0.5,
    iou_threshold=0.5,
):
    all_scores = []
    all_flags = []
    best_ious = []
    maximum_scores = []

    total_gt = 0
    tp = 0
    fp = 0
    fn = 0
    geometry_hits = 0

    for episode in episodes:
        scores = as_numpy(
            episode["scores"]
        ).reshape(-1)

        boxes = as_numpy(
            episode["pred_boxes"]
        ).reshape(-1, 4)

        gt = as_numpy(
            episode["gt_boxes"]
        ).reshape(-1, 4)

        assert len(scores) == len(boxes)

        assert all(
            np.isfinite(value).all()
            for value in (
                scores,
                boxes,
                gt,
            )
        )

        total_gt += len(gt)

        # Semua prediksi untuk AP.
        flags = greedy_match_flags(
            scores,
            boxes,
            gt,
            iou_threshold
        )

        all_scores.extend(
            scores.tolist()
        )

        all_flags.extend(
            flags.tolist()
        )

        # Prediksi di atas cutoff untuk precision/recall.
        keep = (
            scores >= score_threshold
        )

        kept_flags = greedy_match_flags(
            scores[keep],
            boxes[keep],
            gt,
            iou_threshold
        )

        episode_tp = int(
            kept_flags.sum()
        )

        tp += episode_tp
        fp += int(keep.sum()) - episode_tp
        fn += len(gt) - episode_tp

        maximum_scores.append(
            float(scores.max())
            if len(scores)
            else 0.0
        )

        ious = numpy_box_iou(
            boxes,
            gt
        )

        if len(gt):
            best_ious.extend(
                ious.max(axis=0).tolist()
                if len(boxes)
                else [0.0] * len(gt)
            )

        if len(boxes) and len(gt):
            # Confidence-free, one-to-one localization.
            # Prioritas: memaksimalkan jumlah pasangan
            # yang mencapai IoU threshold.
            reward = (
                (ious >= iou_threshold).astype(float)
                + ious / (min(ious.shape) + 1)
            )

            rows, cols = linear_sum_assignment(
                -reward
            )

            geometry_hits += int(
                (
                    ious[rows, cols]
                    >= iou_threshold
                ).sum()
            )

    scores = np.asarray(
        all_scores
    )

    flags = np.asarray(
        all_flags
    )

    ap = 0.0

    if total_gt and len(scores):
        flags_sorted = flags[
            np.argsort(
                -scores,
                kind="stable"
            )
        ]

        cumulative_tp = np.cumsum(
            flags_sorted
        )

        recall = (
            cumulative_tp / total_gt
        )

        precision = (
            cumulative_tp
            /
            np.arange(
                1,
                len(flags_sorted) + 1
            )
        )

        mrec = np.r_[
            0.0,
            recall,
            1.0
        ]

        mpre = np.r_[
            0.0,
            precision,
            0.0
        ]

        mpre = np.maximum.accumulate(
            mpre[::-1]
        )[::-1]

        changes = np.where(
            mrec[1:] != mrec[:-1]
        )[0]

        ap = float(
            np.sum(
                (
                    mrec[changes + 1]
                    - mrec[changes]
                )
                *
                mpre[changes + 1]
            )
        )

    iou_values = np.asarray(
        best_ious
    )

    mean_tp_score = (
        float(
            scores[flags == 1].mean()
        )
        if flags.sum()
        else None
    )

    mean_fp_score = (
        float(
            scores[flags == 0].mean()
        )
        if (flags == 0).any()
        else None
    )

    return {
        "person_ap50": ap,

        "precision50": (
            tp / max(tp + fp, 1)
        ),

        "recall50": (
            tp / max(total_gt, 1)
        ),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "episodes": len(episodes),

        "mean_best_iou": (
            float(iou_values.mean())
            if len(iou_values)
            else 0.0
        ),

        "median_best_iou": (
            float(np.median(iou_values))
            if len(iou_values)
            else 0.0
        ),

        "localization_recall50": (
            float(
                (
                    iou_values >= iou_threshold
                ).mean()
            )
            if len(iou_values)
            else 0.0
        ),

        "one_to_one_localization_recall50": (
            geometry_hits / max(total_gt, 1)
        ),

        "mean_max_score": (
            float(np.mean(maximum_scores))
            if maximum_scores
            else 0.0
        ),

        "mean_tp_score_all_predictions": mean_tp_score,
        "mean_fp_score_all_predictions": mean_fp_score,
    }


def evaluate_episodic_model(
    model,
    data_loader,
    criterion,
    device,
    score_threshold=0.5,
    iou_threshold=0.5,
    show_progress=True,
):
    assert iou_threshold == 0.5, (
        "This revision reports AP50 only."
    )

    model.eval()

    episodes = []
    total_loss = 0.0
    batches = 0

    iterator = (
        tqdm(
            data_loader,
            desc="Person validation"
        )
        if show_progress
        else data_loader
    )

    with torch.inference_mode():
        for batch in iterator:
            assert (
                batch["episode_classes"] == 0
            ).all()

            targets = move_targets_to_device(
                batch["query_targets"],
                device
            )

            assert all(
                (
                    target["labels"] == 0
                ).all()
                for target in targets
            )

            output = model(
                batch["support_images"].to(device),
                batch["query_images"].to(device),
            )

            loss = criterion(
                output,
                targets
            )["loss_total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation loss."
                )

            total_loss += loss.item()
            batches += 1

            for index, target in enumerate(targets):
                episodes.append({
                    "scores": as_numpy(
                        output["pred_logits"][
                            index, :, 0
                        ].sigmoid()
                    ),

                    "pred_boxes": as_numpy(
                        output["pred_boxes"][index]
                    ),

                    "gt_boxes": as_numpy(
                        target["boxes"]
                    ),
                })

    if not batches:
        raise RuntimeError(
            "Empty evaluation loader."
        )

    return {
        **compute_person_metrics(
            episodes,
            score_threshold,
            iou_threshold
        ),

        "loss": total_loss / batches,
    }


def print_person_metrics(
    title,
    metrics,
):
    print(
        title,
        "| loss:",
        round(metrics["loss"], 4),

        "| AP50: %.4f%%"
        % (
            100 * metrics["person_ap50"]
        ),

        "| P/R: %.4f / %.4f"
        % (
            metrics["precision50"],
            metrics["recall50"]
        ),

        "| mean IoU: %.4f"
        % metrics["mean_best_iou"],

        "| geometry recall: %.4f"
        % metrics["one_to_one_localization_recall50"],
    )


print("STEP 25 READY.")
print("AP uses all scores; P/R use the explicit score cutoff.")
print("Localization diagnostics are confidence-free.")
