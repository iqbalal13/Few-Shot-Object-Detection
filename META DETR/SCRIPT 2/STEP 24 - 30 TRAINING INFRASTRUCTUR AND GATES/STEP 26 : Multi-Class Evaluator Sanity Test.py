# ==========================================================
# STEP 26: Person Evaluator Sanity Tests
# ==========================================================

gt = np.array([
    [0.5, 0.5, 0.2, 0.2]
])

far = np.array([
    [0.1, 0.1, 0.1, 0.1]
])


def metric_case(
    scores,
    boxes,
    targets,
):
    return {
        "scores": np.array(scores),

        "pred_boxes": np.array(
            boxes
        ).reshape(-1, 4),

        "gt_boxes": np.array(
            targets
        ).reshape(-1, 4),
    }


# Perfect detection.
perfect = compute_person_metrics([
    metric_case(
        [0.9],
        gt,
        gt
    )
])

assert np.isclose(
    perfect["person_ap50"],
    1.0
)

assert perfect["tp"] == 1


# Duplicate detection: satu TP dan satu FP.
duplicate = compute_person_metrics([
    metric_case(
        [0.9, 0.8],
        np.repeat(gt, 2, axis=0),
        gt
    )
])

assert duplicate["tp"] == 1
assert duplicate["fp"] == 1


# Wrong localization.
miss = compute_person_metrics([
    metric_case(
        [0.9],
        far,
        gt
    )
])

assert miss["person_ap50"] == 0
assert miss["fn"] == 1
assert miss["fp"] == 1


# Gambar tanpa GT person.
empty = compute_person_metrics([
    metric_case(
        [0.9],
        far,
        []
    )
])

assert empty["total_gt"] == 0
assert empty["fp"] == 1


# AP dapat positif walaupun skor di bawah cutoff P/R.
low_score = compute_person_metrics([
    metric_case(
        [0.2],
        gt,
        gt
    )
])

assert low_score["person_ap50"] == 1
assert low_score["recall50"] == 0


# False positive dengan skor lebih tinggi daripada TP.
mixed = compute_person_metrics([
    metric_case(
        [0.9],
        far,
        []
    ),
    metric_case(
        [0.8],
        gt,
        gt
    ),
])

assert np.isclose(
    mixed["person_ap50"],
    0.5
)


# Membedakan best-per-GT dan one-to-one localization.
crowded = compute_person_metrics([
    metric_case(
        [0.9],
        gt,
        np.repeat(gt, 2, axis=0)
    )
])

assert crowded["localization_recall50"] == 1

assert (
    crowded["one_to_one_localization_recall50"]
    == 0.5
)


# Tidak ada prediksi.
no_predictions = compute_person_metrics([
    metric_case(
        [],
        [],
        gt
    )
])

assert no_predictions["fn"] == 1
assert no_predictions["person_ap50"] == 0


# ==========================================================
# End-to-end evaluator wrapper dengan dummy detector.
# ==========================================================

class PerfectPersonDummy(nn.Module):
    def forward(
        self,
        support_images,
        query_images,
    ):
        batch_size = len(query_images)

        logits = torch.full(
            (batch_size, 1, 1),
            math.log(9),
            device=query_images.device,
        )

        boxes = torch.tensor(
            [0.5, 0.5, 0.2, 0.2],
            device=query_images.device,
        ).reshape(
            1, 1, 4
        ).repeat(
            batch_size, 1, 1
        )

        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
        }


dummy_batch = {
    "support_images": torch.zeros(
        1, 3, 32, 32
    ),

    "query_images": torch.zeros(
        1, 3, 32, 32
    ),

    "episode_classes": torch.zeros(
        1,
        dtype=torch.long
    ),

    "query_targets": [{
        "boxes": torch.tensor(
            gt,
            dtype=torch.float32
        ),

        "labels": torch.zeros(
            1,
            dtype=torch.long
        ),
    }],
}

dummy_metrics = evaluate_episodic_model(
    model=PerfectPersonDummy(),
    data_loader=[dummy_batch],
    criterion=criterion,
    device=CONFIG["device"],
    show_progress=False,
)

assert np.isclose(
    dummy_metrics["person_ap50"],
    1.0
)

assert dummy_metrics["recall50"] == 1

print("STEP 26 PASS:")
print("- Perfect detection")
print("- Duplicate detection")
print("- Wrong localization")
print("- Empty ground truth")
print("- Low confidence")
print("- Mixed positive/negative images")
print("- One-to-one localization")
print("- End-to-end evaluator wrapper")

del dummy_batch, dummy_metrics
