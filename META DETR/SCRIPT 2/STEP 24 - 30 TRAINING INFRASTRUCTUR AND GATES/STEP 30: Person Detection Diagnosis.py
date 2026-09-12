# ==========================================================
# STEP 30: Person Detection Diagnosis
#
# Diagnosis:
# 1. Localization
# 2. Confidence
# 3. Consistency under different PERSON supports
#
# Memakai fungsi person AP yang sama dengan Step 25.
# Tidak memakai negative semantic class/ranking.
# ==========================================================

if TINY_GATE_PASSED:
    diagnostic_model = short_model.to(
        CONFIG["device"]
    )

    diagnostic_dataset = val_dataset

    diagnostic_indices = list(
        range(
            min(
                160,
                len(val_dataset)
            )
        )
    )

    diagnostic_stage = (
        "best short model / fixed COCO-Val subset"
    )

else:
    train_dataset.set_epoch(0)

    diagnostic_model = tiny_model.to(
        CONFIG["device"]
    )

    diagnostic_dataset = train_dataset
    diagnostic_indices = tiny_indices

    diagnostic_stage = (
        "tiny model / same tiny training episodes"
    )

diagnostic_model.eval()

records_a = []
records_b = []
changes = []

with torch.inference_mode():
    for index in tqdm(
        diagnostic_indices,
        desc="Person diagnosis"
    ):
        episode = diagnostic_dataset[
            index
        ]

        query_id = episode[
            "query_target"
        ]["image_id"].item()

        support_id_a = episode[
            "support_target"
        ]["image_id"].item()

        # Support B tetap person, dari gambar ketiga.
        eligible_images = diagnostic_dataset[
            "class_to_img_ids"
        ][0] if isinstance(
            diagnostic_dataset,
            dict
        ) else diagnostic_dataset.class_to_img_ids[0]

        excluded = {
            query_id,
            support_id_a,
        }

        alternative_images = [
            image_id
            for image_id in eligible_images
            if image_id not in excluded
        ]

        if not alternative_images:
            raise RuntimeError(
                "Need at least three person images "
                "for independent support diagnosis."
            )

        rng = random.Random(
            CONFIG["seed"]
            + 3000000
            + index
        )

        alternate_image = rng.choice(
            alternative_images
        )

        alternate_ann = rng.choice(
            diagnostic_dataset.image_to_ann_ids[
                alternate_image
            ]
        )

        support_b, _ = diagnostic_dataset._load_support(
            alternate_ann
        )

        query = (
            episode["query_image"]
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        support_a = (
            episode["support_image"]
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        # Gunakan representasi query yang sama
        # untuk kedua kondisi support.
        d, _ = diagnostic_model.encode_query(
            query
        )

        pa = diagnostic_model.encode_support(
            support_a
        )

        pb = diagnostic_model.encode_support(
            support_b
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        out_a, _ = diagnostic_model.condition_and_predict(
            d,
            pa
        )

        out_b, _ = diagnostic_model.condition_and_predict(
            d,
            pb
        )

        for records, output in (
            (records_a, out_a),
            (records_b, out_b),
        ):
            records.append({
                "scores": as_numpy(
                    output["pred_logits"][
                        0, :, 0
                    ].sigmoid()
                ),

                "pred_boxes": as_numpy(
                    output["pred_boxes"][0]
                ),

                "gt_boxes": as_numpy(
                    episode["query_target"]["boxes"]
                ),
            })

        changes.append({
            "logit_change": (
                out_a["pred_logits"]
                - out_b["pred_logits"]
            ).abs().mean().item(),

            "bbox_change": (
                out_a["pred_boxes"]
                - out_b["pred_boxes"]
            ).abs().mean().item(),
        })

        if index == diagnostic_indices[0]:
            preview_episode = episode
            preview_record = records_a[-1]


metrics_a = compute_person_metrics(
    records_a,
    score_threshold=TRAIN_CONFIG[
        "score_threshold"
    ],
)

metrics_b = compute_person_metrics(
    records_b,
    score_threshold=TRAIN_CONFIG[
        "score_threshold"
    ],
)

diagnosis = {
    "stage": diagnostic_stage,

    "support_a": metrics_a,
    "support_b": metrics_b,

    "mean_logit_change": float(
        np.mean([
            item["logit_change"]
            for item in changes
        ])
    ),

    "mean_bbox_change": float(
        np.mean([
            item["bbox_change"]
            for item in changes
        ])
    ),
}

print(
    json.dumps(
        diagnosis,
        indent=2
    )
)

print(
    "A and B are both person support. "
    "Neither is a wrong semantic class."
)

print(
    "AP on this diagnostic subset is not directly comparable "
    "with AP on all validation episodes."
)


# ==========================================================
# Visualisasi: GT hijau dan top-10 prediksi merah.
# Tetap menampilkan prediksi meskipun skor di bawah 0.5.
# ==========================================================

fig, axis = plt.subplots(
    figsize=(9, 9)
)

axis.imshow(
    display_tensor_image(
        preview_episode["query_image"]
    )
)

height, width = preview_episode[
    "query_image"
].shape[-2:]

for cx, cy, w, h in preview_record[
    "gt_boxes"
]:
    axis.add_patch(
        Rectangle(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
            ),
            w * width,
            h * height,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
    )

top_prediction_ids = np.argsort(
    -preview_record["scores"]
)[:10]

for pred_id in top_prediction_ids:
    cx, cy, w, h = preview_record[
        "pred_boxes"
    ][pred_id]

    x = (
        cx - w / 2
    ) * width

    y = (
        cy - h / 2
    ) * height

    axis.add_patch(
        Rectangle(
            (x, y),
            w * width,
            h * height,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
    )

    score = preview_record[
        "scores"
    ][pred_id]

    axis.text(
        x,
        y,
        f"{score:.3f}",
        color="red",
        bbox={
            "facecolor": "white",
            "alpha": 0.7,
            "pad": 1,
        },
    )

axis.set_xlim(
    0,
    width
)

axis.set_ylim(
    height,
    0
)

axis.set_title(
    "Person diagnosis: green GT; red top-10 predictions"
)

axis.axis("off")

plt.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "person_diagnostic_boxes.png"
    ),
    dpi=150,
)

plt.show()
plt.close(fig)

with open(
    os.path.join(
        LOG_DIR,
        "person_diagnosis.json"
    ),
    "w"
) as handle:
    json.dump(
        diagnosis,
        handle,
        indent=2
    )

diagnostic_model.cpu()

del d, pa, pb
del out_a, out_b, output
del query, support_a, _

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("STEP 30 COMPLETE.")

print(
    "Review these results before full source training "
    "or CCTV adaptation."
)
