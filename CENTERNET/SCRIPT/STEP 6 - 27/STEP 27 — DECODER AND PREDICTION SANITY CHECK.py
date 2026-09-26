# ============================================================
# STEP 27 — DECODER / PREDICTION SANITY CHECK
# ============================================================

print("=" * 70)
print("STEP 27 — DECODER / PREDICTION SANITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# Controlled synthetic prediction
# ------------------------------------------------------------

synthetic_heatmap = torch.full(
    (
        1,
        CONFIG["num_classes"],
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH
    ),
    -10.0,
    device=device
)

synthetic_wh = torch.zeros(
    (
        1,
        2,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH
    ),
    device=device
)

synthetic_offset = torch.zeros(
    (
        1,
        2,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH
    ),
    device=device
)


person_class = CONFIG[
    "person_class_index"
]

center_x = 50
center_y = 60


# Very high confidence person center
synthetic_heatmap[
    0,
    person_class,
    center_y,
    center_x
] = 10.0


# Object size in output-map coordinates
synthetic_wh[
    0,
    0,
    center_y,
    center_x
] = 20.0

synthetic_wh[
    0,
    1,
    center_y,
    center_x
] = 40.0


# Fractional center offset
synthetic_offset[
    0,
    0,
    center_y,
    center_x
] = 0.25

synthetic_offset[
    0,
    1,
    center_y,
    center_x
] = 0.50


synthetic_outputs = {
    "heatmap": synthetic_heatmap,
    "wh": synthetic_wh,
    "offset": synthetic_offset
}


synthetic_result = decode_centernet(
    synthetic_outputs,
    K=10,
    score_threshold=0.50
)[0]


print("\nSynthetic decoder test")

print(
    "Detections :",
    len(
        synthetic_result["boxes"]
    )
)

assert len(
    synthetic_result["boxes"]
) >= 1


decoded_box = synthetic_result[
    "boxes"
][0].detach().cpu()

decoded_score = synthetic_result[
    "scores"
][0].detach().cpu()

decoded_label = synthetic_result[
    "labels"
][0].detach().cpu()


print(
    "Box        :",
    decoded_box.tolist()
)

print(
    "Score      :",
    float(decoded_score)
)

print(
    "Class index:",
    int(decoded_label)
)

print(
    "Class name :",
    CONTIGUOUS_TO_NAME[
        int(decoded_label)
    ]
)


# Expected synthetic box:
# center:
# (50.25, 60.50)
#
# WH:
# (20, 40)
#
# bbox on output map:
# x1 = 40.25
# y1 = 40.50
# x2 = 60.25
# y2 = 80.50
#
# * stride 4:
# [161, 162, 241, 322]

expected_box = torch.tensor(
    [
        161.0,
        162.0,
        241.0,
        322.0
    ]
)

assert torch.allclose(
    decoded_box,
    expected_box,
    atol=1e-3
)

assert int(
    decoded_label
) == person_class


# ------------------------------------------------------------
# Visualize controlled decoded box
# ------------------------------------------------------------

canvas = np.zeros(
    (
        CONFIG["input_size"],
        CONFIG["input_size"],
        3
    ),
    dtype=np.float32
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    canvas
)

x1, y1, x2, y2 = \
    decoded_box.tolist()

rect = patches.Rectangle(
    (x1, y1),
    x2 - x1,
    y2 - y1,
    linewidth=3,
    edgecolor="red",
    facecolor="none"
)

ax.add_patch(
    rect
)

ax.text(
    x1,
    max(0, y1 - 8),
    f"person {float(decoded_score):.3f}",
    bbox=dict(
        facecolor="white",
        alpha=0.8
    )
)

ax.set_title(
    "Synthetic CenterNet Decoder Test"
)

ax.axis("off")

plt.show()


# ------------------------------------------------------------
# Real untrained model decoder test
# ------------------------------------------------------------

model.eval()

real_images, _ = next(
    iter(train_loader)
)

real_image = real_images[
    :1
].to(device)

with torch.no_grad():

    real_outputs = model(
        real_image
    )

    maximum_score = float(
        torch.sigmoid(
            real_outputs["heatmap"]
        ).max()
    )

    real_results = decode_centernet(
        real_outputs,
        K=100,
        score_threshold=CONFIG[
            "score_threshold"
        ]
    )


num_real_detections = len(
    real_results[0]["boxes"]
)


print("\nReal untrained model test")

print(
    "Maximum raw confidence :",
    maximum_score
)

print(
    "Detections @ 0.50      :",
    num_real_detections
)

print(
    "NOTE: zero detections before training is expected."
)


assert torch.isfinite(
    real_outputs["heatmap"]
).all()

assert torch.isfinite(
    real_outputs["wh"]
).all()

assert torch.isfinite(
    real_outputs["offset"]
).all()


del real_image
del real_outputs
del real_results
del synthetic_outputs

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSTEP 27 PASSED")
