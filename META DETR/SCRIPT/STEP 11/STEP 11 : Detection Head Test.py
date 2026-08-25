# ==========================================================
# STEP 11 : Detection Head Test
# ==========================================================

dummy_decoder = torch.randn(
    1,
    CONFIG["num_queries"],
    CONFIG["hidden_dim"]
).to(CONFIG["device"])

with torch.no_grad():

    class_logits, pred_boxes = detection_head(
        dummy_decoder
    )

print("=" * 70)
print("Detection Head Test")
print("=" * 70)

print("Classification :", class_logits.shape)
print("Bounding Boxes :", pred_boxes.shape)

assert class_logits.shape == (
    1,
    CONFIG["num_queries"],
    CONFIG["num_output_classes"]
)

assert pred_boxes.shape == (
    1,
    CONFIG["num_queries"],
    4
)

assert (
    pred_boxes.min() >= 0
    and
    pred_boxes.max() <= 1
)

print("✓ STEP 11 PASSED")
