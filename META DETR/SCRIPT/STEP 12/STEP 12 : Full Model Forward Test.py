# ==========================================================
# STEP 12 : Full Model Forward Test
# ==========================================================

dummy_support = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

dummy_query = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

model.eval()

with torch.no_grad():

    outputs = model(
        dummy_support,
        dummy_query
    )

print("=" * 70)
print("Simplified Meta-DETR Forward Test")
print("=" * 70)

print(
    "Pred Logits :",
    outputs["pred_logits"].shape
)

print(
    "Pred Boxes  :",
    outputs["pred_boxes"].shape
)

assert outputs["pred_logits"].shape == (
    1,
    CONFIG["num_queries"],
    CONFIG["num_output_classes"]
)

assert outputs["pred_boxes"].shape == (
    1,
    CONFIG["num_queries"],
    4
)

print("✓ STEP 12 PASSED")
