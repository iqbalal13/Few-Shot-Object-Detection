# ==========================================================
# STEP 4 : Backbone Sanity Test
# ==========================================================

dummy_image = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

backbone.eval()

with torch.no_grad():

    feature = backbone(dummy_image)


print("=" * 70)
print("Backbone Sanity Test")
print("=" * 70)

print("Input Shape   :", dummy_image.shape)
print("Feature Shape :", feature.shape)

assert feature.shape[1] == CONFIG["hidden_dim"]

print("✓ Backbone output dimension correct")
