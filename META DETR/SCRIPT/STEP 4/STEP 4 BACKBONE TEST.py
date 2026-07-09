# ==========================================================
# STEP 4 : Backbone Test
# ==========================================================

dummy = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
)

dummy = dummy.to(CONFIG["device"])
backbone = backbone.to(CONFIG["device"])

with torch.no_grad():
    feature = backbone(dummy)

print("="*60)
print("Feature Shape")
print("="*60)

print(feature.shape)
