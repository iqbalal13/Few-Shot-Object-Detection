# ==========================================================
# STEP 6 : Support Encoder Test
# ==========================================================

dummy_support = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

with torch.no_grad():

    support_prototype = support_encoder(dummy_support)

print("="*60)
print("Support Prototype Shape")
print("="*60)

print(support_prototype.shape)
