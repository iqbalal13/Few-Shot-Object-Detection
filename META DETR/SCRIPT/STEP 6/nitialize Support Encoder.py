# ==========================================================
# STEP 6 : Initialize Support Encoder
# ==========================================================

support_encoder = SupportEncoder(
    backbone,
    prototype_extractor
).to(CONFIG["device"])

print(support_encoder)
