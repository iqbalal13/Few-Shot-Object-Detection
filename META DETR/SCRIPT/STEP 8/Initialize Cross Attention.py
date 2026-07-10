# ==========================================================
# STEP 8 : Initialize Cross Attention
# ==========================================================

cross_attention = PrototypeCrossAttention().to(CONFIG["device"])

print(cross_attention)
