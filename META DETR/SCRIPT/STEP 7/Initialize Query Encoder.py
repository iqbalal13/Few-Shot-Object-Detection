# ==========================================================
# STEP 7 : Initialize Query Encoder
# ==========================================================

query_encoder = QueryEncoder(backbone).to(CONFIG["device"])

print(query_encoder)
