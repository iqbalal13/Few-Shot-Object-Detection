# ==========================================================
# STEP 9 : Initialize Transformer Encoder
# ==========================================================

transformer_encoder = TransformerEncoder(
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_encoder_layers"]
).to(CONFIG["device"])

print(transformer_encoder)
