# ==========================================================
# STEP 10 : Initialize Transformer Decoder
# ==========================================================

transformer_decoder = TransformerDecoder(
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_decoder_layers"],
    num_queries=CONFIG["num_queries"]
).to(CONFIG["device"])

print(transformer_decoder)
