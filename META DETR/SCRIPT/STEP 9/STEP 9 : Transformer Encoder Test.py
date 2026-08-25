# ==========================================================
# STEP 9 : Transformer Encoder Test
# ==========================================================

dummy_tokens = torch.randn(
    1,
    400,
    CONFIG["hidden_dim"]
).to(CONFIG["device"])

with torch.no_grad():

    encoder_output = transformer_encoder(
        dummy_tokens
    )

print("=" * 70)
print("Transformer Encoder Test")
print("=" * 70)
print("Input  :", dummy_tokens.shape)
print("Output :", encoder_output.shape)

assert encoder_output.shape == dummy_tokens.shape

print("✓ STEP 9 PASSED")
