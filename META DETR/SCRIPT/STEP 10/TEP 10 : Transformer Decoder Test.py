# ==========================================================
# STEP 10 : Transformer Decoder Test
# ==========================================================

dummy_memory = torch.randn(
    1,
    400,
    CONFIG["hidden_dim"]
).to(CONFIG["device"])

with torch.no_grad():

    decoder_output = transformer_decoder(
        dummy_memory
    )

print("=" * 70)
print("Transformer Decoder Test")
print("=" * 70)

print("Memory :", dummy_memory.shape)
print("Output :", decoder_output.shape)

assert decoder_output.shape == (
    1,
    CONFIG["num_queries"],
    CONFIG["hidden_dim"]
)

print("✓ STEP 10 PASSED")
