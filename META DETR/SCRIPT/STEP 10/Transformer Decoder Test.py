# ==========================================================
# STEP 10 : Transformer Decoder Test
# ==========================================================

with torch.no_grad():

    decoder_output = transformer_decoder(
        encoder_output
    )

print("="*60)
print("Transformer Decoder Output")
print("="*60)

print(decoder_output.shape)
