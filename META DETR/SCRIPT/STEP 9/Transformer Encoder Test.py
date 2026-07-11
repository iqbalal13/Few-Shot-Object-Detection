# ==========================================================
# STEP 9 : Transformer Encoder Test
# ==========================================================

with torch.no_grad():

    encoder_output = transformer_encoder(
        attended_feature
    )

print("="*60)
print("Transformer Encoder Output")
print("="*60)

print(encoder_output.shape)
