# ==========================================================
# STEP 5 : Prototype Sanity Test
# ==========================================================

with torch.no_grad():

    support_tokens, prototype = \
        prototype_extractor(feature)


print("Support Tokens :", support_tokens.shape)
print("Prototype      :", prototype.shape)

assert support_tokens.shape[-1] == CONFIG["hidden_dim"]
assert prototype.shape[-1] == CONFIG["hidden_dim"]

print("✓ Prototype extraction correct")
