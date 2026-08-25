# ==========================================================
# STEP 6 : Support Encoder Sanity Test
# ==========================================================

with torch.no_grad():

    support_tokens, support_prototype = \
        support_encoder(feature)


print(
    "Support Tokens    :",
    support_tokens.shape
)

print(
    "Support Prototype :",
    support_prototype.shape
)

assert support_tokens.shape[-1] == CONFIG["hidden_dim"]
assert support_prototype.shape[-1] == CONFIG["hidden_dim"]

print("✓ Support Encoder correct")
