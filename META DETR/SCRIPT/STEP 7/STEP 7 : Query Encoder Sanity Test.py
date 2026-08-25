# ==========================================================
# STEP 7 : Query Encoder Sanity Test
# ==========================================================

with torch.no_grad():

    query_tokens, spatial_shape = \
        query_encoder(feature)


print(
    "Query Tokens  :",
    query_tokens.shape
)

print(
    "Spatial Shape :",
    spatial_shape
)

assert query_tokens.shape[-1] == CONFIG["hidden_dim"]

print("✓ Query Encoder correct")
