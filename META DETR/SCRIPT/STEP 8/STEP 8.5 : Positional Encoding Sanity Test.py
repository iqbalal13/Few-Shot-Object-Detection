# ==========================================================
# STEP 8.5 : Positional Encoding Sanity Test
# ==========================================================

with torch.no_grad():

    query_position = \
        position_encoding(feature)


print(
    "Position Encoding :",
    query_position.shape
)

assert (
    query_position.shape
    ==
    query_tokens.shape
)

print("✓ Positional Encoding correct")
