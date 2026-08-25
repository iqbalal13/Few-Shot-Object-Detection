# ==========================================================
# STEP 8 : Cross Attention Sanity Test
# ==========================================================

with torch.no_grad():

    guided_query = cross_attention(

        query_tokens,

        support_tokens,

        support_prototype
    )


print(
    "Input Query Tokens :",
    query_tokens.shape
)

print(
    "Guided Query       :",
    guided_query.shape
)

assert guided_query.shape == query_tokens.shape

print("✓ Cross Attention correct")
