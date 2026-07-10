# ==========================================================
# STEP 8 : Cross Attention Test
# ==========================================================

with torch.no_grad():

    attended_feature = cross_attention(
        support_prototype,
        query_feature
    )

print("="*60)
print("Cross Attention Output")
print("="*60)

print(attended_feature.shape)
