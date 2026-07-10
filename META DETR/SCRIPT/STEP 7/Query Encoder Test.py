# ==========================================================
# STEP 7 : Query Encoder Test
# ==========================================================

dummy_query = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

with torch.no_grad():

    query_feature = query_encoder(dummy_query)

print("="*60)
print("Query Feature Shape")
print("="*60)

print(query_feature.shape)
