# ==========================================================
# STEP 5 : Prototype Test
# ==========================================================

with torch.no_grad():

    prototype = prototype_extractor(feature)

print("="*60)
print("Prototype Shape")
print("="*60)

print(prototype.shape)
