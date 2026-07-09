# ==========================================================
# STEP 4 : Remove Classification Layer
# ==========================================================

backbone = nn.Sequential(
    *list(resnet101.children())[:-2]
)

print(backbone)
