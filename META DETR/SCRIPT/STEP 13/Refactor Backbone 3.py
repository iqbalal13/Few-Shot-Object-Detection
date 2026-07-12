dummy = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

with torch.no_grad():

    feature = backbone_v2(dummy)

print(feature.shape)
