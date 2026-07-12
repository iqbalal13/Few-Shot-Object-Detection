dummy_support = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

dummy_query = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"]
).to(CONFIG["device"])

with torch.no_grad():

    class_logits, boxes = model(
        dummy_support,
        dummy_query
    )

print(class_logits.shape)

print(boxes.shape)
