matcher = HungarianMatcher()

criterion = SetCriterion(matcher)

outputs = {
    "pred_logits": torch.randn(2, 100, 2),
    "pred_boxes": torch.rand(2, 100, 4)
}

targets = [
    {
        "labels": torch.randint(0, 2, (5,)),
        "boxes": torch.rand(5, 4)
    },
    {
        "labels": torch.randint(0, 2, (3,)),
        "boxes": torch.rand(3, 4)
    }
]

losses = criterion(outputs, targets)

for k, v in losses.items():
    print(k, v.item())
