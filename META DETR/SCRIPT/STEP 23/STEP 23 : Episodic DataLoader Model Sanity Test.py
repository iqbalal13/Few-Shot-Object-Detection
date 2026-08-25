# ==========================================================
# STEP 23 : Episodic DataLoader + Model Sanity Test
# ==========================================================

import torch


batch = next(
    iter(train_loader)
)


episode_classes = (
    batch["episode_classes"]
)

support_images = (
    batch["support_images"]
)

query_images = (
    batch["query_images"]
)

support_targets = (
    batch["support_targets"]
)

query_targets = (
    batch["query_targets"]
)


print("=" * 70)
print("STEP 23 : DATALOADER SANITY CHECK")
print("=" * 70)

print(
    "Episode Class    :",
    episode_classes.tolist()
)

print(
    "Support Batch    :",
    support_images.shape
)

print(
    "Query Batch      :",
    query_images.shape
)

print(
    "Support Targets  :",
    len(support_targets)
)

print(
    "Query Targets    :",
    len(query_targets)
)


# ==========================================================
# Validate episode consistency
# ==========================================================

for i in range(
    len(query_targets)
):

    episode_class = (
        episode_classes[i].item()
    )

    assert (
        support_targets[i]["labels"]
        ==
        episode_class
    ).all()

    assert (
        query_targets[i]["labels"]
        ==
        episode_class
    ).all()


# ==========================================================
# Model Forward Test
# ==========================================================

model.eval()

with torch.no_grad():

    outputs = model(

        support_images.to(
            CONFIG["device"]
        ),

        query_images.to(
            CONFIG["device"]
        )
    )


assert (
    outputs["pred_logits"].shape[-1]
    ==
    CONFIG["num_output_classes"]
)

assert (
    outputs["pred_boxes"].shape[-1]
    ==
    4
)


print("-" * 70)

print(
    "Pred Logits      :",
    outputs["pred_logits"].shape
)

print(
    "Pred Boxes       :",
    outputs["pred_boxes"].shape
)

print("=" * 70)
print("✓ STEP 23 PASSED")
print("=" * 70)
