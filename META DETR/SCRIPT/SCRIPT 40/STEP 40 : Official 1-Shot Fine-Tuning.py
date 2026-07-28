# ==========================================================
# STEP 40 : Official 1-Shot Fine-Tuning
# ==========================================================

from tqdm import tqdm
from itertools import cycle
import torch

print("=" * 70)
print("STEP 40 : OFFICIAL 1-SHOT FINE-TUNING")
print("=" * 70)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

FS_EPOCHS = 30

model = model.to(CONFIG["device"])

print(f"Epochs          : {FS_EPOCHS}")
print(f"Support Images  : {len(support_dataset)}")
print(f"Query Images    : {len(query_dataset)}")
print("=" * 70)

# ==========================================================
# Training
# ==========================================================

for epoch in range(FS_EPOCHS):

    model.train()

    running_loss = 0.0
    running_ce = 0.0
    running_bbox = 0.0
    running_giou = 0.0

    support_iter = cycle(support_loader)

    progress_bar = tqdm(
        query_loader,
        desc=f"1-Shot Epoch [{epoch+1}/{FS_EPOCHS}]"
    )

    for query_images, query_targets in progress_bar:

        # --------------------------------------------------
        # Support Batch
        # --------------------------------------------------

        support_images, support_targets = next(support_iter)

        support_images = torch.stack(
            support_images
        ).to(CONFIG["device"])

        # --------------------------------------------------
        # Query Batch
        # --------------------------------------------------

        query_images = torch.stack(
            query_images
        ).to(CONFIG["device"])

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------

        new_targets = []

        for target in query_targets:

            new_targets.append({

                "boxes": target["boxes"].to(CONFIG["device"]),
                "labels": target["labels"].to(CONFIG["device"])

            })

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        class_logits, pred_boxes = model(

            support_images,
            query_images

        )

        outputs = {

            "pred_logits": class_logits,
            "pred_boxes": pred_boxes

        }

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss_dict = criterion(

            outputs,
            new_targets

        )

        loss = sum(loss_dict.values())

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        running_loss += loss.item()

        running_ce += loss_dict["loss_ce"].item()

        running_bbox += loss_dict["loss_bbox"].item()

        running_giou += loss_dict["loss_giou"].item()

        progress_bar.set_postfix({

            "Loss": f"{loss.item():.4f}",

            "CE": f"{loss_dict['loss_ce'].item():.3f}",

            "BBox": f"{loss_dict['loss_bbox'].item():.3f}",

            "GIoU": f"{loss_dict['loss_giou'].item():.3f}"

        })

    # ------------------------------------------------------
    # Epoch Summary
    # ------------------------------------------------------

    num_batches = len(query_loader)

    print("\n" + "-" * 60)

    print(f"Epoch {epoch+1}/{FS_EPOCHS}")

    print(f"Loss      : {running_loss/num_batches:.4f}")

    print(f"CE        : {running_ce/num_batches:.4f}")

    print(f"BBox      : {running_bbox/num_batches:.4f}")

    print(f"GIoU      : {running_giou/num_batches:.4f}")

    print("-" * 60)

print("=" * 70)
print("STEP 40 COMPLETED")
print("=" * 70)
