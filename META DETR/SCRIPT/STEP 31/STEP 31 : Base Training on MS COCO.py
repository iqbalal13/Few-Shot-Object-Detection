# ==========================================================
# STEP 31 : Base Training on MS COCO
# ==========================================================

from tqdm import tqdm
import torch

model = model.to(CONFIG["device"])

print("=" * 60)
print("Start Base Training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
    )

    for images, targets in progress_bar:

        # --------------------------------------------------
        # Images
        # --------------------------------------------------
        images = torch.stack(images).to(CONFIG["device"])

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------
        new_targets = []

        for target in targets:

            new_targets.append({

                "boxes": target["boxes"].to(CONFIG["device"]),

                "labels": target["labels"].to(CONFIG["device"])

            })

        # --------------------------------------------------
        # Forward
        # Base Training:
        # support_image == query_image
        # --------------------------------------------------
        class_logits, pred_boxes = model(images)

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

        total_loss = (

            loss_dict["loss_ce"]

            + loss_dict["loss_bbox"]

            + loss_dict["loss_giou"]

        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------
        optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1
        )

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------
        running_loss += total_loss.item()

        progress_bar.set_postfix({

            "Loss": f"{total_loss.item():.4f}"

        })

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)

    print(

        f"Epoch {epoch+1}/{NUM_EPOCHS}"

        f" | Average Loss : {epoch_loss:.4f}"

    )

print("=" * 60)
print("Base Training Finished")
print("=" * 60)
