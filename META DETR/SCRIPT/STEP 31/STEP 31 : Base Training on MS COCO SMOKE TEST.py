# ==========================================================
# STEP 31 : Base Training on MS COCO (Smoke Test)
# ==========================================================

from tqdm import tqdm
import torch

# ----------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------
MAX_STEPS_PER_EPOCH = 2

model = model.to(CONFIG["device"])

print("=" * 60)
print("Start Episodic Base Training (Smoke Test)")
print("=" * 60)
print(f"Epochs           : {NUM_EPOCHS}")
print(f"Batch Size       : {COCO_CONFIG['batch_size']}")
print(f"Max Steps/Epoch  : {MAX_STEPS_PER_EPOCH}")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
    )

    actual_steps = 0

    for step, batch in enumerate(progress_bar):

        # --------------------------------------------------
        # Stop after MAX_STEPS_PER_EPOCH
        # --------------------------------------------------
        if actual_steps >= MAX_STEPS_PER_EPOCH:
            break

        # --------------------------------------------------
        # Support Images
        # --------------------------------------------------
        support_images = torch.stack(
            batch["support_images"]
        ).to(CONFIG["device"])

        # --------------------------------------------------
        # Query Images
        # --------------------------------------------------
        query_images = torch.stack(
            batch["query_images"]
        ).to(CONFIG["device"])

        # --------------------------------------------------
        # Query Targets
        # --------------------------------------------------
        query_targets = []

        for target in batch["query_targets"]:

            query_targets.append({

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
            query_targets
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

        actual_steps += 1

        progress_bar.set_postfix({

            "Loss": f"{total_loss.item():.4f}",
            "Step": f"{actual_steps}/{MAX_STEPS_PER_EPOCH}"

        })

    scheduler.step()

    epoch_loss = running_loss / actual_steps

    print(

        f"Epoch {epoch+1}/{NUM_EPOCHS}"
        f" | Steps : {actual_steps}"
        f" | Average Loss : {epoch_loss:.4f}"

    )

print("=" * 60)
print("Smoke Test Finished")
print("=" * 60)
