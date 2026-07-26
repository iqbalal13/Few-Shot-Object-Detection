# ==========================================================
# STEP 37C : Few-Shot Fine-Tuning on CCTV
# ==========================================================

from tqdm import tqdm
import os
import torch

# ----------------------------------------------------------
# Fine-tuning Configuration
# ----------------------------------------------------------
FINE_TUNE_EPOCHS = 30
MAX_STEPS_PER_EPOCH = len(cctv_train_loader)

model = model.to(CONFIG["device"])

print("=" * 60)
print("Start CCTV Fine-Tuning")
print("=" * 60)
print(f"Epochs          : {FINE_TUNE_EPOCHS}")
print(f"Dataset Size    : {len(cctv_train_dataset)}")
print(f"Steps / Epoch   : {MAX_STEPS_PER_EPOCH}")
print("=" * 60)

for epoch in range(FINE_TUNE_EPOCHS):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        cctv_train_loader,
        desc=f"FineTune Epoch [{epoch+1}/{FINE_TUNE_EPOCHS}]"
    )

    actual_steps = 0

    for images, targets in progress_bar:

        # -----------------------------
        # Images
        # -----------------------------
        images = torch.stack(images).to(CONFIG["device"])

        # -----------------------------
        # Targets
        # -----------------------------
        new_targets = []

        for target in targets:

            new_targets.append({

                "boxes": target["boxes"].to(CONFIG["device"]),
                "labels": target["labels"].to(CONFIG["device"])

            })

        # -----------------------------
        # Forward
        # -----------------------------
        class_logits, pred_boxes = model(images)

        outputs = {

            "pred_logits": class_logits,
            "pred_boxes": pred_boxes

        }

        # -----------------------------
        # Loss
        # -----------------------------
        loss_dict = criterion(outputs, new_targets)

        # =====================================================
        # DEBUG : tampilkan komponen loss pada iterasi pertama
        # =====================================================
        if epoch == 0 and actual_steps == 0:

            print("\n" + "=" * 60)
            print("LOSS COMPONENTS (FIRST ITERATION)")
            print("=" * 60)

            for k, v in loss_dict.items():
                print(f"{k:15s}: {v.item():.6f}")

            print("=" * 60 + "\n")

        total_loss = (
            loss_dict["loss_ce"]
            + loss_dict["loss_bbox"]
            + loss_dict["loss_giou"]
        )

        # -----------------------------
        # Backprop
        # -----------------------------
        optimizer.zero_grad()

        total_loss.backward()

        # ------------------------------------------------------
        # Gradient Clipping
        # ------------------------------------------------------
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += total_loss.item()

        actual_steps += 1

        progress_bar.set_postfix({

            "Loss": f"{total_loss.item():.4f}"

        })

    # ------------------------------------------------------
    # Optimizer telah diinisialisasi ulang pada STEP 35.
    # Scheduler base training COCO tidak digunakan lagi.
    # ------------------------------------------------------
    # scheduler.step()

    epoch_loss = running_loss / actual_steps

    print(
        f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS}"
        f" | Loss : {epoch_loss:.4f}"
    )

# ----------------------------------------------------------
# Save Fine-Tuned Model
# ----------------------------------------------------------
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

fine_tune_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "meta_detr_cctv_finetuned.pth"
)

torch.save({

    "epoch": FINE_TUNE_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": CONFIG

}, fine_tune_checkpoint)

print("=" * 60)
print("Fine-Tuning Finished")
print(f"Checkpoint : {fine_tune_checkpoint}")
print("=" * 60)
