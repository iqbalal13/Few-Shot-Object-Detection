# ==========================================================
# STEP 33 : COCO Validation (Episodic)
# ==========================================================

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

print("=" * 60)
print("Preparing COCO Validation Dataset")
print("=" * 60)

# ----------------------------------------------------------
# Validation Dataset
# ----------------------------------------------------------
coco_val = COCO(
    "/content/datasets/coco/annotations/instances_val2017.json"
)

val_dataset = COCOEpisodeDataset(
    coco=coco_val,
    image_dir="/content/datasets/coco/val2017",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=COCO_CONFIG["batch_size"],
    shuffle=False,
    num_workers=COCO_CONFIG["num_workers"],
    pin_memory=COCO_CONFIG["pin_memory"],
    collate_fn=collate_fn
)

print(f"Validation Images : {len(val_dataset)}")
print(f"Validation Batches: {len(val_loader)}")
print("=" * 60)

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------
model.eval()

running_loss = 0.0
running_ce = 0.0
running_bbox = 0.0
running_giou = 0.0

with torch.no_grad():

    progress_bar = tqdm(
        val_loader,
        desc="Validation"
    )

    for batch in progress_bar:

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

        running_loss += total_loss.item()
        running_ce += loss_dict["loss_ce"].item()
        running_bbox += loss_dict["loss_bbox"].item()
        running_giou += loss_dict["loss_giou"].item()

        progress_bar.set_postfix({

            "Loss": f"{total_loss.item():.4f}"

        })

# ----------------------------------------------------------
# Result
# ----------------------------------------------------------
num_batches = len(val_loader)

print("\n" + "=" * 60)
print("COCO VALIDATION RESULT")
print("=" * 60)
print(f"Average Total Loss : {running_loss / num_batches:.4f}")
print(f"Average CE Loss    : {running_ce / num_batches:.4f}")
print(f"Average BBox Loss  : {running_bbox / num_batches:.4f}")
print(f"Average GIoU Loss  : {running_giou / num_batches:.4f}")
print("=" * 60)
