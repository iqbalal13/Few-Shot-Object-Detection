# ==========================================================
# STEP 37B : CCTV Train DataLoader
# ==========================================================

TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, "IMAGES")
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, "LABELS")

cctv_train_dataset = CCTVDataset(
    image_dir=TRAIN_IMAGE_DIR,
    label_dir=TRAIN_LABEL_DIR,
    transform=transform
)

cctv_train_loader = DataLoader(
    dataset=cctv_train_dataset,
    batch_size=COCO_CONFIG["batch_size"],
    shuffle=COCO_CONFIG["shuffle"],
    num_workers=COCO_CONFIG["num_workers"],
    pin_memory=COCO_CONFIG["pin_memory"],
    collate_fn=cctv_collate_fn   # <<< GANTI DI SINI
)

print("=" * 60)
print("Dataset :", len(cctv_train_dataset))
print("Batches :", len(cctv_train_loader))
print("=" * 60)
