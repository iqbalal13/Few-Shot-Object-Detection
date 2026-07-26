# ==========================================================
# STEP 35 : Load COCO Base Checkpoint
# ==========================================================

checkpoint = torch.load(
    "/meta_detr_base_coco_final.pth",
    map_location=CONFIG["device"]
)

# ----------------------------------------------------------
# Load pretrained weights dari Base Training COCO
# ----------------------------------------------------------
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# ----------------------------------------------------------
# Reinitialize optimizer untuk Fine-Tuning CCTV
# (Transfer Learning)
# ----------------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

print("=" * 60)
print("COCO Pretrained Model Loaded")
print("Optimizer Reinitialized for CCTV Fine-Tuning")
print("=" * 60)
