
# STEP 11 — CONFIGURATION (CCTV PATH FIX)

from pathlib import Path

ROOT = Path("/content/Meta-DETR")
COCO_ROOT = Path("/content/datasets/coco")
CCTV_ROOT = Path("/content/datasets/cctv")
WORKSPACE = Path("/content/metadetr_work")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Dataset CCTV asli di Google Drive
CCTV_DRIVE_ROOT = Path(
    "/content/drive/MyDrive/DATASET/GAMBAR CCTV"
)

# Checkpoint Stage 1 hasil training COCO-80
STAGE1_CHECKPOINT = (
    WORKSPACE / "outputs/stage1_coco80/checkpoint.pth"
)

DRIVE_STAGE1 = Path(
    "/content/drive/MyDrive/MetaDETR_Thesis/Stage1_COCO80"
)

CHECKPOINT_PATH = (
    ROOT / "checkpoints/meta_detr_coco_base_checkpoint.pth"
)

SEED = 42
SHARED_SOURCE_MANIFEST = None
BACKUP_TO_DRIVE = False

STAGE1_EPOCHS = 1
STAGE1_TIME_BUDGET_SECONDS = 3600
BATCH_SIZE_STAGE1 = 1

RUN_DOWNLOAD_COCO = False
RUN_PREPARE_STAGE1 = False
RUN_STAGE1_SMOKE = False
RUN_STAGE1_TRAIN = False

SOURCE_CHECKPOINT_APPROVED = True

STAGE2_SHOTS = (1, 3, 5)
STAGE2_EPOCHS = 10
STAGE2_LR = 2e-5
STAGE2_PATIENCE = 4

RUN_PREPARE_CCTV = True
RUN_STAGE2_SMOKE = False
RUN_STAGE2_TRAIN = False
RUN_CCTV_VAL_EVAL = False
RUN_CCTV_TEST = False

print("CONFIG READY")
print("CCTV DRIVE:", CCTV_DRIVE_ROOT)
print("CCTV OUTPUT:", CCTV_ROOT)
print("COCO TRAINING:", RUN_STAGE1_TRAIN)
print("CCTV TRAINING:", RUN_STAGE2_TRAIN)
