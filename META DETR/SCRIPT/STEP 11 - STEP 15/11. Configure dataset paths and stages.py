
from pathlib import Path

# Direktori proyek
ROOT = Path("/content/Meta-DETR")
COCO_ROOT = Path("/content/datasets/coco")
CCTV_ROOT = Path("/content/datasets/cctv")
WORKSPACE = Path("/content/metadetr_work")

WORKSPACE.mkdir(parents=True, exist_ok=True)

# Checkpoint resmi lama hanya referensi path.
# TIDAK perlu didownload ulang.
CHECKPOINT_PATH = (
    ROOT / "checkpoints" /
    "meta_detr_coco_base_checkpoint.pth"
)

# Checkpoint COCO-80 HASIL TRAINING KAMU
STAGE1_CHECKPOINT = (
    WORKSPACE / "outputs" /
    "stage1_coco80" / "checkpoint.pth"
)

# Google Drive tempat model kamu disimpan
DRIVE_STAGE1 = Path(
    "/content/drive/MyDrive/"
    "MetaDETR_Thesis/Stage1_COCO80"
)

# Konfigurasi LOCKED
BACKUP_TO_DRIVE = False
SHARED_SOURCE_MANIFEST = None
SEED = 42

STAGE1_EPOCHS = 1
STAGE1_TIME_BUDGET_SECONDS = 3600
BATCH_SIZE_STAGE1 = 1

STAGE2_EPOCHS = 10
STAGE2_LR = 2e-5
STAGE2_PATIENCE = 4
STAGE2_SHOTS = (1, 3, 5)

# PENTING: MATIKAN seluruh training COCO
RUN_DOWNLOAD_COCO = False
RUN_PREPARE_STAGE1 = False
RUN_STAGE1_SMOKE = False
RUN_STAGE1_TRAIN = False

# CCTV belum dijalankan
RUN_PREPARE_CCTV = False
RUN_STAGE2_SMOKE = False
RUN_STAGE2_TRAIN = False
RUN_CCTV_VAL_EVAL = False
RUN_CCTV_TEST = False

# Persetujuan kualitas source checkpoint
SOURCE_CHECKPOINT_APPROVED = False

print("CONFIG READY")
print("Training COCO dinonaktifkan.")
print("Model akan diambil dari Google Drive lewat Step 14A.")
print("Drive:", DRIVE_STAGE1)
print("Checkpoint lokal:", STAGE1_CHECKPOINT)
