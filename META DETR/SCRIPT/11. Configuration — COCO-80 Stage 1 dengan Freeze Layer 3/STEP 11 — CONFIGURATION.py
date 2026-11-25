
# STEP 11 — CONFIGURATION (FULL TWO-STAGE END-TO-END / RUN ALL)

from pathlib import Path

ROOT = Path("/content/Meta-DETR")
COCO_ROOT = Path("/content/datasets/coco")
CCTV_ROOT = Path("/content/datasets/cctv")

WORKSPACE = Path("/content/metadetr_work_freeze3_full_epoch_bs8")
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
    "/content/drive/MyDrive/MetaDETR_Thesis/Stage1_COCO80_Freeze3_FullEpoch_BS8"
)

CHECKPOINT_PATH = (
    ROOT / "checkpoints/meta_detr_coco_base_checkpoint.pth"
)

# =====================================================
# GENERAL CONFIGURATION
# =====================================================

SEED = 42
SHARED_SOURCE_MANIFEST = None
BACKUP_TO_DRIVE = True  # Drive login at beginning; backup automatically after G3

# Run All is intentionally restricted to Stage 1.
RUN_OPTIONAL_STAGE1_POSTPROCESS = False  # skip extra person-only COCO AP
RUN_STAGE2_WORKFLOW = (RUN_MODE in ('full_pipeline', 'stage2_cctv'))

# =====================================================
# STAGE 1 — COCO-80 (SUDAH SELESAI)
# =====================================================

STAGE1_EPOCHS = 1  # one complete pass over all 8,000 COCO images
STAGE1_FREEZE_BACKBONE_AT_LAYER = 3  # LOCKED: stem+layer1+layer2+layer3 frozen
STAGE1_TIME_BUDGET_SECONDS = 9000  # > 1 epoch observed; training stops after 1,000 steps
BATCH_SIZE_STAGE1 = 8  # 8,000 images / 8 = 1,000 optimizer steps

RUN_DOWNLOAD_COCO = (RUN_MODE in ('full_pipeline', 'stage1'))
RUN_PREPARE_STAGE1 = (RUN_MODE in ('full_pipeline', 'stage1'))
RUN_STAGE1_SMOKE = (RUN_MODE in ('full_pipeline', 'stage1'))
RUN_STAGE1_TRAIN = (RUN_MODE in ('full_pipeline', 'stage1'))

SOURCE_CHECKPOINT_APPROVED = (RUN_MODE in ('full_pipeline', 'stage2_cctv'))  # strict COCO-Val & full-epoch gates still required

# =====================================================
# STAGE 2 — CCTV FEW-SHOT
# =====================================================

STAGE2_SHOTS = (1, 3, 5)
STAGE2_EPOCHS = 10
STAGE2_LR = 2e-5
STAGE2_PATIENCE = 4

RUN_PREPARE_CCTV = RUN_STAGE2_WORKFLOW

# PERUBAHAN: AKTIFKAN GPU SMOKE TEST
RUN_STAGE2_SMOKE = RUN_STAGE2_WORKFLOW

# Training dan evaluasi final masih OFF
RUN_STAGE2_TRAIN = RUN_STAGE2_WORKFLOW
RUN_CCTV_VAL_EVAL = RUN_STAGE2_WORKFLOW
RUN_CCTV_TEST = RUN_STAGE2_WORKFLOW  # final locked test, no test-driven tuning

# =====================================================
# STATUS CONFIGURATION
# =====================================================

print("=" * 55)
print("CONFIG READY")
print("=" * 55)

print("ROOT:", ROOT)
print("CCTV DRIVE:", CCTV_DRIVE_ROOT)
print("CCTV OUTPUT:", CCTV_ROOT)
print("STAGE1 CHECKPOINT:", STAGE1_CHECKPOINT)

print("STAGE1 FREEZE THROUGH LAYER:", STAGE1_FREEZE_BACKBONE_AT_LAYER)
print("COCO TRAINING:", RUN_STAGE1_TRAIN)
print("CCTV PREPARATION:", RUN_PREPARE_CCTV)
print("STAGE2 SMOKE TEST:", RUN_STAGE2_SMOKE)
print("STAGE2 TRAINING:", RUN_STAGE2_TRAIN)
print("CCTV TEST EVALUATION:", RUN_CCTV_TEST)

print("SHOTS:", STAGE2_SHOTS)
print("=" * 55)

EXPECTED_STAGE1_STEPS = (8000 + BATCH_SIZE_STAGE1 - 1) // BATCH_SIZE_STAGE1
assert EXPECTED_STAGE1_STEPS == 1000, "Batch-size-8 Stage 1 must be 1,000 steps."
print("EXPECTED FULL-EPOCH OPTIMIZER STEPS:", EXPECTED_STAGE1_STEPS)
