from pathlib import Path
COCO_ROOT = Path('/content/datasets/coco')
CCTV_ROOT = Path('/content/datasets/cctv')
WORKSPACE = Path('/content/metadetr_work')
SHARED_SOURCE_MANIFEST = None  # e.g. '/content/drive/MyDrive/experiment/source_image_ids.json'
SEED = 42
STAGE1_EPOCHS = 25    # PILOT value, NOT LOCKED
STAGE2_EPOCHS = 10    # PILOT value, NOT LOCKED
BATCH_SIZE_STAGE1 = 1
RUN_PREPARE_STAGE1 = False
RUN_STAGE1_TRAIN = False
RUN_PREPARE_CCTV = False
RUN_STAGE2_TRAIN = False
STAGE2_SHOT = 1       # choose 1, 3, or 5, each starts independently from Stage1
print('Data ready:',COCO_ROOT.exists(),CCTV_ROOT.exists())
print('Official COCO 60-base initializer:', CHECKPOINT_PATH)
print('Stage1 target:',WORKSPACE/'outputs'/'stage1_coco80'/'checkpoint.pth')
