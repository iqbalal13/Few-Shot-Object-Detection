from pathlib import Path
COCO_ROOT = Path('/content/datasets/coco')
CCTV_ROOT = Path('/content/datasets/cctv')
WORKSPACE = Path('/content/metadetr_work')
SHARED_SOURCE_MANIFEST = None  # Optional exact shared COCO image-ID manifest
SEED = 42
STAGE1_EPOCHS = 25        # pilot only; NOT locked in PDF
STAGE2_EPOCHS = 10        # pilot only; NOT locked in PDF
STAGE2_LR = 2e-5         # pilot only; NOT locked in PDF
STAGE2_PATIENCE = 4      # pilot only; NOT locked in PDF
BATCH_SIZE_STAGE1 = 1
STAGE2_SHOTS = (1,3,5)   # independent experiments, no chaining
RUN_PREPARE_STAGE1 = False
RUN_STAGE1_SMOKE = False
RUN_STAGE1_TRAIN = False
RUN_PREPARE_CCTV = False
RUN_STAGE2_SMOKE = False
RUN_STAGE2_TRAIN = False
RUN_CCTV_VAL_EVAL = False
RUN_CCTV_TEST = False    # FINAL ONLY, after best checkpoints are chosen on Val
print('COCO available:', COCO_ROOT.is_dir(), 'CCTV available:', CCTV_ROOT.is_dir())
print('Official initializer:', CHECKPOINT_PATH)
print('Stage1 output:',WORKSPACE/'outputs/stage1_coco80/checkpoint.pth')
