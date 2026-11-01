from metadetr_pipeline.stage2 import run_finetune
STAGE1_CHECKPOINT = WORKSPACE/'outputs'/'stage1_coco80'/'checkpoint.pth'
if RUN_STAGE2_TRAIN:
    assert STAGE1_CHECKPOINT.is_file(), 'Stage 1 must finish before Stage 2'
    assert (WORKSPACE/'cctv'/'fewshot'/f'cctv_train_{STAGE2_SHOT}shot.json').is_file()
    new_ckpt = run_finetune(ROOT, WORKSPACE, CCTV_ROOT, STAGE1_CHECKPOINT,
                            shot=STAGE2_SHOT, epochs=STAGE2_EPOCHS, seed=SEED)
    print('Stage2 checkpoint:',new_ckpt)
else:
    print('Stage2 disabled. Stage 1 checkpoint available:',STAGE1_CHECKPOINT.exists())
