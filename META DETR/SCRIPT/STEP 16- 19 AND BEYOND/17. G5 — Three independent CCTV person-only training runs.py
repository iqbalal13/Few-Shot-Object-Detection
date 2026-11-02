if RUN_STAGE2_TRAIN:
    assert G3_READY, 'Finish COCO-80 training and COCO-Val evaluation first.'
    assert STAGE1_CHECKPOINT.is_file(),'Stage1 COCO-80 checkpoint is missing'
    selected={}
    for k in STAGE2_SHOTS:
        assert (WORKSPACE/'gates'/f'g4_stage2_{k}shot_smoke.json').is_file(),f'Run 1-step Stage2 smoke for shot={k} first'
        best=run_finetune(ROOT,WORKSPACE,CCTV_ROOT,STAGE1_CHECKPOINT,shot=k,
                          epochs=STAGE2_EPOCHS,seed=SEED,lr=STAGE2_LR,
                          smoke_only=False,patience=STAGE2_PATIENCE)
        selected[k]=str(best)
    print('G5 best checkpoints selected on CCTV-Val:',selected)
else:
    print('G5 training disabled. Only enable after all three shot smoke tests pass.')
