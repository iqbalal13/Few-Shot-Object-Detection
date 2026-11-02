from metadetr_pipeline.stage2 import prep_cctv
if RUN_PREPARE_CCTV or RUN_STAGE2_SMOKE or RUN_STAGE2_TRAIN or RUN_CCTV_VAL_EVAL or RUN_CCTV_TEST:
    assert G3_READY, 'CCTV locked: complete COCO-80 training and COCO-Val output first (Step 14B).'
    assert CCTV_ROOT.is_dir(),f'CCTV dataset missing: {CCTV_ROOT}'
    cctv_manifest=prep_cctv(CCTV_ROOT,WORKSPACE,seed=SEED)
    print('G4 manifest nested IDs:',cctv_manifest['images'])
    print('!!! Manually audit camera/time and near-duplicate splits before proceeding.')
else:
    print('G4 data prep disabled; no CCTV data loaded.')

