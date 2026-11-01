from metadetr_pipeline.stage1 import prepare as prepare_source, command as stage1_command
if RUN_PREPARE_STAGE1 or RUN_STAGE1_TRAIN:
    assert COCO_ROOT.is_dir(), f'COCO data missing: {COCO_ROOT}'
    source_report = prepare_source(ROOT, COCO_ROOT, WORKSPACE, seed=SEED,
                                   shared_manifest=SHARED_SOURCE_MANIFEST,
                                   replace_symlink=True)  # only replaces a symlink, NEVER a real directory
    print('Stage 1 selected images:', len(source_report['image_ids']))
    print('Rare classes:', source_report['rare_categories_lt_10_instances'])
    cmd = stage1_command(ROOT, CHECKPOINT_PATH, WORKSPACE/'outputs'/'stage1_coco80',
                         STAGE1_EPOCHS, BATCH_SIZE_STAGE1, SEED, 2)
    print('Stage1 command:', ' '.join(cmd))
    if RUN_STAGE1_TRAIN:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
else:
    print('Stage1 disabled. Set RUN_PREPARE_STAGE1=True first, inspect manifest, then RUN_STAGE1_TRAIN=True.')
