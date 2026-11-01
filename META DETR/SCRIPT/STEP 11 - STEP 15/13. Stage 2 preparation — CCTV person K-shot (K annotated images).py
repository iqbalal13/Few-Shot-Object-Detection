from metadetr_pipeline.stage2 import prep_cctv
if RUN_PREPARE_CCTV:
    manifest = prep_cctv(CCTV_ROOT, WORKSPACE, seed=SEED)
    print('Nested shot IDs:',manifest['images'])
    print('CCTV source annotations:', WORKSPACE/'cctv'/'annotations')
else:
    print('CCTV prep disabled. Set RUN_PREPARE_CCTV=True after mounting your CCTV dataset.')
