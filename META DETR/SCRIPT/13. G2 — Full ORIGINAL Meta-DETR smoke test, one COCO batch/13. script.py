from metadetr_pipeline.stage1_smoke import smoke_source
from metadetr_pipeline.data_protocol import save_json
G2_REPORT = WORKSPACE/'gates'/'g2_source_smoke.json'
if RUN_STAGE1_SMOKE:
    expected_g1 = ROOT/'data/coco/annotations/instances_train2017.json'
    assert expected_g1.is_file(), f'Expected COCO-80 G1 annotation symlink: {expected_g1}'
    print('G2 preflight: COCO-80 annotation path verified:', expected_g1)
    report=smoke_source(ROOT,CHECKPOINT_PATH,seed=SEED,batch_size=BATCH_SIZE_STAGE1,
                        freeze_backbone_at_layer=STAGE1_FREEZE_BACKBONE_AT_LAYER)
    save_json(G2_REPORT,{'passed':True,'seed':SEED,'loss':report['loss'],
        'source_manifest_sha256':__import__('hashlib').sha256((WORKSPACE/'source_manifest.json').read_bytes()).hexdigest(),
        'freeze_backbone_at_layer':report['freeze_backbone_at_layer'],
        'backbone_freeze_report':report['backbone_freeze_report'],
        'status':'full_original_meta_detr_optimizer_step'})
    print('G2 PASS saved:',G2_REPORT)
else:
    print('G2 not requested. Existing report:',G2_REPORT.exists())
