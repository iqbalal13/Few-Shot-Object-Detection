from metadetr_pipeline.stage1 import prepare as prepare_source, command as stage1_command
from metadetr_pipeline.data_protocol import save_json
if RUN_PREPARE_STAGE1 or RUN_STAGE1_SMOKE or RUN_STAGE1_TRAIN:
    assert coco_complete(COCO_ROOT), f'MS COCO dataset missing/incomplete: {COCO_ROOT}. Re-run Step 11B.'
    source_report = prepare_source(ROOT, COCO_ROOT, WORKSPACE, seed=SEED,
                                   shared_manifest=SHARED_SOURCE_MANIFEST,
                                   replace_symlink=True)
    counts = source_report['counts_per_category']
    assert len(source_report['image_ids']) == 8000, 'G1 LOCKED v2.1 expected exactly 8,000 unique images'
    assert len(set(source_report['image_ids'])) == 8000, 'G1 duplicate image IDs'
    assert len(counts) == 80 and all(v > 0 for v in counts.values()), 'G1 80-category coverage failed'
    if source_report['stats']['selection_mode']!='external_shared_manifest':
        assert all(int(source_report['images_per_category'][k])>=int(v) for k,v in source_report['stats']['initial_category_draws'].items()), 'G1 per-class quota not fulfilled'
    print('G1 PASS: images=',len(source_report['image_ids']),
          'category coverage=',len(counts), 'rare (<10 instances)=',source_report['rare_categories_lt_10_instances'])
    print('Class-wise sampled images (80 categories):', source_report['images_per_category'])
    print('Person-containing sampled images:',source_report['stats']['person_image_count'])
    print('Manifest:', WORKSPACE/'source_manifest.json')
else:
    print('G1 not run. Enable RUN_PREPARE_STAGE1 before source training.')
