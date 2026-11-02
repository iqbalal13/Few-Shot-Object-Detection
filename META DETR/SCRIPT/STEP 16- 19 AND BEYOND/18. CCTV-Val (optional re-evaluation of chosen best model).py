from metadetr_pipeline.evaluation import evaluate_cctv
if RUN_CCTV_VAL_EVAL:
    assert G3_READY, 'Finish COCO-80 + COCO-Val first.'
    for k in STAGE2_SHOTS:
        best=WORKSPACE/'outputs'/f'stage2_cctv_{k}shot'/'best.pth'
        assert best.is_file(),f'Missing CCTV-Val best checkpoint for {k}-shot'
        metrics=evaluate_cctv(ROOT,WORKSPACE,CCTV_ROOT,best,shot=k,split='val',seed=SEED)
        print('CCTV-VAL',k,'shot:',{name:metrics[name] for name in ('AP50','AP75','AP50_95','NCAcc_at_IoU50','latency_ms_mean')})
else:
    print('Validation re-evaluation disabled (training already evaluates Val per epoch).')
