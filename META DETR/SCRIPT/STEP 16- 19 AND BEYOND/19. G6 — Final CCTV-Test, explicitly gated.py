from metadetr_pipeline.evaluation import evaluate_cctv_test
if RUN_CCTV_TEST:
    assert G3_READY, 'Finish COCO-80 + COCO-Val first.'
    test_results={}
    for k in STAGE2_SHOTS:
        best=WORKSPACE/'outputs'/f'stage2_cctv_{k}shot'/'best.pth'
        assert best.is_file(),f'Missing independently Val-selected best.pth for {k}-shot'
        test_results[k]=evaluate_cctv_test(ROOT,WORKSPACE,CCTV_ROOT,best,k,seed=SEED)
    print('G6 FINAL CCTV-TEST results:')
    for k,m in test_results.items():
        print(f'{k}-shot: AP50={m["AP50"]:.4f}, AP75={m["AP75"]:.4f}, AP50:95={m["AP50_95"]:.4f}, NCAcc={m["NCAcc_at_IoU50"]:.4f}, cached latency={m["latency_ms_mean"]:.2f}ms')
else:
    print('G6 TEST DISABLED. Keep it off until models are selected using CCTV-Val.')
