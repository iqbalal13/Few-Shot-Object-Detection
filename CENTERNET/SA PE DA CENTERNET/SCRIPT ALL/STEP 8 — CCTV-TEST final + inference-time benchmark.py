from sape_da.evaluate import evaluate_person, benchmark_real_cctv

rows=[]
for seed in SEEDS:
    for shot in SHOTS:
        run_dir = TARGET_ROOT/f'shot_{shot}'/f'seed_{seed}'
        metric_file = run_dir/'cctv_test_locked_metrics.json'
        if metric_file.is_file():
            result=json.loads(metric_file.read_text())
            print('Using saved CCTV-Test evaluation:',shot,seed)
        else:
            nn=SAPEDACenterNet()
            load_locked_checkpoint(checkpoints[(seed,shot)],nn)
            with torch.inference_mode():
                detection=evaluate_person(
                    nn,CCTV_TEST_JSON,CCTV_TEST_IMAGES,
                    result_path=run_dir/'cctv_test_predictions.json',
                    score_threshold=.001,ncacc_threshold=.5)
                latency=benchmark_real_cctv(
                    nn,CCTV_TEST_JSON,CCTV_TEST_IMAGES,warmup=10,limit=50)
            result={'detection':detection,'inference':latency,'shot':shot,'seed':seed}
            metric_file.write_text(json.dumps(result,indent=2))
            del nn
            torch.cuda.empty_cache()
        assert int(result['detection']['images']) == int(CCTV_SUMMARY['test']['images'])
        rows.append({'seed':seed,'shot':shot,
            'mAP50_percent':100*result['detection']['AP50'],
            'NCAcc50_percent':100*result['detection']['NCAcc50'],
            'inference_ms_per_image':result['inference']['ms_mean'],
            'TP':result['detection']['TP'],
            'FP':result['detection']['FP'],
            'FN':result['detection']['FN'],
            'test_images':result['detection']['images']})
print('CCTV-Test complete; no COCO Stage1 training executed.')
