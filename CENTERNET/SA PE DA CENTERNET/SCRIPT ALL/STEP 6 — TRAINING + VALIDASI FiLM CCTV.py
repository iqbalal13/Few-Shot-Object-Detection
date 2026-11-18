from sape_da.locked_train import fit_target_ap50

TARGET_UPDATES = 100
CCTV_RUNS = {}
for seed in SEEDS:
    for shot in SHOTS:
        print(f'===== TRAIN CCTV FiLM: {shot}-shot, seed={seed} =====')
        model_target = SAPEDACenterNet()
        info = fit_target_ap50(
            model_target,
            CCTV_TRAIN_JSON,CCTV_TRAIN_IMAGES,
            CCTV_VAL_JSON,CCTV_VAL_IMAGES,
            SUPPORT_MANIFESTS[seed],shot,seed,TARGET_ROOT,SOURCE_FINAL,
            steps=TARGET_UPDATES,valid_every=20,lr=1e-3,reg_lambda=1e-3)
        CCTV_RUNS[(seed,shot)] = info
        del model_target
        torch.cuda.empty_cache()

assert len(CCTV_RUNS) == 9
print('DONE: CCTV FiLM training/validation, 9 independent shot × seed runs.')
print('SOURCE BASE COCO checkpoint was NOT trained or overwritten.')
