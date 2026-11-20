from sape_da.locked_train import load_locked_checkpoint
checkpoints = {}
for seed in SEEDS:
    for shot in SHOTS:
        folder = TARGET_ROOT/f'shot_{shot}'/f'seed_{seed}'
        best = folder/'target_best.pth'
        completed = folder/'complete.json'
        assert best.is_file() and completed.is_file(), f'Checkpoint CCTV belum lengkap: {folder}'
        ckmodel = SAPEDACenterNet()
        ck = load_locked_checkpoint(best,ckmodel)
        assert ck['phase'] == 'target'
        assert int(ck['metadata']['shot']) == shot and int(ck['metadata']['seed']) == seed
        assert Path(ck['metadata']['source_checkpoint']) == SOURCE_FINAL
        checkpoints[(seed,shot)] = best
        print(f'OK {shot}-shot seed {seed}: best_val_AP50={json.loads(completed.read_text())["best_val_AP50"]:.4f}')
        del ckmodel, ck
print('PASS 9/9 CCTV checkpoints; original base checkpoint untouched.')
