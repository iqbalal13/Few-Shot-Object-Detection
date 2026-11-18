from sape_da.shared import shared_cctv_support, resolve_support_ids
SHARED_DIR = Path('/content/drive/MyDrive/FSOD_SHARED_PROTOCOL')
SHARED_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (42,43,44)
SHOTS = (1,3,5)
SUPPORT_MANIFESTS = {}
for seed in SEEDS:
    target = SHARED_DIR/f'CCTV_support_seed{seed}.json'
    report = shared_cctv_support(CCTV_TRAIN_JSON,target,seed=seed)
    assert set(report['shots']['1']).issubset(report['shots']['3'])
    assert set(report['shots']['3']).issubset(report['shots']['5'])
    SUPPORT_MANIFESTS[seed]=target
    print('Seed',seed,'support filenames:',report['shots'])
print('SHARED SUPPORT READY. Source COCO checkpoint tetap tidak berubah.')
