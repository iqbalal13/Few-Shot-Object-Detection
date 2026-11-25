# 11A-2 — RESTORE STAGE1 COCO80 CHECKPOINT + REAL COCO-VAL METRICS
# No COCO dataset or COCO-base initializer is required in CCTV mode.
if RUN_MODE == 'stage2_cctv':
    from pathlib import Path
    import hashlib
    import json
    import uuid
    import shutil
    import torch

    if not DRIVE_STAGE1.is_dir():
        raise FileNotFoundError(f'Stage1 Drive backup folder missing: {DRIVE_STAGE1}')
    if not CCTV_DRIVE_ROOT.is_dir():
        raise FileNotFoundError(f'CCTV Drive dataset folder missing: {CCTV_DRIVE_ROOT}')

    def digest_file(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def restore_verified(src, dst, *, required=True):
        src, dst = Path(src), Path(dst)
        if not src.is_file():
            if required:
                raise FileNotFoundError(f'Missing Stage1 backup: {src}')
            print('Optional Stage1 artifact absent:', src.name)
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            if src.stat().st_size != dst.stat().st_size or digest_file(src) != digest_file(dst):
                raise RuntimeError('Stage1 local-vs-Drive collision, will not overwrite: ' + str(dst))
            print('STAGE1 VERIFIED:', dst.name)
            return True
        tmp = dst.with_name(dst.name + '.restore_' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copy2(src, tmp)
            if tmp.stat().st_size != src.stat().st_size or digest_file(tmp) != digest_file(src):
                raise IOError('Stage1 Drive restoration mismatch: ' + str(src))
            tmp.replace(dst)
        finally:
            if tmp.is_file():
                tmp.unlink()
        print('STAGE1 RESTORED:', dst.name)
        return True

    source_dir = WORKSPACE / 'outputs' / 'stage1_coco80'
    for name in ('checkpoint.pth', 'stage1_timebudget_status.json', 'results.txt',
                 'source_run_signature.json'):
        restore_verified(DRIVE_STAGE1 / name, source_dir / name)
    for name in ('train_steps_metrics.jsonl', 'stage1_console.log'):
        restore_verified(DRIVE_STAGE1 / name, source_dir / name, required=False)
    restore_verified(DRIVE_STAGE1 / 'source_manifest.json',
                     WORKSPACE / 'source_manifest.json', required=False)

    stage1_status = json.loads((source_dir / 'stage1_timebudget_status.json').read_text())
    assert stage1_status.get('status') == 'finished_with_coco_val', 'Stage1 COCO-Val incomplete'
    assert stage1_status.get('epoch_complete') is True, 'Stage1 was not a full epoch'
    assert int(stage1_status.get('steps_completed', -1)) == 1000, 'Expected 1,000 Stage1 steps'
    assert int(stage1_status.get('freeze_backbone_at_layer', -1)) == 3, 'Wrong source freeze level'

    valid_coco_ap = False
    for line in (source_dir / 'results.txt').read_text().splitlines():
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        val = report.get('test_coco_eval_bbox')
        if isinstance(val, list) and len(val) >= 3 and all(isinstance(x, (float,int)) for x in val[:3]):
            valid_coco_ap = True
            print('EXISTING COCO-VAL: AP50:95=', round(100*val[0],2),
                  'AP50=', round(100*val[1],2))
    assert valid_coco_ap, 'No COCO-Val metrics in Stage1 Drive backup'

    source_checkpoint = torch.load(STAGE1_CHECKPOINT, map_location='cpu', weights_only=False)
    assert isinstance(source_checkpoint, dict) and isinstance(source_checkpoint.get('model'), dict), (
        'Restored Stage1 checkpoint has invalid Meta-DETR structure')
    del source_checkpoint
    print('STAGE1 COCO80 APPROVED FOR CCTV: checkpoint + full epoch + COCO-Val PASS')
    print('Original CCTV dataset:', CCTV_DRIVE_ROOT)
else:
    print('Stage1 training mode: checkpoint restoration skipped.')
