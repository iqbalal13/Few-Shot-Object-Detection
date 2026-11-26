# Automatic post-G3 checkpoint/evaluation backup; no additional training.
if BACKUP_TO_DRIVE and RUN_MODE in ("stage1", "full_pipeline"):
    from pathlib import Path
    import hashlib
    import json
    import shutil
    import uuid

    stage1_dir = WORKSPACE / 'outputs' / 'stage1_coco80'
    status_path = stage1_dir / 'stage1_timebudget_status.json'
    assert status_path.is_file(), 'G3 status missing; training/evaluation not complete.'
    trained = json.loads(status_path.read_text(encoding='utf-8'))
    assert trained.get('status') == 'finished_with_coco_val', (
        'COCO-Val incomplete; cannot certify full-epoch result for automatic backup: ', trained)
    assert trained.get('epoch_complete') is True and int(trained.get('steps_completed', 0)) == EXPECTED_STAGE1_STEPS, (
        'Expected exactly 1,000 trained steps (8,000 images / batch 8) before auto backup.')

    files = {
        'checkpoint.pth': stage1_dir / 'checkpoint.pth',
        'stage1_timebudget_status.json': status_path,
        'stage1_console.log': stage1_dir / 'stage1_console.log',
        'train_steps_metrics.jsonl': stage1_dir / 'train_steps_metrics.jsonl',
        'results.txt': stage1_dir / 'results.txt',
        'source_run_signature.json': stage1_dir / 'source_run_signature.json',
        'source_manifest.json': WORKSPACE / 'source_manifest.json',
        'gates/g2_source_smoke.json': WORKSPACE / 'gates' / 'g2_source_smoke.json',
        'eval/latest.pth': stage1_dir / 'eval' / 'latest.pth',
        'coco80_subset/annotations/instances_train2017.json': (
            WORKSPACE / 'coco80_subset' / 'annotations' / 'instances_train2017.json'),
    }
    required = ('checkpoint.pth', 'stage1_timebudget_status.json',
                'results.txt', 'source_manifest.json', 'source_run_signature.json')
    for name in required:
        assert files[name].is_file(), 'Required Stage 1 backup source missing: ' + name

    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def backup_verified(source, target):
        source, target = Path(source), Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != source.stat().st_size or sha256_file(target) != sha256_file(source):
                raise RuntimeError('Backup collision — refusing overwrite: ' + str(target))
            print('ALREADY VERIFIED:', target.relative_to(DRIVE_STAGE1))
            return
        temp = target.with_name(target.name + '.backup_' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copy2(source, temp)
            if temp.stat().st_size != source.stat().st_size or sha256_file(temp) != sha256_file(source):
                raise IOError('Backup verification failed: ' + str(target))
            temp.replace(target)
        finally:
            if temp.exists():
                temp.unlink()
        print('BACKUP VERIFIED:', target.relative_to(DRIVE_STAGE1),
              f'({target.stat().st_size / (1024**2):.1f} MiB)')

    assert DRIVE_STAGE1.is_dir(), 'Drive preflight missing; rerun Step 11A before backup.'
    for name, source in files.items():
        if source.is_file():
            backup_verified(source, DRIVE_STAGE1 / name)
        else:
            print('OPTIONAL BACKUP FILE ABSENT:', name)
    print('AUTO BACKUP COMPLETE:', DRIVE_STAGE1)
else:
    print('Source checkpoint kept in Drive; Stage1 repeat-backup skipped.')
