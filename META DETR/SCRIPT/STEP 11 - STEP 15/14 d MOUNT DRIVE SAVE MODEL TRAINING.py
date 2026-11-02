#14 d
import json, shutil, time
from pathlib import Path
# Avoid stale sys.modules after switching from the older v8.1 overlay.
import importlib, sys
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))
importlib.invalidate_caches()
module_path = FRAMEWORK / 'metadetr_pipeline' / 'coco_person_ap.py'
assert module_path.is_file(), (
    f'coco_person_ap.py tidak ditemukan: {module_path}\n'
    'Jalankan ulang STEP 9 dari notebook v8.2.1 ini (bukan Step 14 training).'
)
from metadetr_pipeline.coco_person_ap import (
    metrics_from_snapshot, rerun_original_coco_validation)
print('Step 14D module loaded:', module_path)

RUN_COCO_PERSON_REEVALUATION = False  # EXPENSIVE FALLBACK, NEVER TRAINS
person_json = LOCAL_STAGE1/'person_coco_val_metrics.json'
snapshot = LOCAL_STAGE1/'eval'/'latest.pth'

if not snapshot.is_file():
    if not RUN_COCO_PERSON_REEVALUATION:
        print('COCOeval detail snapshot unavailable.')
        print('Cek apakah evaluasi asli masih ada di /content/metadetr_work/outputs/stage1_coco80/eval/latest.pth')
        print('Backup sebelumnya hanya menyimpan results.txt agregat, BUKAN per-category AP.')
        print('Jika hilang, pulihkan COCO dan set RUN_COCO_PERSON_REEVALUATION=True, lalu jalankan cell ini.')
    else:
        assert (LOCAL_STAGE1/'checkpoint.pth').is_file(), 'Stage1 checkpoint must be restored'
        assert (WORKSPACE/'source_manifest.json').is_file(), 'Original 8,000-image manifest required'
        print('Starting official COCO validation ONLY, no optimizer step.',flush=True)
        rerun_original_coco_validation(
            ROOT, COCO_ROOT, WORKSPACE, LOCAL_STAGE1/'checkpoint.pth', snapshot,
            seed=SEED, batch_size=1, workers=2)

if snapshot.is_file():
    metrics = metrics_from_snapshot(snapshot,person_json)
    print('\nCOCO-Val person-only precision metrics (original 80-class inference):')
    for name in ('AP50_95','AP50','AP75','AP50_95_small','AP50_95_medium','AP50_95_large'):
        if metrics.get(name) is not None:
            print(f'  {name:18s}: {100*metrics[name]:7.3f}%')
    print('Number of validation images:', metrics['num_eval_images'])
    print('Global AP50:95, inferred from same precision matrix:',
          f"{metrics['global_AP50_95_from_precision']*100:.3f}%")
    # 80-class score should agree with 14.34% Stage1 COCO-Val from original log.
    orig_results=LOCAL_STAGE1/'results.txt'
    if orig_results.is_file():
        original_rows=[]
        for line in orig_results.read_text().splitlines():
            try:
                row=json.loads(line)
                if isinstance(row.get('test_coco_eval_bbox'),list):
                    original_rows.append(row)
            except (ValueError,TypeError):pass
        if original_rows:
            global_ap=float(original_rows[-1]['test_coco_eval_bbox'][0])
            diff=abs(global_ap-metrics['global_AP50_95_from_precision'])
            print(f'Global AP consistency vs original results.txt: delta={diff:.6f}')
            if diff > .002:
                print('WARNING: snapshot might be from a different evaluation. Check provenance!')

    # Backup the exact per-class snapshot and the new lightweight metrics.
    DRIVE_COCO_EVAL.parent.mkdir(parents=True,exist_ok=True)
    for src_path, dest_path in ((snapshot,DRIVE_COCO_EVAL),
                                (person_json,DRIVE_STAGE1/'person_coco_val_metrics.json')):
        if dest_path.exists() and dest_path.stat().st_size != src_path.stat().st_size:
            raise RuntimeError(f'Conflicting file on Drive, not overwriting: {dest_path}')
        if not dest_path.exists():
            tmp=dest_path.with_name(dest_path.name+'.copying')
            shutil.copy2(src_path,tmp)
            if tmp.stat().st_size != src_path.stat().st_size:
                tmp.unlink(missing_ok=True)
                raise IOError('Backup incomplete')
            tmp.replace(dest_path)
        print('SAVED ON DRIVE:',dest_path)
    print('AP person extracted. No Stage 1 or Stage 2 training executed.')
