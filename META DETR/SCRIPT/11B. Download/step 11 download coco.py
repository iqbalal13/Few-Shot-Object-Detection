#step 11 download coco
from pathlib import Path
import shutil, subprocess, zipfile, os, time

OFFICIAL_COCO_ARCHIVES = [
    ('annotations_trainval2017.zip', 'annotations/annotations_trainval2017.zip', None),
    ('val2017.zip', 'zips/val2017.zip', 5000),
    ('train2017.zip', 'zips/train2017.zip', 118287),
]

def coco_image_count(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))

def coco_complete(root: Path) -> bool:
    return (coco_image_count(root/'train2017') == 118287
            and coco_image_count(root/'val2017') == 5000
            and (root/'annotations/instances_train2017.json').is_file()
            and (root/'annotations/instances_val2017.json').is_file())

def download_official_coco(root: Path, enabled=True):
    root = Path(root)
    if not enabled:
        print('COCO download disabled. Existing COCO complete:', coco_complete(root))
        return
    if coco_complete(root):
        print('✅ COCO dataset already complete; skipping download.')
        return
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / (1024**3)
    print(f'Free disk before COCO download: {free_gb:.1f} GiB')
    if free_gb < 42:
        raise RuntimeError(f'Free space {free_gb:.1f} GiB is too low for official full COCO + extraction. '
                           'Use a Colab runtime with >= 42 GiB free, or make room first.')
    for filename, suffix, expected_count in OFFICIAL_COCO_ARCHIVES:
        if expected_count is None:
            ready = all((root/'annotations'/n).is_file()
                        for n in ('instances_train2017.json','instances_val2017.json'))
        else:
            folder = root / ('train2017' if expected_count == 118287 else 'val2017')
            ready = coco_image_count(folder) == expected_count
        if ready:
            print(f'✅ Already extracted {filename}; skipping')
            continue
        archive = root / filename
        # Only official COCO archive endpoints; use HTTP fallback if HTTPS is blocked.
        for proto in ('https', 'http'):
            url = f'{proto}://images.cocodataset.org/{suffix}'
            print('Downloading from official COCO:', url, flush=True)
            command = ['wget','--continue','--tries=5','--waitretry=8',
                       '--timeout=90','--read-timeout=90','--progress=dot:giga',
                       '-O',str(archive),url]
            p = subprocess.run(command, cwd=str(root))
            if p.returncode == 0 and zipfile.is_zipfile(archive):
                break
            print(f'Download failed/incomplete on {proto}; trying other COCO endpoint.')
        else:
            raise RuntimeError(f'Failed to download a valid COCO archive: {archive}. '
                               'Keep the partial zip; rerun this cell to resume.')
        print(f'Extracting {filename}: {archive.stat().st_size/(1024**3):.2f} GiB compressed')
        subprocess.run(['unzip','-oq',str(archive),'-d',str(root)], check=True)
        if expected_count is None:
            assert (root/'annotations/instances_train2017.json').is_file()
            assert (root/'annotations/instances_val2017.json').is_file()
            print('✅ Official COCO train/val annotations extracted.')
        else:
            count = coco_image_count(folder)
            if count != expected_count:
                raise RuntimeError(f'{folder} has {count} JPEGs; expected {expected_count}. '
                                   'Do not train; rerun download/extraction.')
            print(f'✅ Validated {count} JPEGs in {folder}')
        archive.unlink()  # Save disk only after successful extraction/validation.
    assert coco_complete(root), 'COCO validation failed after extraction.'
    print('✅ COMPLETE: official COCO train2017, val2017, and annotations available:', root)

download_official_coco(COCO_ROOT, enabled=RUN_DOWNLOAD_COCO)
