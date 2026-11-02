from google.colab import drive
from pathlib import Path
import shutil

# 1. Hubungkan Google Drive
drive.mount('/content/drive')

# 2. Lokasi hasil training Meta-DETR
SOURCE = Path(
    '/content/metadetr_work/outputs/stage1_coco80'
)

# 3. Folder penyimpanan permanen
DEST = Path(
    '/content/drive/MyDrive/MetaDETR_Thesis/Stage1_COCO80'
)
DEST.mkdir(parents=True, exist_ok=True)

# 4. File yang perlu diamankan
files_to_save = [
    'checkpoint.pth',
    'partial_latest.pth',
    'stage1_timebudget_status.json',
    'stage1_console.log',
    'train_steps_metrics.jsonl',
    'results.txt',
]

for filename in files_to_save:
    src = SOURCE / filename

    if src.is_file():
        shutil.copy2(src, DEST / filename)
        print(f'SAVED: {filename}')
    else:
        print(f'NOT FOUND: {filename}')

# 5. Simpan manifest sampling COCO
manifest = Path(
    '/content/metadetr_work/source_manifest.json'
)

if manifest.is_file():
    shutil.copy2(manifest, DEST / manifest.name)
    print('SAVED: source_manifest.json')

# 6. Verifikasi
print('\nFiles in Google Drive:')
for f in DEST.iterdir():
    print(f.name, f.stat().st_size / (1024**2), 'MB')
