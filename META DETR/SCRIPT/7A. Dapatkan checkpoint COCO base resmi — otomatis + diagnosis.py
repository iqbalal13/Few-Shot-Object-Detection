from pathlib import Path
import os, subprocess, sys, shutil

CKPT_DIR = ROOT / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CKPT_DIR / 'meta_detr_coco_base_checkpoint.pth'
OFFICIAL_CHECKPOINT_URL = 'https://drive.google.com/file/d/19tfI_XNZolDId_G5s45YTgcFKt8Ji7c8/view?usp=sharing'
AUTO_DOWNLOAD = True
MIN_CHECKPOINT_SIZE = 10 * 1024 * 1024  # filter file HTML/error/parsial; bukan validasi isi PyTorch

def checkpoint_file_plausible(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < MIN_CHECKPOINT_SIZE:
        return False
    with path.open('rb') as f:
        header = f.read(512).lstrip().lower()
    return not (header.startswith(b'<!doctype') or header.startswith(b'<html') or b'<html' in header[:200])

def put_checkpoint(candidate):
    candidate = Path(candidate)
    if not checkpoint_file_plausible(candidate):
        raise ValueError(f'File bukan kandidat checkpoint yang valid: {candidate}')
    if candidate.resolve() != CHECKPOINT_PATH.resolve():
        shutil.copy2(candidate, CHECKPOINT_PATH)
    print('Checkpoint ditemukan:', CHECKPOINT_PATH)
    print('Ukuran:', round(CHECKPOINT_PATH.stat().st_size/1048576, 1), 'MiB')

# Pertama cek lokasi standar hasil upload/Drive yang sudah ter-mount.
if not checkpoint_file_plausible(CHECKPOINT_PATH):
    if CHECKPOINT_PATH.exists():
        print('File tujuan ada tetapi terlalu kecil / tampak HTML. Disingkirkan untuk mencegah load palsu.')
        CHECKPOINT_PATH.unlink()
    candidates = [
        Path('/content/checkpoint.pth'),
        Path('/content/meta_detr_coco_base_checkpoint.pth'),
        Path('/content/drive/MyDrive/Meta-DETR/checkpoint.pth'),
        Path('/content/drive/MyDrive/Meta-DETR/meta_detr_coco_base_checkpoint.pth'),
    ]
    for candidate in candidates:
        if checkpoint_file_plausible(candidate):
            put_checkpoint(candidate)
            break

if not checkpoint_file_plausible(CHECKPOINT_PATH) and AUTO_DOWNLOAD:
    print('Mencoba download checkpoint resmi:', OFFICIAL_CHECKPOINT_URL)
    temporary_path = CKPT_DIR / 'meta_detr_download.tmp'
    temporary_path.unlink(missing_ok=True)
    # gdown >= 6 removed --fuzzy. Passing the official file ID works in modern gdown.
    OFFICIAL_CHECKPOINT_ID = '19tfI_XNZolDId_G5s45YTgcFKt8Ji7c8'
    command = [sys.executable, '-m', 'gdown', OFFICIAL_CHECKPOINT_ID,
               '-O', str(temporary_path)]
    print('gdown command:', ' '.join(command))
    log = Path('/content/meta_detr_checkpoint_download.log')
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace', timeout=900, check=False)
        log.write_text(proc.stdout or '')
        print('gdown exit code:', proc.returncode)
        print('Log:', log)
        print('\n'.join((proc.stdout or '').splitlines()[-60:]))
        if proc.returncode == 0 and checkpoint_file_plausible(temporary_path):
            temporary_path.replace(CHECKPOINT_PATH)
            print('Download checkpoint selesai.')
        else:
            print('Unduhan belum berhasil/berkas tidak valid. Ikuti langkah 7B.')
            temporary_path.unlink(missing_ok=True)
    except subprocess.TimeoutExpired:
        print('Download melewati batas waktu. Gunakan opsi 7B.')
        temporary_path.unlink(missing_ok=True)

if checkpoint_file_plausible(CHECKPOINT_PATH):
    print('✅ Checkpoint kandidat siap:', CHECKPOINT_PATH)
    print('Size (MiB):', round(CHECKPOINT_PATH.stat().st_size/1048576, 2))
else:
    print('⚠️ Checkpoint belum tersedia. Download manual melalui:', OFFICIAL_CHECKPOINT_URL)
    print('Setelah upload/mount, jalankan cell 7B. Cell 8 tidak akan berhasil sampai file tersedia.')
