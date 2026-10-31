from pathlib import Path
import shutil

CHECKPOINT_MODE = 'local'  # pilih: 'local' | 'drive' | 'upload'
assert CHECKPOINT_MODE in ('local', 'drive', 'upload')

if checkpoint_file_plausible(CHECKPOINT_PATH):
    print('✅ Checkpoint sudah tersedia; metode fallback tidak diperlukan.')
else:
    if CHECKPOINT_MODE == 'drive':
        from google.colab import drive
        drive.mount('/content/drive')
        candidate = Path('/content/drive/MyDrive/Meta-DETR/checkpoint.pth')
        assert candidate.is_file(), f'File tidak ditemukan di Drive: {candidate}'
        put_checkpoint(candidate)
    elif CHECKPOINT_MODE == 'upload':
        from google.colab import files
        uploaded = files.upload()  # pilih file checkpoint.pth resmi yang baru diunduh
        assert uploaded, 'Tidak ada file yang dipilih.'
        uploaded_path = Path('/content') / next(iter(uploaded.keys()))
        put_checkpoint(uploaded_path)
    else:
        candidate = Path('/content/checkpoint.pth')
        if checkpoint_file_plausible(candidate):
            put_checkpoint(candidate)
        else:
            print('Belum ada /content/checkpoint.pth. Upload checkpoint resmi lewat Files Colab,')
            print('atau ubah CHECKPOINT_MODE menjadi drive/upload lalu jalankan ulang cell ini.')

CHECKPOINT_READY = checkpoint_file_plausible(CHECKPOINT_PATH)
print('CHECKPOINT_READY:', CHECKPOINT_READY)
if CHECKPOINT_READY:
    print('✅ Path model:', CHECKPOINT_PATH)
else:
    print('⚠️ Jangan jalankan cell load model sebelum CHECKPOINT_READY=True.')
