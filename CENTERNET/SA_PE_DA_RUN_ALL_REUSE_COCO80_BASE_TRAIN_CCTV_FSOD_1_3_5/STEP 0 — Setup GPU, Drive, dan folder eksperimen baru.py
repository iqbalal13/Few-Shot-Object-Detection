%pip -q install "gdown>=5.2,<6" "pycocotools>=2.0.7" "pyyaml>=6" "pandas>=2" requests opencv-python-headless
from pathlib import Path
from google.colab import drive
from IPython.display import display
import json
import torch
import pandas as pd
import torchvision

assert torch.cuda.is_available(), 'Aktifkan GPU A100 dari menu Runtime Colab.'
print('GPU:',torch.cuda.get_device_name(0))
drive.mount('/content/drive')

PROJECT = Path('/content/SA_PE_DA_REUSE_BASE_CCTV')
PROJECT.mkdir(parents=True,exist_ok=True)
%cd /content/SA_PE_DA_REUSE_BASE_CCTV
(PROJECT/'sape_da').mkdir(exist_ok=True)
(PROJECT/'sape_da/__init__.py').touch(exist_ok=True)

# Checkpoint BASE (1 epoch) milik eksperimen lama — READ ONLY.
BASE_ROOT = Path('/content/drive/MyDrive/SA_PE_DA_CenterNet_COCO80_1EPOCH')
SOURCE_FINAL = BASE_ROOT/'source80_8k_1epoch'/'source80_final.pth'
SOURCE_DONE = BASE_ROOT/'source80_8k_1epoch'/'source80_training_complete.json'
SOURCE_VAL_METRICS = BASE_ROOT/'metrics'/'source80_coco_val5000.json'

# Nama run CCTV baru: ubah RUN_ID jika ingin melatih eksperimen BARU dari awal lagi.
CCTV_RUN_ID = 'cctv_film_retrain_run01'
TARGET_ROOT = BASE_ROOT/'target_reuse_source_checkpoint'/CCTV_RUN_ID
METRICS_ROOT = BASE_ROOT/'metrics_reuse_source_checkpoint'/CCTV_RUN_ID
TARGET_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_ROOT.mkdir(parents=True, exist_ok=True)

CCTV_DRIVE_ROOT = Path('/content/drive/MyDrive/DATASET/GAMBAR CCTV')
assert SOURCE_FINAL.is_file(), ('Base COCO-80 checkpoint TIDAK ditemukan: '+str(SOURCE_FINAL)+
    '. Pastikan notebook 1 epoch sebelumnya telah menyimpan source80_final.pth di Drive.')
assert CCTV_DRIVE_ROOT.is_dir(), 'Dataset CCTV tidak ditemukan: '+str(CCTV_DRIVE_ROOT)
print('SOURCE BASE:', SOURCE_FINAL)
print('TARGET CCTV OUTPUT:', TARGET_ROOT)
print('COCO TRAINING: OFF | CCTV FiLM TRAINING: ON')
