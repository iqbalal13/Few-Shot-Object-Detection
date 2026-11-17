from pathlib import Path
from google.colab import drive
import json, hashlib, shutil, os

drive.mount('/content/drive')
PROJECT = Path('/content/SA_PE_DA_LOCKED_COCO80')
PROJECT.mkdir(parents=True,exist_ok=True)
%cd /content/SA_PE_DA_LOCKED_COCO80
for sub in ('sape_da','tests'):(PROJECT/sub).mkdir(exist_ok=True)
(PROJECT/'sape_da/__init__.py').touch()

# Berbagi manifest COCO dengan Meta-DETR
SHARED_DIR = Path('/content/drive/MyDrive/FSOD_SHARED_PROTOCOL')
SHARED_DIR.mkdir(parents=True,exist_ok=True)
SOURCE_MANIFEST_PATH = SHARED_DIR/'COCO80_8000_seed42_source_manifest.json'

# RUN BARU; jangan overwrite source/target eksperimen person-only sebelumnya.
SAVE_ROOT = Path('/content/drive/MyDrive/SA_PE_DA_CenterNet_COCO80_1EPOCH')
SOURCE_ROOT = SAVE_ROOT/'source80_8k_1epoch'
TARGET_ROOT = SAVE_ROOT/'target'
METRICS_ROOT = SAVE_ROOT/'metrics'
for folder in (SOURCE_ROOT,TARGET_ROOT,METRICS_ROOT):folder.mkdir(parents=True,exist_ok=True)
SOURCE_FINAL = SOURCE_ROOT/'source80_final.pth'
SOURCE_DONE = SOURCE_ROOT/'source80_training_complete.json'
SOURCE_VAL_METRICS = METRICS_ROOT/'source80_coco_val5000.json'
print('Source checkpoint:',SOURCE_FINAL)
print('Shared COCO manifest:',SOURCE_MANIFEST_PATH)
print('CCTV images:', '/content/drive/MyDrive/DATASET/GAMBAR CCTV')
print('MODE: SINGLE COCO EPOCH = exactly 8,000 image exposures (batch size 1).')
print('Output is isolated from prior 60-minute experiment:', SAVE_ROOT)
