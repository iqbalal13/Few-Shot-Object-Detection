from pathlib import Path

DRIVE_STAGE1 = Path(
    "/content/drive/MyDrive/MetaDETR_Thesis/"
    "Stage1_COCO80_Freeze3_FullEpoch_BS8_RERUN_20261116"
)

DRIVE_STAGE1.mkdir(parents=True, exist_ok=True)

print("Backup baru:", DRIVE_STAGE1)
print("Checkpoint tersedia:", STAGE1_CHECKPOINT.is_file())

assert STAGE1_CHECKPOINT.is_file(), "Checkpoint hasil training tidak ditemukan"
