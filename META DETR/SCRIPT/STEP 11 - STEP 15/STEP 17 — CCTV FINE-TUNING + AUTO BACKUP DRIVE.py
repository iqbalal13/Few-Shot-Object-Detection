
# STEP 17 — CCTV FINE-TUNING + AUTO BACKUP DRIVE
# Tidak mengulang Stage 1 COCO-80.

from pathlib import Path
from google.colab import drive
import shutil
import json

from metadetr_pipeline.stage2 import run_finetune

# Pastikan Drive terhubung
drive.mount("/content/drive")

BACKUP_ROOT = Path(
    "/content/drive/MyDrive/MetaDETR_Thesis/Stage2_CCTV"
)
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

# Pemeriksaan sebelum training
assert G3_READY, "Stage 1 belum disetujui"
assert STAGE1_CHECKPOINT.is_file(), "Checkpoint COCO-80 hilang"

for shot in (1, 3, 5):
    smoke_path = (
        WORKSPACE / "gates" /
        f"g4_stage2_{shot}shot_smoke.json"
    )

    assert smoke_path.is_file(), (
        f"Smoke test {shot}-shot belum ada"
    )

    smoke = json.loads(smoke_path.read_text())
    assert smoke.get("smoke_pass") is True, (
        f"Smoke test {shot}-shot tidak PASS"
    )

print("PRECHECK PASS — Stage 2 siap dijalankan")

# Tiga training independen dari checkpoint Stage 1
for shot in (1, 3, 5):

    output_dir = (
        WORKSPACE / "outputs" /
        f"stage2_cctv_{shot}shot"
    )

    # Hindari menimpa model yang sudah selesai
    if (output_dir / "best.pth").exists():
        raise RuntimeError(
            f"{shot}-shot sudah memiliki best.pth. "
            "Backup/periksa hasilnya sebelum mengulang."
        )

    print(f"\n{'='*55}")
    print(f"START TRAINING CCTV {shot}-SHOT")
    print(f"{'='*55}")

    best_path = Path(run_finetune(
        ROOT,
        WORKSPACE,
        CCTV_ROOT,
        STAGE1_CHECKPOINT,
        shot=shot,
        epochs=STAGE2_EPOCHS,
        seed=SEED,
        lr=STAGE2_LR,
        smoke_only=False,
        patience=STAGE2_PATIENCE
    ))

    assert best_path.is_file(), (
        f"best.pth {shot}-shot tidak ditemukan"
    )

    # Backup otomatis setiap shot
    drive_folder = BACKUP_ROOT / f"{shot}shot"
    drive_folder.mkdir(parents=True, exist_ok=True)

    for name in [
        "best.pth",
        "latest.pth",
        "history.json",
        "transfer_report.json"
    ]:
        src = output_dir / name
        if src.is_file():
            dst = drive_folder / name
            shutil.copy2(src, dst)
            assert dst.stat().st_size == src.stat().st_size
            print("BACKUP OK:", dst)

    print(f"FINISHED {shot}-SHOT — BACKUP COMPLETE")

print("\nALL STAGE 2 TRAINING COMPLETED")
