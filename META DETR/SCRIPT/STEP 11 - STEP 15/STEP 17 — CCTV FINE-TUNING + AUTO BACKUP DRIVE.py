
# STEP 17 — CCTV 1/3/5-SHOT TRAINING + DRIVE BACKUP
# Tidak menjalankan Stage 1 COCO lagi.

from pathlib import Path
import json
import shutil
from google.colab import drive
from metadetr_pipeline.stage2 import run_finetune

# Preflight: pastikan hasil step sebelumnya tersedia.
assert G3_READY, "Checkpoint Stage 1 belum approved"
assert STAGE1_CHECKPOINT.is_file(), "Checkpoint Stage 1 hilang"

for shot in (1, 3, 5):
    smoke_path = (
        WORKSPACE / "gates" /
        f"g4_stage2_{shot}shot_smoke.json"
    )
    assert smoke_path.is_file(), f"Smoke {shot}-shot belum ada"
    report = json.loads(smoke_path.read_text())
    assert report.get("smoke_pass") is True, f"Smoke {shot}-shot gagal"

    subset = (
        WORKSPACE / "cctv" / "fewshot" /
        f"cctv_train_{shot}shot.json"
    )
    assert subset.is_file(), f"Subset {shot}-shot tidak ditemukan"

assert (CCTV_ROOT / "annotations" / "val.json").is_file()

drive.mount("/content/drive")

BACKUP = Path(
    "/content/drive/MyDrive/MetaDETR_Thesis/Stage2_CCTV"
)
BACKUP.mkdir(parents=True, exist_ok=True)

# Simpan protokol dan manifest few-shot.
shutil.copytree(
    WORKSPACE / "cctv" / "fewshot",
    BACKUP / "fewshot_manifests",
    dirs_exist_ok=True
)

results = {}

for shot in (1, 3, 5):
    output_dir = (
        WORKSPACE / "outputs" /
        f"stage2_cctv_{shot}shot"
    )

    if (output_dir / "best.pth").exists():
        raise RuntimeError(
            f"{shot}-shot sudah memiliki best.pth. "
            "Periksa checkpoint sebelum menjalankan ulang."
        )

    print(f"\nSTART CCTV {shot}-SHOT")

    best = Path(run_finetune(
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

    assert best.is_file(), f"Model {shot}-shot gagal disimpan"

    dest = BACKUP / f"{shot}shot"
    dest.mkdir(parents=True, exist_ok=True)

    for name in (
        "best.pth",
        "latest.pth",
        "history.json",
        "transfer_report.json"
    ):
        src = output_dir / name

        if src.is_file():
            dst = dest / name
            shutil.copy2(src, dst)

            assert dst.stat().st_size == src.stat().st_size
            print("BACKUP OK:", dst)

    results[shot] = str(dest / "best.pth")
    print(f"CCTV {shot}-SHOT COMPLETE")

print("\nTRAINING FINISHED")
print(json.dumps(results, indent=2))
