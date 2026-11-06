
# ==========================================================
# CELL DIAGNOSIS SEBELUM STEP 16 — PATCH + VALIDASI DATALOADER
# Bukan mengganti Step 16 dan tidak mengubah dataset/COCO.
# ==========================================================

from pathlib import Path
from PIL import Image
import importlib
import inspect
import json
import signal
import time

import metadetr_pipeline.stage2 as stage2

module_path = Path(stage2.__file__).resolve()

old_line = (
    "support_tf=T.Compose(["
    "T.RandomResize([448],max_size=672),normalize])"
)

new_line = (
    "support_tf=T.Compose(["
    "T.RandomResize([672],max_size=1152),normalize])"
)

# ==========================================================
# A. PATCH RESIZE SUPPORT DI stage2.py
# ==========================================================

source = module_path.read_text(encoding="utf-8")

if old_line in source:
    assert source.count(old_line) == 1, (
        "Ada lebih dari satu support_tf lama"
    )

    module_path.write_text(
        source.replace(old_line, new_line),
        encoding="utf-8"
    )

    print("PATCH APPLIED:", module_path, flush=True)

elif new_line in source:
    print("PATCH ALREADY APPLIED:", module_path, flush=True)

else:
    raise RuntimeError(
        "Baris support_tf tidak ditemukan; "
        "jangan lanjut Step 16."
    )

# Reload module supaya perubahan aktif
importlib.invalidate_caches()
stage2 = importlib.reload(stage2)

assert new_line in inspect.getsource(
    stage2.create_train_loader
), "Fungsi create_train_loader masih memakai konfigurasi lama"

print(
    "ACTIVE SUPPORT RESIZE: min=672, max=1152",
    flush=True
)

# ==========================================================
# B. CEK RESOLUSI SEMUA SUPPORT 1/3/5-SHOT
# ==========================================================

for shot in (1, 3, 5):

    subset = (
        WORKSPACE / "cctv" / "fewshot"
        / f"cctv_train_{shot}shot.json"
    )

    assert subset.is_file(), (
        f"Manifest belum ada (jalankan Step 15): {subset}"
    )

    dataset = json.loads(
        subset.read_text(encoding="utf-8")
    )

    assert len(dataset["images"]) == shot, (
        f"{shot}-shot: jumlah gambar tidak sesuai"
    )

    for item in dataset["images"]:

        img_path = (
            CCTV_ROOT / "train" / "images"
            / item["file_name"]
        )

        assert img_path.is_file(), (
            f"Gambar tidak ditemukan: {img_path}"
        )

        with Image.open(img_path) as im:
            w, h = im.size

        scale = min(
            672 / min(w, h),
            1152 / max(w, h)
        )

        ratio = scale ** 2

        print(
            f"{shot}-shot | {item['file_name']} | "
            f"{w}x{h} | area_ratio={ratio:.4f}",
            flush=True
        )

        if ratio < 0.20:
            raise RuntimeError(
                "Resize support masih di bawah "
                f"ambang 20%: {img_path.name}"
            )

print(
    "RESIZE CHECK PASS untuk 1/3/5-shot",
    flush=True
)

# ==========================================================
# C. TEST DATALOADER 1-SHOT TANPA TRAINING GPU
# ==========================================================

subset_1 = (
    WORKSPACE / "cctv" / "fewshot"
    / "cctv_train_1shot.json"
)

images_dir = CCTV_ROOT / "train" / "images"

args = stage2.make_original_args(ROOT, SEED)

loader = stage2.create_train_loader(
    ROOT,
    args,
    subset_1,
    images_dir,
    workers=0
)

def _timeout_handler(signum, frame):
    raise TimeoutError(
        "next(iter(loader)) melewati 30 detik"
    )

previous_handler = signal.getsignal(
    signal.SIGALRM
)

start = time.perf_counter()

print(
    "DATALOADER START (1-shot)",
    flush=True
)

try:
    signal.signal(
        signal.SIGALRM,
        _timeout_handler
    )

    signal.alarm(30)

    batch = next(iter(loader))

    signal.alarm(0)

except TimeoutError as e:
    raise RuntimeError(
        "DATALOADER MASIH TIMEOUT: "
        "jangan jalankan Step 16. "
        "Perlu audit sampling/transformasi."
    ) from e

finally:
    signal.alarm(0)
    signal.signal(
        signal.SIGALRM,
        previous_handler
    )

print(
    f"DATALOADER PASS: "
    f"{time.perf_counter() - start:.2f} detik",
    flush=True
)

print(
    "PATCH + DATALOADER CHECK PASS. "
    "Step 16 boleh diuji.",
    flush=True
)
