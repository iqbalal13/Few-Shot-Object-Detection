
# ============================================================
# STEP 16 PRECHECK — CCTV 1/3/5-SHOT
# Cek resize, annotation, dan dataloader
# TANPA menjalankan training GPU
# ============================================================

from pathlib import Path
from PIL import Image
import json
import time
import signal

assert CCTV_ROOT.is_dir(), "CCTV dataset tidak ditemukan"

def timeout_handler(signum, frame):
    raise TimeoutError("Dataloader melebihi 30 detik")

for shot in (1, 3, 5):

    print(f"\n{'=' * 60}")
    print(f"CHECKING CCTV {shot}-SHOT")
    print("=" * 60, flush=True)

    args = stage2.make_original_args(ROOT, SEED)

    subset = (
        WORKSPACE / "cctv" / "fewshot"
        / f"cctv_train_{shot}shot.json"
    )

    images_dir = CCTV_ROOT / "train" / "images"

    assert subset.is_file(), (
        f"Few-shot manifest tidak ditemukan: {subset}"
    )

    # --------------------------------------------------------
    # A. Validasi semua gambar support dalam manifest
    # --------------------------------------------------------

    manifest = json.loads(subset.read_text(encoding="utf-8"))

    assert len(manifest["images"]) == shot, (
        f"{shot}-shot memiliki jumlah gambar yang salah"
    )

    for item in manifest["images"]:

        img_path = images_dir / item["file_name"]
        assert img_path.is_file(), f"Missing image: {img_path}"

        with Image.open(img_path) as img:
            w, h = img.size

        # Samakan dengan resize support 672/1152
        scale = min(
            672 / min(w, h),
            1152 / max(w, h)
        )

        area_ratio = scale ** 2

        print(
            f"Image: {img_path.name} | "
            f"Resolution: {w}x{h} | "
            f"Area ratio: {area_ratio:.4f}",
            flush=True
        )

        assert area_ratio >= 0.20, (
            f"Support area terlalu kecil: {area_ratio:.4f}"
        )

    # --------------------------------------------------------
    # B. Buat dataloader Meta-DETR original
    # --------------------------------------------------------

    loader = stage2.create_train_loader(
        ROOT,
        args,
        subset,
        images_dir,
        workers=0
    )

    ds = loader.dataset

    assert ds.NUM_SUPP == 1
    assert ds.activated_class_ids == [1]

    # --------------------------------------------------------
    # C. Validasi resize yang benar-benar aktif
    # --------------------------------------------------------

    resize = ds.support_transforms.transforms[0]

    assert resize.sizes == [672], (
        f"Wrong support sizes: {resize.sizes}"
    )

    assert resize.max_size == 1152, (
        f"Wrong max_size: {resize.max_size}"
    )

    # --------------------------------------------------------
    # D. Cek area bounding box setelah transformasi
    # --------------------------------------------------------

    ann = ds.anns_by_class[1][0]

    with Image.open(Path(ds.root) / ann["image_path"]) as img:
        image, original = ds.prepare(
            img.convert("RGB"),
            {"image_id": 1, "annotations": [ann]}
        )

    assert original["boxes"].shape[0] == 1
    assert float(original["area"][0]) > 0

    _, transformed = ds.support_transforms(
        image,
        original
    )

    assert transformed["boxes"].shape[0] == 1

    real_ratio = float(
        (
            transformed["area"][0]
            / original["area"][0]
        ).item()
    )

    print(
        f"Actual transformed area ratio: {real_ratio:.4f}",
        flush=True
    )

    assert real_ratio >= 0.20, (
        f"Actual bounding box area ratio failed: {real_ratio:.4f}"
    )

    # --------------------------------------------------------
    # E. Uji pengambilan batch tanpa GPU
    # --------------------------------------------------------

    previous_handler = signal.getsignal(signal.SIGALRM)
    t0 = time.perf_counter()

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        batch = next(iter(loader))

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    elapsed = time.perf_counter() - t0

    print(
        f"DATALOADER PASS — {shot}-SHOT | "
        f"{elapsed:.2f} seconds",
        flush=True
    )

    del batch, loader

print("\n" + "=" * 60)
print("ALL CCTV 1/3/5-SHOT PRECHECKS PASSED")
print("READY FOR STEP 16 GPU SMOKE TEST")
print("=" * 60)
