# ==========================================================
# STEP 13: COCO Source Setup — Person Only
# ==========================================================

import os
import sys
import time
import zipfile
import subprocess
import urllib.request
import urllib.error
import http.client

assert "CONFIG" in globals(), "Jalankan STEP 3 terlebih dahulu."
assert "COCO_DIR" in globals(), "Jalankan STEP 2 terlebih dahulu."
assert "torch" in globals(), "Jalankan STEP 1 terlebih dahulu."

COCO_CONFIG = {
    "num_classes": 1,
    "episode_way": 1,
    "support_shot": 1,
    "num_train_episodes": 8000,
    "num_val_episodes": 800,
    "image_size": CONFIG["image_size"],
    "min_bbox_size": 2.0,
    "batch_size": 1,
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
    "seed": CONFIG["seed"],
    "train_images": "train2017",
    "val_images": "val2017",
    "train_annotation": "annotations/instances_train2017.json",
    "val_annotation": "annotations/instances_val2017.json",
}

# Akses bucket COCO melalui hostname HTTPS Amazon S3.
COCO_DOWNLOAD_BASES = (
    "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/",
    "https://s3.amazonaws.com/images.cocodataset.org/",
)

COCO_ASSETS = (
    ("zips/train2017.zip", "train2017", 118287),
    ("zips/val2017.zip", "val2017", 5000),
    ("annotations/annotations_trainval2017.zip", None, None),
)


def count_coco_images(folder):
    if not os.path.isdir(folder):
        return 0

    with os.scandir(folder) as entries:
        return sum(
            entry.is_file()
            and entry.name.lower().endswith(".jpg")
            and entry.stat().st_size > 0
            for entry in entries
        )


def coco_asset_is_complete(root, asset):
    _, folder, expected_count = asset

    if folder is not None:
        return (
            count_coco_images(os.path.join(root, folder))
            >= expected_count
        )

    return all(
        os.path.isfile(os.path.join(root, COCO_CONFIG[key]))
        and os.path.getsize(os.path.join(root, COCO_CONFIG[key])) > 0
        for key in ("train_annotation", "val_annotation")
    )


def coco_root_is_complete(root):
    return all(
        coco_asset_is_complete(root, asset)
        for asset in COCO_ASSETS
    )


def download_coco_zip(relative_url, archive_path):
    if zipfile.is_zipfile(archive_path):
        print(
            "Menggunakan ZIP yang tersedia:",
            os.path.basename(archive_path),
        )
        return

    part_path = archive_path + ".part"

    if zipfile.is_zipfile(part_path):
        os.replace(part_path, archive_path)
        return

    last_error = None

    for base_url in COCO_DOWNLOAD_BASES:
        url = base_url + relative_url

        for attempt in range(1, 4):
            offset = (
                os.path.getsize(part_path)
                if os.path.isfile(part_path)
                else 0
            )

            headers = {
                "User-Agent": "COCO-Notebook/1.0",
            }

            if offset:
                headers["Range"] = f"bytes={offset}-"

            print(
                f"\nUnduh {relative_url} "
                f"| percobaan {attempt}/3"
            )
            print("URL:", url)

            request = urllib.request.Request(
                url,
                headers=headers,
            )

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=60,
                ) as response:
                    status = response.getcode()

                    if status == 206:
                        content_range = response.headers.get(
                            "Content-Range",
                            "",
                        )

                        if not content_range.startswith(
                            f"bytes {offset}-"
                        ):
                            raise RuntimeError(
                                "Posisi resume dari server tidak cocok."
                            )

                        mode = "ab" if offset else "wb"

                    elif status == 200:
                        # Server mengirim file dari awal.
                        offset = 0
                        mode = "wb"

                    else:
                        raise RuntimeError(
                            f"Respons HTTP tidak sesuai: {status}"
                        )

                    length = response.headers.get(
                        "Content-Length"
                    )

                    expected_bytes = (
                        int(length)
                        if length
                        else None
                    )

                    received = 0
                    last_report = time.monotonic()

                    with open(part_path, mode) as output_file:
                        while True:
                            chunk = response.read(
                                4 * 1024 * 1024
                            )

                            if not chunk:
                                break

                            output_file.write(chunk)
                            received += len(chunk)

                            if time.monotonic() - last_report >= 5:
                                downloaded_gib = (
                                    offset + received
                                ) / (1024 ** 3)

                                print(
                                    "\rTerunduh: "
                                    f"{downloaded_gib:.2f} GiB",
                                    end="",
                                    flush=True,
                                )

                                last_report = time.monotonic()

                    print()

                    if (
                        expected_bytes is not None
                        and received != expected_bytes
                    ):
                        raise http.client.IncompleteRead(
                            b"",
                            expected_bytes - received,
                        )

                if not zipfile.is_zipfile(part_path):
                    os.remove(part_path)

                    raise zipfile.BadZipFile(
                        "Hasil unduhan bukan ZIP yang lengkap."
                    )

                os.replace(part_path, archive_path)
                return

            except urllib.error.HTTPError as error:
                last_error = error

                if (
                    error.code == 416
                    and os.path.isfile(part_path)
                ):
                    os.remove(part_path)

                print(
                    f"HTTP {error.code}: {error.reason}"
                )

            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                zipfile.BadZipFile,
            ) as error:
                last_error = error
                print("Unduhan terhenti:", error)

    raise RuntimeError(
        f"Gagal mengunduh {relative_url}: {last_error}"
    ) from last_error


# ==========================================================
# PILIH LOKASI DATASET
# ==========================================================

candidates = list(dict.fromkeys([
    COCO_DIR,
    globals().get("COCO_ROOT", COCO_DIR),
    "/content/datasets/coco",
    "/content/MetaDETR_Final_Clean/datasets/coco",
]))

COCO_ROOT = next(
    (
        root
        for root in candidates
        if coco_root_is_complete(root)
    ),
    COCO_DIR,
)

os.makedirs(COCO_ROOT, exist_ok=True)

print("COCO root:", COCO_ROOT)


# ==========================================================
# DOWNLOAD DAN EKSTRAK BAGIAN YANG BELUM LENGKAP
# ==========================================================

for asset in COCO_ASSETS:
    relative_url, _, _ = asset
    filename = os.path.basename(relative_url)

    if coco_asset_is_complete(COCO_ROOT, asset):
        print("Sudah tersedia:", filename)
        continue

    archive_path = os.path.join(
        COCO_ROOT,
        filename,
    )

    download_coco_zip(
        relative_url,
        archive_path,
    )

    print("Mengekstrak:", filename)

    root_abs = os.path.realpath(COCO_ROOT)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = os.path.realpath(
                os.path.join(
                    root_abs,
                    member.filename,
                )
            )

            if os.path.commonpath(
                [root_abs, destination]
            ) != root_abs:
                raise RuntimeError(
                    "Path ZIP tidak valid: "
                    + member.filename
                )

        # Pembacaan saat ekstraksi sekaligus memeriksa
        # CRC tiap file ZIP.
        archive.extractall(root_abs)

    if not coco_asset_is_complete(COCO_ROOT, asset):
        raise RuntimeError(
            f"Hasil ekstraksi {filename} belum lengkap."
        )

    os.remove(archive_path)

    print("Ekstraksi selesai:", filename)


# ==========================================================
# VARIABEL UNTUK STEP 14 DAN SETERUSNYA
# ==========================================================

TRAIN_IMAGE_DIR = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["train_images"],
)

VAL_IMAGE_DIR = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["val_images"],
)

TRAIN_ANN_PATH = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["train_annotation"],
)

VAL_ANN_PATH = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["val_annotation"],
)

assert coco_root_is_complete(COCO_ROOT), (
    "Dataset COCO belum lengkap."
)


# ==========================================================
# PYCOCOTOOLS
# ==========================================================

try:
    from pycocotools.coco import COCO

except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pycocotools",
    ])

    from pycocotools.coco import COCO


# ==========================================================
# RINGKASAN
# ==========================================================

print("=" * 70)
print("STEP 13: COCO READY")
print("=" * 70)

print("COCO root          :", COCO_ROOT)

print(
    "Training classes   :",
    "person only; filter pada STEP 14 dan 16",
)

print(
    "Support per episode:",
    COCO_CONFIG["support_shot"],
)

print(
    "Train episodes     :",
    COCO_CONFIG["num_train_episodes"],
)

print(
    "Validation episodes:",
    COCO_CONFIG["num_val_episodes"],
)

print(
    "Train images       :",
    count_coco_images(TRAIN_IMAGE_DIR),
)

print(
    "Validation images  :",
    count_coco_images(VAL_IMAGE_DIR),
)

print("=" * 70)
