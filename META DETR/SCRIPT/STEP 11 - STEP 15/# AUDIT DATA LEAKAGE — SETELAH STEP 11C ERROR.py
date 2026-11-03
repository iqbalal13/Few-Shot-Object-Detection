# AUDIT DATA LEAKAGE — SETELAH STEP 11C ERROR

from pathlib import Path
from collections import defaultdict
import re

assert "pairs" in globals(), (
    "Jalankan Step 11C terlebih dahulu agar "
    "daftar pasangan gambar-label tersedia."
)

def get_recording_id(filename):
    stem = Path(filename).stem
    stem = re.sub(
        r"(?:\s*\(\d+\))+$", "", stem
    ).casefold()

    match = re.fullmatch(
        r"(.+)_frame_\d+", stem
    )

    return match.group(1) if match else None

recordings = defaultdict(
    lambda: defaultdict(list)
)

for split, image_label_pairs in pairs.items():
    for image_path, label_path in image_label_pairs:
        rec_id = get_recording_id(image_path.name)
        if rec_id is not None:
            recordings[rec_id][split].append(
                image_path.name
            )

overlaps = {
    rec: split_files
    for rec, split_files in recordings.items()
    if len(split_files) > 1
}

print("TOTAL RECORDING IDs:", len(recordings))
print("RECORDINGS ANTAR-SPLIT:", len(overlaps))
print("=" * 60)

for rec, split_files in sorted(overlaps.items()):
    print("\nRECORDING:", rec)

    for split, filenames in sorted(
        split_files.items()
    ):
        print(
            f"  {split.upper()}: "
            f"{len(filenames)} gambar"
        )

        for name in filenames[:5]:
            print("   ", name)

print("\n" + "=" * 60)

if overlaps:
    print(
        "BELUM AMAN: sejumlah recording "
        "digunakan pada lebih dari satu split."
    )
else:
    print(
        "PASS: tidak ditemukan recording ID "
        "yang digunakan pada beberapa split."
    )
