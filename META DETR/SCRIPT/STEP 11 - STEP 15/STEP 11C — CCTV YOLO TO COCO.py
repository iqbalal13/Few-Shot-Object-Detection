# STEP 11C — CCTV YOLO TO COCO
# Compatible with TRAIN/IMAGE/LABEL,
# VALIDATION/IMAGE/LABEL, TESTING/IMAGE/LABEL

from google.colab import drive
from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display
import json
import math
import re
import shutil

drive.mount("/content/drive")

SOURCE = Path("/content/drive/MyDrive/DATASET/GAMBAR CCTV")
TARGET = CCTV_ROOT

print("CCTV SOURCE:", SOURCE)
assert SOURCE.is_dir(), f"Folder CCTV tidak ditemukan: {SOURCE}"

SPLITS = {
    "train": ("TRAIN", "IMAGE", "LABEL"),
    "val": ("VALIDATION", "IMAGE", "LABEL"),
    "test": ("TESTING", "IMAGE", "LABEL")
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_folder(parent, name):
    matches = [
        p for p in parent.iterdir()
        if p.is_dir() and p.name.casefold() == name.casefold()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Folder {name} tidak ditemukan di {parent}. "
            f"Folder tersedia: "
            f"{[p.name for p in parent.iterdir() if p.is_dir()]}"
        )
    return matches[0]

def file_key(path):
    # frame_120 (1).jpg dan frame_120.txt -> frame_120
    stem = Path(path).stem
    return re.sub(
        r"(?:\s*\(\d+\))+$", "", stem
    ).strip().casefold()

def index_files(files, split, kind):
    result = {}
    for p in files:
        key = file_key(p)
        if key in result:
            raise ValueError(
                f"{split}: duplikat {kind}: "
                f"{result[key].name} dan {p.name}"
            )
        result[key] = p
    return result

def recording_key(name):
    match = re.fullmatch(r"(.+)_frame_\d+", name)
    return match.group(1) if match else None

assert SOURCE.is_dir(), f"Folder CCTV tidak ada: {SOURCE}"

# A. Cari pasangan gambar-label
pairs = {}
keys = {}

for split, names in SPLITS.items():
    split_dir = find_folder(SOURCE, names[0])
    img_dir = find_folder(split_dir, names[1])
    lbl_dir = find_folder(split_dir, names[2])

    images = index_files(
        [p for p in img_dir.iterdir()
         if p.is_file() and p.suffix.casefold() in IMAGE_EXT],
        split, "gambar"
    )

    labels = index_files(
        [p for p in lbl_dir.iterdir()
         if p.is_file() and p.suffix.casefold() == ".txt"],
        split, "label"
    )

    matched = sorted(set(images) & set(labels))
    missing = sorted(set(images) - set(labels))
    extra = sorted(set(labels) - set(images))

    print(
        f"{split.upper()}: "
        f"{len(images)} images, "
        f"{len(labels)} labels, "
        f"{len(matched)} matched"
    )
    print(
        f"  Gambar tanpa label: {len(missing)}; "
        f"label tanpa gambar: {len(extra)}"
    )

    if missing:
        print("  Dilewati:", [images[k].name for k in missing[:5]])
    if extra:
        print("  Label tanpa gambar:", [labels[k].name for k in extra[:5]])

    if not matched:
        raise ValueError(
            f"{split}: Tidak ada gambar dan TXT yang berpasangan. "
            "Pastikan LABEL berisi file .txt asli."
        )

    pairs[split] = [
        (images[k], labels[k]) for k in matched
    ]
    keys[split] = set(matched)

# B. Cek kemungkinan kebocoran split
for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:
    overlap = keys[a] & keys[b]
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE {a}/{b}: "
            f"frame sama {sorted(overlap)[:5]}"
        )

    rec_a = {
        recording_key(k) for k in keys[a]
    } - {None}
    rec_b = {
        recording_key(k) for k in keys[b]
    } - {None}

    same_recordings = rec_a & rec_b
    if same_recordings:
        raise ValueError(
            f"POTENSI DATA LEAKAGE {a}/{b}: "
            f"rekaman sama {sorted(same_recordings)[:5]}. "
            "Pisahkan berdasarkan rekaman."
        )

print("SPLIT CHECK PASSED")

# C. Konversi YOLO ke COCO
datasets = {}

for split, items in pairs.items():
    data = {
        "info": {"description": "CCTV person detection"},
        "licenses": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }],
        "images": [],
        "annotations": []
    }

    positive = 0

    for image_id, (img_path, lbl_path) in enumerate(items, 1):
        with Image.open(img_path) as img:
            W, H = img.size

        data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": W,
            "height": H
        })

        box_count = 0

        lines = lbl_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue

            values = line.split()
            if len(values) != 5:
                raise ValueError(
                    f"{lbl_path.name}:{line_no}: "
                    "label harus berisi 5 angka YOLO"
                )

            try:
                cls, cx, cy, bw, bh = map(float, values)
            except ValueError as exc:
                raise ValueError(
                    f"{lbl_path.name}:{line_no}: "
                    "ada nilai bukan angka"
                ) from exc

            if not all(map(
                math.isfinite, (cls, cx, cy, bw, bh)
            )):
                raise ValueError(
                    f"{lbl_path.name}:{line_no}: NaN/Inf"
                )

            if cls != 0:
                raise ValueError(
                    f"{lbl_path.name}:{line_no}: "
                    f"class {cls}, diharapkan person=0"
                )

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            if not (
                0 <= cx <= 1 and 0 <= cy <= 1 and
                0 < bw <= 1 and 0 < bh <= 1 and
                x1 >= -1e-5 and y1 >= -1e-5 and
                x2 <= 1 + 1e-5 and y2 <= 1 + 1e-5
            ):
                raise ValueError(
                    f"{lbl_path.name}:{line_no}: "
                    "bbox YOLO tidak valid"
                )

            x1 = max(0, x1) * W
            y1 = max(0, y1) * H
            x2 = min(1, x2) * W
            y2 = min(1, y2) * H

            w, h = x2 - x1, y2 - y1

            if min(w, h) <= 0:
                raise ValueError("BBox memiliki area nol")

            data["annotations"].append({
                "id": len(data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            box_count += 1

        if box_count:
            positive += 1

    if split == "train" and positive < 5:
        raise ValueError(
            f"Train membutuhkan minimal 5 gambar "
            f"beranotasi person. Tersedia {positive}."
        )

    if split != "train" and not data["annotations"]:
        raise ValueError(
            f"{split}: tidak ada anotasi person untuk evaluasi"
        )

    datasets[split] = data

    print(
        f"COCO {split}: "
        f"{len(data['images'])} gambar, "
        f"{positive} gambar positif, "
        f"{len(data['annotations'])} bbox"
    )

# D. Simpan dataset lokal
ann_dir = TARGET / "annotations"
ann_dir.mkdir(parents=True, exist_ok=True)

for split, items in pairs.items():
    image_out = TARGET / split / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    for src, _ in items:
        dst = image_out / src.name
        if (
            not dst.is_file()
            or dst.stat().st_size != src.stat().st_size
        ):
            shutil.copy2(src, dst)

    json_path = ann_dir / f"{split}.json"
    temp = ann_dir / f"{split}.json.tmp"

    temp.write_text(
        json.dumps(datasets[split], ensure_ascii=False),
        encoding="utf-8"
    )
    temp.replace(json_path)

    print("SAVED:", json_path)

# E. Preview satu gambar TRAIN beserta bbox
example = datasets["train"]["images"][0]
preview_path = (
    TARGET / "train" / "images" / example["file_name"]
)

with Image.open(preview_path) as img:
    preview = img.convert("RGB")

draw = ImageDraw.Draw(preview)

for ann in datasets["train"]["annotations"]:
    if ann["image_id"] == example["id"]:
        x, y, w, h = ann["bbox"]
        draw.rectangle(
            [x, y, x + w, y + h],
            outline="red", width=4
        )

preview.thumbnail((800, 800))
display(preview)

print("=" * 50)
print("SUCCESS: CCTV DATASET READY FOR STEP 15")
print("Stage 1 COCO training: OFF")
print("Stage 2 CCTV training: OFF")
