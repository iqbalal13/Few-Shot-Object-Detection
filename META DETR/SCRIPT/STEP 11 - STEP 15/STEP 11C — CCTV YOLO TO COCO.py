
# ==============================================================
# STEP 11C — CCTV YOLO TXT TO COCO JSON
# SAME-VIDEO PROTOCOL
# ==============================================================
# Recording ID sama: WARNING
# Nama frame identik: BLOCK
# File gambar identik lintas split: BLOCK
# Gambar/TXT tidak cocok: BLOCK
# Dataset Google Drive asli: TIDAK DIUBAH
# ==============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display
from collections import defaultdict

import hashlib
import json
import math
import re
import shutil

SOURCE = Path(
    '/content/drive/MyDrive/DATASET/GAMBAR CCTV'
)

TARGET = CCTV_ROOT

SPLIT_DIRS = {
    'train': 'TRAIN',
    'val': 'VALIDATION',
    'test': 'TESTING'
}

IMAGE_EXT = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp'
}

SAME_VIDEO_PROTOCOL = True
REQUIRE_ALL_PAIRS = True

# ==============================================================
# A. VALIDASI PATH
# ==============================================================

if not SOURCE.is_dir():
    from google.colab import drive
    drive.mount('/content/drive')

if not SOURCE.is_dir():
    raise FileNotFoundError(
        f'Dataset CCTV tidak ditemukan: {SOURCE}'
    )

if not isinstance(TARGET, Path):
    TARGET = Path(TARGET)

if TARGET.resolve() != Path(
    '/content/datasets/cctv'
).resolve():
    raise RuntimeError(
        f'CCTV_ROOT tidak sesuai konfigurasi: {TARGET}'
    )


def find_subdir(parent, name):
    found = [
        p for p in parent.iterdir()
        if p.is_dir()
        and p.name.casefold() == name.casefold()
    ]

    if len(found) != 1:
        raise FileNotFoundError(
            f'Folder {name} tidak ada/ambigu di {parent}'
        )

    return found[0]


def file_key(path):
    return re.sub(
        r'(?:\s*\(\d+\))+$',
        '',
        Path(path).stem
    ).strip().casefold()


def make_index(files, split, kind):
    result = {}

    for p in files:
        key = file_key(p)

        if key in result:
            raise ValueError(
                f'{split}: duplikat {kind}: '
                f'{result[key].name} dan {p.name}'
            )

        result[key] = p

    return result


def recording_key(key):
    m = re.fullmatch(
        r'(.+)_frame_(\d+)',
        key
    )

    return m.group(1) if m else None


# ==============================================================
# B. BACA PASANGAN GAMBAR DAN LABEL
# ==============================================================

pairs = {}
keys = {}
failures = []

for split, name in SPLIT_DIRS.items():
    split_dir = find_subdir(SOURCE, name)

    imdir = find_subdir(split_dir, 'IMAGE')
    lbdir = find_subdir(split_dir, 'LABEL')

    images = make_index(
        [
            p for p in imdir.iterdir()
            if p.is_file()
            and p.suffix.casefold() in IMAGE_EXT
        ],
        split,
        'gambar'
    )

    labels = make_index(
        [
            p for p in lbdir.iterdir()
            if p.is_file()
            and p.suffix.casefold() == '.txt'
        ],
        split,
        'label'
    )

    matched = sorted(
        images.keys() & labels.keys()
    )

    missing = sorted(
        images.keys() - labels.keys()
    )

    orphan = sorted(
        labels.keys() - images.keys()
    )

    print(
        f'{split.upper()}: '
        f'{len(images)} images, '
        f'{len(labels)} TXT, '
        f'{len(matched)} matched; '
        f'{len(missing)} images tanpa TXT, '
        f'{len(orphan)} TXT tanpa image'
    )

    if missing:
        print(
            'Gambar tanpa TXT:',
            [images[k].name for k in missing[:20]]
        )

    if orphan:
        print(
            'TXT tanpa gambar:',
            [labels[k].name for k in orphan[:20]]
        )

    if REQUIRE_ALL_PAIRS and (missing or orphan):
        failures.append(split)

    if not matched:
        failures.append(split)

    pairs[split] = [
        (images[k], labels[k])
        for k in matched
    ]

    keys[split] = set(matched)

if failures:
    raise ValueError(
        'GAGAL PAIRING pada split: '
        + ', '.join(sorted(set(failures)))
        + '. Perbaiki gambar/TXT sesuai CSV Step 11A. '
        'Jangan mengganti nama TXT yang tidak sesuai gambar.'
    )


# ==============================================================
# C. VALIDASI DUPLIKAT GAMBAR ANTAR-SPLIT
# ==============================================================

image_digests = defaultdict(list)

for split, items in pairs.items():
    for image_path, _ in items:
        digest = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        image_digests[digest].append(
            (split, image_path.name)
        )

for digest, items in image_digests.items():
    if len({s for s, _ in items}) > 1:
        raise ValueError(
            'Gambar dengan isi file IDENTIK '
            'di beberapa split: '
            + str(items)
        )

for a, b in [
    ('train', 'val'),
    ('train', 'test'),
    ('val', 'test')
]:
    overlap = keys[a] & keys[b]

    if overlap:
        raise ValueError(
            f'FRAME SAMA {a}/{b}: '
            f'{sorted(overlap)[:10]}'
        )

    ra = {
        recording_key(k)
        for k in keys[a]
    } - {None}

    rb = {
        recording_key(k)
        for k in keys[b]
    } - {None}

    same_recordings = sorted(ra & rb)

    if same_recordings:
        if SAME_VIDEO_PROTOCOL:
            print(
                f'WARNING {a}/{b}: '
                f'{len(same_recordings)} recording ID sama; '
                f'contoh={same_recordings[:10]}'
            )
            print(
                'Ini evaluasi SAME-VIDEO, '
                'BUKAN generalisasi cross-video.'
            )
        else:
            raise ValueError(
                f'Recording overlap {a}/{b}: '
                f'{same_recordings[:10]}'
            )


# ==============================================================
# D. KONVERSI YOLO TO COCO
# ==============================================================

coco_data = {}

for split, items in pairs.items():
    dataset = {
        'info': {
            'description':
                'CCTV person detection — same-video protocol'
        },
        'licenses': [],
        'categories': [{
            'id': 1,
            'name': 'person',
            'supercategory': 'person'
        }],
        'images': [],
        'annotations': []
    }

    positive = 0

    for image_id, (img_path, lbl_path) in enumerate(
        items,
        start=1
    ):
        with Image.open(img_path) as im:
            width, height = im.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    f'Ukuran gambar tidak valid: {img_path.name}'
                )

        dataset['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': width,
            'height': height
        })

        box_count = 0

        lines = lbl_path.read_text(
            encoding='utf-8-sig'
        ).splitlines()

        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue

            tokens = line.split()

            if len(tokens) != 5:
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    'butuh 5 nilai YOLO'
                )

            try:
                cls, cx, cy, bw, bh = map(
                    float,
                    tokens
                )
            except ValueError as exc:
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    'nilai bukan angka'
                ) from exc

            if not all(
                map(
                    math.isfinite,
                    (cls, cx, cy, bw, bh)
                )
            ):
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    'nilai NaN/Inf'
                )

            if cls != 0:
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    f'class={cls}, harus person=0'
                )

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
                and x1 >= -1e-5
                and y1 >= -1e-5
                and x2 <= 1 + 1e-5
                and y2 <= 1 + 1e-5
            ):
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    'bbox YOLO tidak valid'
                )

            x1 = max(0, x1) * width
            y1 = max(0, y1) * height
            x2 = min(1, x2) * width
            y2 = min(1, y2) * height

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                raise ValueError(
                    f'{lbl_path.name}:{line_no}: '
                    'bbox area nol'
                )

            dataset['annotations'].append({
                'id': len(dataset['annotations']) + 1,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': 0,
                'segmentation': []
            })

            box_count += 1

        if box_count:
            positive += 1

    if split == 'train' and positive < 5:
        raise ValueError(
            f'TRAIN butuh minimal 5 gambar positif, '
            f'hanya {positive}'
        )

    if (
        split != 'train'
        and not dataset['annotations']
    ):
        raise ValueError(
            f'{split.upper()}: '
            'tidak ada anotasi person untuk evaluasi'
        )

    coco_data[split] = dataset

    print(
        f'COCO {split.upper()}: '
        f'{len(dataset["images"])} gambar, '
        f'{positive} gambar positif, '
        f'{len(dataset["annotations"])} bbox'
    )


# ==============================================================
# E. SIMPAN KE DATASET LOKAL COLAB
# ==============================================================

# Gunakan staging supaya file lama tidak ditimpa
# sebelum seluruh validasi dan copy selesai.

staging = TARGET.with_name(
    TARGET.name + '_staging'
)

backup = TARGET.with_name(
    TARGET.name + '_previous'
)

if staging.exists():
    shutil.rmtree(staging)

if backup.exists():
    raise RuntimeError(
        f'Folder backup run lama masih ada: {backup}. '
        'Periksa manual sebelum mengulangi.'
    )

try:
    (staging / 'annotations').mkdir(
        parents=True
    )

    for split, items in pairs.items():
        out_img = staging / split / 'images'

        out_img.mkdir(parents=True)

        for image_path, _ in items:
            shutil.copy2(
                image_path,
                out_img / image_path.name
            )

        json_path = (
            staging / 'annotations' / f'{split}.json'
        )

        json_path.write_text(
            json.dumps(
                coco_data[split],
                ensure_ascii=False
            ),
            encoding='utf-8'
        )

    if TARGET.exists():
        TARGET.rename(backup)

    try:
        staging.rename(TARGET)
    except Exception:
        if backup.exists() and not TARGET.exists():
            backup.rename(TARGET)
        raise

    if backup.exists():
        shutil.rmtree(backup)

except Exception:
    if staging.exists():
        shutil.rmtree(staging)
    raise


# ==============================================================
# F. PREVIEW BOUNDING BOX TRAIN
# ==============================================================

example = coco_data['train']['images'][0]

preview_path = (
    TARGET
    / 'train'
    / 'images'
    / example['file_name']
)

with Image.open(preview_path) as im:
    preview = im.convert('RGB')

draw = ImageDraw.Draw(preview)

for ann in coco_data['train']['annotations']:
    if ann['image_id'] == example['id']:
        x, y, w, h = ann['bbox']

        draw.rectangle(
            (x, y, x + w, y + h),
            outline='red',
            width=4
        )

preview.thumbnail((800, 800))
display(preview)

print('SUCCESS: CCTV DATASET READY FOR STEP 15')
print('COCO Stage 1 training: OFF')
print('Stage 2 CCTV training: OFF')

print(
    'Keterbatasan: evaluasi SAME-VIDEO; '
    'review temporal/near-duplicate tetap diperlukan.'
)
