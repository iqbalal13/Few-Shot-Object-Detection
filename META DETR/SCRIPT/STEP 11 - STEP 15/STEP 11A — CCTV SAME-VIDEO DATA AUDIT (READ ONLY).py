
# ==============================================================
# STEP 11A — CCTV SAME-VIDEO DATA AUDIT (READ ONLY)
# Tidak mengubah ROOT, dataset asli, atau model Meta-DETR.
# ==============================================================

def run_cctv_audit_11a():
    from pathlib import Path
    from collections import defaultdict
    from itertools import combinations
    from PIL import Image, ImageOps
    from IPython.display import display
    import hashlib
    import pandas as pd
    import re

    CCTV_AUDIT_ROOT = Path(
        '/content/drive/MyDrive/DATASET/GAMBAR CCTV'
    )
    REPORT_DIR = Path('/content/cctv_audit_reports')

    SPLIT_DIRS = {
        'TRAIN': 'TRAIN',
        'VALIDATION': 'VALIDATION',
        'TESTING': 'TESTING'
    }

    IMAGE_EXT = {
        '.jpg', '.jpeg', '.png', '.bmp', '.webp'
    }

    NEAR_HASH_DISTANCE = 4
    NEAR_FRAME_GAP = 120
    PREVIEW_LIMIT = 3

    if not CCTV_AUDIT_ROOT.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')

    if not CCTV_AUDIT_ROOT.is_dir():
        raise FileNotFoundError(
            f'Dataset CCTV tidak ditemukan: {CCTV_AUDIT_ROOT}'
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # A. HELPER
    # ==========================================================

    def folder(parent, name):
        matches = [
            p for p in parent.iterdir()
            if p.is_dir()
            and p.name.casefold() == name.casefold()
        ]

        if len(matches) != 1:
            raise FileNotFoundError(
                f'Folder {name} tidak ditemukan/ambigu '
                f'di {parent}'
            )

        return matches[0]

    def canonical(path):
        stem = re.sub(
            r'(?:\s*\(\d+\))+$',
            '',
            Path(path).stem
        )
        return stem.strip().casefold()

    def rec_and_frame(key):
        m = re.fullmatch(
            r'(.+)_frame_(\d+)',
            key
        )
        return (
            (m.group(1), int(m.group(2)))
            if m else (None, None)
        )

    def d_hash(path):
        with Image.open(path) as im:
            im = im.convert('L').resize((9, 8))
            px = list(im.getdata())

        v = 0
        for y in range(8):
            for x in range(8):
                v = (
                    (v << 1)
                    | int(
                        px[y * 9 + x]
                        > px[y * 9 + x + 1]
                    )
                )
        return v

    def preview(path):
        try:
            with Image.open(path) as im:
                display(
                    ImageOps.contain(
                        im.convert('RGB'),
                        (480, 270)
                    )
                )
        except Exception as exc:
            print('Preview gagal:', path, exc)

    # ==========================================================
    # B. INVENTARIS FILE
    # ==========================================================

    summary = []
    missing = []
    orphan = []
    ambiguous = []
    rows = []

    for split, dirname in SPLIT_DIRS.items():
        subdir = folder(CCTV_AUDIT_ROOT, dirname)
        image_dir = folder(subdir, 'IMAGE')
        label_dir = folder(subdir, 'LABEL')

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.casefold() in IMAGE_EXT
        )

        labels = sorted(
            p for p in label_dir.iterdir()
            if p.is_file()
            and p.suffix.casefold() == '.txt'
        )

        imap = defaultdict(list)
        lmap = defaultdict(list)

        for p in images:
            imap[canonical(p)].append(p)

        for p in labels:
            lmap[canonical(p)].append(p)

        match_keys = {
            k for k in imap.keys() & lmap.keys()
            if len(imap[k]) == 1
            and len(lmap[k]) == 1
        }

        for k in sorted(imap.keys() - lmap.keys()):
            for p in imap[k]:
                missing.append({
                    'split': split,
                    'image_name': p.name,
                    'image_path': str(p)
                })

        for k in sorted(lmap.keys() - imap.keys()):
            for p in lmap[k]:
                orphan.append({
                    'split': split,
                    'label_name': p.name,
                    'label_path': str(p)
                })

        for kind, mapping in [
            ('IMAGE', imap),
            ('LABEL', lmap)
        ]:
            for k, paths in mapping.items():
                if len(paths) > 1:
                    ambiguous.append({
                        'split': split,
                        'kind': kind,
                        'key': k,
                        'files': ' | '.join(
                            x.name for x in paths
                        )
                    })

        for p in images:
            key = canonical(p)
            rec, frame = rec_and_frame(key)

            rows.append({
                'split': split,
                'recording_id': rec,
                'frame_number': frame,
                'frame_key': key,
                'image_name': p.name,
                'image_path': str(p),
                'label_path': (
                    str(lmap[key][0])
                    if key in match_keys else ''
                )
            })

        summary.append({
            'split': split,
            'images': len(images),
            'labels': len(labels),
            'matched': len(match_keys),
            'images_without_label': sum(
                len(imap[k])
                for k in imap.keys() - lmap.keys()
            ),
            'labels_without_image': sum(
                len(lmap[k])
                for k in lmap.keys() - imap.keys()
            ),
            'ambiguous_names': (
                sum(len(v) > 1 for v in imap.values())
                + sum(len(v) > 1 for v in lmap.values())
            )
        })

    frames = pd.DataFrame(rows)

    if (
        frames.empty
        or any(
            x['images'] == 0 or x['labels'] == 0
            for x in summary
        )
    ):
        raise ValueError(
            'Ada split kosong. Audit tidak bisa PASS.'
        )

    # ==========================================================
    # C. DUPLIKAT FRAME ANTAR-SPLIT
    # ==========================================================

    duplicate_names = frames[
        frames.frame_key.isin(
            frames.groupby('frame_key')
            .split.nunique()
            .loc[lambda x: x > 1].index
        )
    ].copy()

    # File dengan byte identik meskipun nama berbeda
    frames['sha256'] = [
        hashlib.sha256(
            Path(p).read_bytes()
        ).hexdigest()
        for p in frames.image_path
    ]

    duplicate_bytes = frames[
        frames.sha256.isin(
            frames.groupby('sha256')
            .split.nunique()
            .loc[lambda x: x > 1].index
        )
    ].copy()

    # ==========================================================
    # D. RECORDING ID SAMA = WARNING
    # ==========================================================

    rec_frame = frames.dropna(
        subset=['recording_id']
    ).copy()

    counts = (
        rec_frame.groupby('recording_id')
        .split.nunique()
    )

    shared_ids = set(
        counts[counts > 1].index
    )

    shared_frames = rec_frame[
        rec_frame.recording_id.isin(shared_ids)
    ].copy()

    if shared_ids:
        shared = (
            shared_frames
            .groupby(['recording_id', 'split'])
            .size()
            .unstack(fill_value=0)
            .reindex(
                columns=list(SPLIT_DIRS),
                fill_value=0
            )
            .reset_index()
        )
    else:
        shared = pd.DataFrame(
            columns=['recording_id'] + list(SPLIT_DIRS)
        )

    # ==========================================================
    # E. FRAME BERDEKATAN DALAM RECORDING ID SAMA
    # ==========================================================

    close_frames = []

    for rec, group in shared_frames.groupby(
        'recording_id'
    ):
        items = group.to_dict('records')

        for a, b in combinations(items, 2):
            if a['split'] == b['split']:
                continue

            if (
                a['frame_number'] is None
                or b['frame_number'] is None
            ):
                continue

            gap = abs(
                int(a['frame_number'])
                - int(b['frame_number'])
            )

            if gap < NEAR_FRAME_GAP:
                close_frames.append({
                    'recording_id': rec,
                    'split_a': a['split'],
                    'image_a': a['image_name'],
                    'split_b': b['split'],
                    'image_b': b['image_name'],
                    'frame_gap': gap
                })

    # ==========================================================
    # F. KANDIDAT GAMBAR MIRIP (dHash)
    # ==========================================================

    hashes = {
        p: d_hash(p)
        for p in frames.image_path
    }

    near_frames = []
    all_items = frames.to_dict('records')

    for a, b in combinations(all_items, 2):
        if a['split'] == b['split']:
            continue

        distance = (
            hashes[a['image_path']]
            ^ hashes[b['image_path']]
        ).bit_count()

        if distance <= NEAR_HASH_DISTANCE:
            near_frames.append({
                'split_a': a['split'],
                'image_a': a['image_name'],
                'split_b': b['split'],
                'image_b': b['image_name'],
                'dhash_distance': distance
            })

    unknown = frames[
        frames.recording_id.isna()
    ].copy()

    # ==========================================================
    # G. CSV REPORTS
    # ==========================================================

    reports = {
        '01_summary.csv':
            pd.DataFrame(summary),

        '02_images_without_label.csv':
            pd.DataFrame(missing),

        '03_labels_without_image.csv':
            pd.DataFrame(orphan),

        '04_shared_recording_warning.csv':
            shared,

        '05_shared_recording_image_details.csv':
            shared_frames,

        '06_duplicate_frame_names.csv':
            duplicate_names,

        '07_identical_image_bytes.csv':
            duplicate_bytes,

        '08_near_duplicate_candidates.csv':
            pd.DataFrame(near_frames),

        '09_near_frame_numbers.csv':
            pd.DataFrame(close_frames),

        '10_ambiguous_names.csv':
            pd.DataFrame(ambiguous),

        '11_unknown_recording_names.csv':
            unknown
    }

    for filename, df in reports.items():
        df.to_csv(
            REPORT_DIR / filename,
            index=False
        )

    # ==========================================================
    # H. HASIL LANGSUNG DI COLAB
    # ==========================================================

    print('\n=== RINGKASAN DATASET ===')
    display(reports['01_summary.csv'])

    sections = [
        (
            'GAMBAR TANPA LABEL',
            '02_images_without_label.csv'
        ),
        (
            'LABEL TANPA GAMBAR',
            '03_labels_without_image.csv'
        ),
        (
            'RECORDING ID SAMA — WARNING',
            '04_shared_recording_warning.csv'
        ),
        (
            'FRAME SAMA — BLOCK',
            '06_duplicate_frame_names.csv'
        ),
        (
            'FILE GAMBAR IDENTIK — BLOCK',
            '07_identical_image_bytes.csv'
        ),
        (
            'KANDIDAT FRAME MIRIP — REVIEW',
            '08_near_duplicate_candidates.csv'
        ),
        (
            'NOMOR FRAME BERDEKATAN — REVIEW',
            '09_near_frame_numbers.csv'
        )
    ]

    for label, filename in sections:
        df = reports[filename]

        print(f'\n=== {label}: {len(df)} ===')

        if not df.empty:
            display(df.head(100))

    # ==========================================================
    # I. STATUS
    # ==========================================================

    bad = {
        'gambar_tanpa_label': len(missing),
        'label_tanpa_gambar': len(orphan),
        'nama_ambigu': len(ambiguous),
        'frame_nama_identik': len(duplicate_names),
        'gambar_file_identik': len(duplicate_bytes),
        'nama_rekaman_tidak_dikenali': len(unknown)
    }

    print('\n=== HASIL AUDIT SAME-VIDEO ===')
    print('HARUS DIPERBAIKI:', bad)
    print('WARNING recording ID sama:', len(shared))
    print('REVIEW kandidat mirip:', len(near_frames))
    print('REVIEW frame berdekatan:', len(close_frames))
    print('Laporan CSV:', REPORT_DIR)

    print(
        'Catatan: dHash mirip bukan bukti '
        'gambar identik.'
    )
    print(
        'Frame gap bukan jarak waktu '
        'tanpa metadata FPS.'
    )

    if any(bad.values()):
        print(
            'STATUS: BLOCKED — perbaiki masalah data '
            'sebelum Step 11C'
        )
    else:
        print(
            'STATUS: FORMAT PASS — tinjau warning '
            'temporal dan kemiripan'
        )

    # ==========================================================
    # J. PREVIEW
    # ==========================================================

    if missing:
        print('\nPREVIEW GAMBAR TANPA TXT')

        for item in missing[:PREVIEW_LIMIT]:
            print(item['image_name'])
            preview(item['image_path'])

    return {
        'reports': reports,
        'blocking': bad,
        'shared_recordings': shared,
        'near_duplicates': pd.DataFrame(near_frames)
    }


# Jalankan audit
cctv_audit_result = run_cctv_audit_11a()
