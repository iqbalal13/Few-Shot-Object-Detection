
# ============================================================
# STEP 5 — DOWNLOAD & EXTRACT COCO 2017
# ============================================================

import os
import subprocess
import zipfile

ROOT = CONFIG["data_root"]
os.makedirs(ROOT, exist_ok=True)

datasets = [
    ("train2017.zip",
     "https://images.cocodataset.org/zips/train2017.zip",
     "train2017", 118287),

    ("val2017.zip",
     "https://images.cocodataset.org/zips/val2017.zip",
     "val2017", 5000),

    ("annotations_trainval2017.zip",
     "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
     "annotations", None),
]

def is_complete(folder, count):
    path = os.path.join(ROOT, folder)

    if count is None:
        return all(os.path.isfile(CONFIG[k]) for k in
                   ("train_annotations", "val_annotations"))

    return os.path.isdir(path) and sum(
        f.name.endswith(".jpg") for f in os.scandir(path)
    ) == count

for filename, url, folder, count in datasets:
    print(f"\n=== {filename} ===", flush=True)

    if is_complete(folder, count):
        print("Already complete, skipping.")
        continue

    zip_path = os.path.join(ROOT, filename)

    if not zipfile.is_zipfile(zip_path):
        subprocess.run([
            "wget", "-c", "--no-check-certificate",
            "--show-progress", "-O", zip_path, url
        ], check=True)

    assert zipfile.is_zipfile(zip_path), f"Invalid ZIP: {filename}"

    print("Extracting...", flush=True)
    subprocess.run(
        ["unzip", "-oq", zip_path, "-d", ROOT],
        check=True
    )

    assert is_complete(folder, count), f"Incomplete: {folder}"

    os.remove(zip_path)
    print("DONE")

print("\nSTEP 5 PASSED — COCO 2017 READY")
