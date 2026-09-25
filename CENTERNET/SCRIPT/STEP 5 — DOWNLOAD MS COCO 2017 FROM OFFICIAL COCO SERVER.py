# ============================================================
# STEP 5 — DOWNLOAD MS COCO 2017 FROM OFFICIAL COCO SERVER
# ============================================================

DATA_ROOT = CONFIG["data_root"]
os.makedirs(DATA_ROOT, exist_ok=True)

files = {
    "train2017.zip":
        "http://images.cocodataset.org/zips/train2017.zip",

    "val2017.zip":
        "http://images.cocodataset.org/zips/val2017.zip",

    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

for filename, url in files.items():

    zip_path = os.path.join(DATA_ROOT, filename)

    if not os.path.exists(zip_path):
        print(f"\nDownloading {filename}...")
        subprocess.run(
            ["wget", "-c", url, "-O", zip_path],
            check=True
        )
    else:
        print(f"\n{filename} already downloaded.")

    print(f"Extracting {filename}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_ROOT)

print("\n" + "=" * 60)
print("COCO 2017 DOWNLOAD & EXTRACTION COMPLETE")
print("=" * 60)

print("Dataset root :", DATA_ROOT)
