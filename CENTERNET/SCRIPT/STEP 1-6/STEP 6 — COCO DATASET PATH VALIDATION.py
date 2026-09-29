# ============================================================
# STEP 6 — COCO DATASET PATH VALIDATION
# ============================================================

required_paths = {
    "COCO Train Images": CONFIG["train_images"],
    "COCO Val Images": CONFIG["val_images"],
    "COCO Train Annotations": CONFIG["train_annotations"],
    "COCO Val Annotations": CONFIG["val_annotations"],
}

print("=" * 60)
print("COCO PATH VALIDATION")
print("=" * 60)

all_ok = True

for name, path in required_paths.items():

    exists = os.path.exists(path)

    print(
        f"{name:25s}: "
        f"{'OK' if exists else 'MISSING'}"
    )

    print(f"  {path}")

    if not exists:
        all_ok = False


if all_ok:

    train_count = len([
        f for f in os.listdir(CONFIG["train_images"])
        if f.endswith(".jpg")
    ])

    val_count = len([
        f for f in os.listdir(CONFIG["val_images"])
        if f.endswith(".jpg")
    ])

    print("\n" + "=" * 60)
    print("COCO DATASET READY")
    print("=" * 60)

    print("Train images :", train_count)
    print("Val images   :", val_count)

    assert train_count == 118287, \
        f"Unexpected train image count: {train_count}"

    assert val_count == 5000, \
        f"Unexpected val image count: {val_count}"

    print("\nDATASET VALIDATION PASSED")

else:

    raise FileNotFoundError(
        "COCO dataset incomplete. Re-run Step 5."
    )
