
# ============================================================
# STEP 17 DIAGNOSTIC — CHECK TRAINING PROGRESS
# READ ONLY — TIDAK MENGUBAH CHECKPOINT
# ============================================================

from pathlib import Path
import json

for shot in (1, 3, 5):
    output = (
        WORKSPACE / "outputs" /
        f"stage2_cctv_{shot}shot"
    )

    print(f"\n{'=' * 55}")
    print(f"CCTV {shot}-SHOT")
    print("=" * 55)

    for filename in (
        "best.pth",
        "latest.pth",
        "history.json",
        "transfer_report.json"
    ):
        path = output / filename

        if path.is_file():
            print(
                filename,
                "FOUND",
                round(path.stat().st_size / 1024**2, 2),
                "MB"
            )
        else:
            print(filename, "NOT FOUND")

    history_path = output / "history.json"

    if not history_path.is_file():
        print("STATUS: NO TRAINING HISTORY")
        continue

    history = json.loads(
        history_path.read_text(encoding="utf-8")
    )

    print("Epoch records:", len(history))

    if not history:
        print("STATUS: EMPTY HISTORY")
        continue

    best_score = float("-inf")
    best_index = -1

    for i, record in enumerate(history):
        score = float(record["val_AP50_95"])

        if score > best_score:
            best_score = score
            best_index = i

        print(
            f"Epoch {record['epoch'] + 1}: "
            f"AP50:95 = {score:.5f}"
        )

    no_improvement = len(history) - 1 - best_index

    completed_epochs = (
        len(history) >= STAGE2_EPOCHS
    )

    early_stopped = (
        no_improvement >= STAGE2_PATIENCE
    )

    print("\nBest validation AP:", best_score)
    print("Epoch without improvement:", no_improvement)
    print("Reached target epochs:", completed_epochs)
    print("Early stopping reached:", early_stopped)

    if completed_epochs or early_stopped:
        print("STATUS: APPEARS COMPLETED")
    else:
        print("STATUS: POSSIBLY INCOMPLETE")

print("\nDIAGNOSTIC FINISHED")
