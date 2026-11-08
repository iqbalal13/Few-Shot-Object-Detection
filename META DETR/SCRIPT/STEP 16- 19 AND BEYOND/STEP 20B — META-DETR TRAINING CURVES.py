
# ============================================================
# STEP 20B — META-DETR TRAINING CURVES
#
# 1. Training Loss vs Epoch
# 2. Validation AP@0.50 vs Epoch
#
# CCTV 1-SHOT / 3-SHOT / 5-SHOT
# NO RETRAINING — READ HISTORY.JSON ONLY
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHOTS = (1, 3, 5)

COLORS = {
    1: "#2563EB",
    3: "#F59E0B",
    5: "#16A34A"
}

SAVE_DIR = WORKSPACE / "outputs" / "training_curves"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

all_records = []

# ============================================================
# 1. LOAD TRAINING HISTORY
# ============================================================

for shot in SHOTS:

    history_path = (
        WORKSPACE / "outputs"
        / f"stage2_cctv_{shot}shot"
        / "history.json"
    )

    assert history_path.is_file(), (
        f"History not found: {history_path}"
    )

    with open(history_path, "r") as f:
        history = json.load(f)

    assert len(history) > 0, (
        f"Empty training history for {shot}-shot"
    )

    print(
        f"{shot}-shot: {len(history)} epochs loaded"
    )

    for record in history:

        epoch = int(record["epoch"]) + 1

        # Actual training loss from Meta-DETR
        train_stats = record["train_stats"]
        train_loss = float(train_stats["loss"])

        # Validation AP@0.50
        val_ap50 = float(record["val_AP50"]) * 100

        all_records.append({
            "Shot": shot,
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation AP50 (%)": val_ap50
        })

df = pd.DataFrame(all_records)

df = df.sort_values(
    by=["Shot", "Epoch"]
).reset_index(drop=True)

assert not df.empty
assert np.isfinite(
    df[["Train Loss", "Validation AP50 (%)"]]
    .to_numpy(dtype=float)
).all(), "NaN or infinite metrics detected"

# ============================================================
# 2. GRAPH 1 — TRAINING LOSS VS EPOCH
# ============================================================

plt.figure(figsize=(10, 6))

for shot in SHOTS:

    data = df[df["Shot"] == shot]

    plt.plot(
        data["Epoch"],
        data["Train Loss"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color=COLORS[shot],
        label=f"{shot}-shot"
    )

plt.title(
    "Meta-DETR Training Loss vs Epoch",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)

plt.xticks(range(1, int(df["Epoch"].max()) + 1))

plt.grid(
    True,
    linestyle="--",
    alpha=0.3
)

plt.legend(title="Few-Shot Configuration")
plt.tight_layout()

loss_path = SAVE_DIR / "MetaDETR_Training_Loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 3. GRAPH 2 — VALIDATION AP@0.50 VS EPOCH
# ============================================================

plt.figure(figsize=(10, 6))

for shot in SHOTS:

    data = df[df["Shot"] == shot]

    plt.plot(
        data["Epoch"],
        data["Validation AP50 (%)"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color=COLORS[shot],
        label=f"{shot}-shot"
    )

plt.title(
    "Meta-DETR Validation AP@0.50 vs Epoch",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation AP@0.50 (%)", fontsize=12)

plt.xticks(range(1, int(df["Epoch"].max()) + 1))

plt.ylim(
    bottom=0,
    top=max(50, df["Validation AP50 (%)"].max() * 1.15)
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.3
)

plt.legend(title="Few-Shot Configuration")
plt.tight_layout()

ap_path = SAVE_DIR / "MetaDETR_Validation_AP50.png"

plt.savefig(
    ap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 4. SAVE TRAINING HISTORY AS CSV
# ============================================================

csv_path = SAVE_DIR / "MetaDETR_Training_History.csv"

df.to_csv(csv_path, index=False)

# ============================================================
# 5. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("META-DETR TRAINING SUMMARY")
print("=" * 65)

summary = []

for shot in SHOTS:

    data = df[df["Shot"] == shot]

    summary.append({
        "Configuration": f"{shot}-shot",
        "Epochs": len(data),
        "Initial Loss": round(
            data["Train Loss"].iloc[0], 4
        ),
        "Final Loss": round(
            data["Train Loss"].iloc[-1], 4
        ),
        "Final Val AP50 (%)": round(
            data["Validation AP50 (%)"].iloc[-1], 2
        ),
        "Max Val AP50 (%)": round(
            data["Validation AP50 (%)"].max(), 2
        )
    })

display(pd.DataFrame(summary))

print("\nFILES SAVED:")
print("Training Loss:", loss_path)
print("Validation AP50:", ap_path)
print("History CSV:", csv_path)

print("\nNO TRAINING OR GPU INFERENCE PERFORMED")
print("STEP 20B COMPLETED")
