
# ============================================================
# STEP 20 — META-DETR AP@0.50 COMPARISON
# CCTV VALIDATION VS TESTING
# 1-SHOT / 3-SHOT / 5-SHOT
# READ SAVED METRICS — NO GPU INFERENCE
# ============================================================

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SHOTS = (1, 3, 5)

records = []

for shot in SHOTS:

    folder = (
        WORKSPACE / "outputs"
        / f"stage2_cctv_{shot}shot"
    )

    val_path = folder / "val_metrics.json"
    test_path = folder / "test_metrics.json"

    assert val_path.is_file(), (
        f"Validation metrics missing: {val_path}"
    )

    assert test_path.is_file(), (
        f"Testing metrics missing: {test_path}"
    )

    with open(val_path, "r") as f:
        val = json.load(f)

    with open(test_path, "r") as f:
        test = json.load(f)

    records.append({
        "Shot": f"{shot}-shot",
        "Validation": float(val["AP50"]) * 100,
        "Testing": float(test["AP50"]) * 100
    })

df = pd.DataFrame(records)

# ============================================================
# BAR CHART
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(df))
width = 0.34

bars_val = plt.bar(
    x - width / 2,
    df["Validation"],
    width,
    label="Validation",
    color="#3B82F6"
)

bars_test = plt.bar(
    x + width / 2,
    df["Testing"],
    width,
    label="Testing",
    color="#F59E0B"
)

# Nilai di atas setiap batang
for bars in (bars_val, bars_test):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.2f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

plt.xticks(x, df["Shot"], fontsize=12)

plt.ylabel("AP@0.50 (%)", fontsize=12)
plt.xlabel("Few-Shot Configuration", fontsize=12)

plt.title(
    "Meta-DETR: AP@0.50 on CCTV Dataset\n"
    "Validation vs Testing",
    fontsize=15,
    fontweight="bold",
    pad=18
)

plt.ylim(0, 60)
plt.yticks(np.arange(0, 61, 10))
plt.grid(axis="y", linestyle="--", alpha=0.25)

plt.legend(fontsize=11)
plt.tight_layout()

# ============================================================
# SAVE GRAPH
# ============================================================

save_dir = WORKSPACE / "outputs"
save_dir.mkdir(parents=True, exist_ok=True)

graph_path = (
    save_dir / "MetaDETR_AP50_Val_vs_Test.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nAP@0.50 COMPARISON:")
display(df.round(2))

print("\nGraph saved to:", graph_path)
print("NO RETRAINING OR INFERENCE PERFORMED")
