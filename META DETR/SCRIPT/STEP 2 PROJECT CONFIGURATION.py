# ==========================================================
# STEP 2 : Project Configuration
# ==========================================================

import os

# Root Project
PROJECT_ROOT = "/content/MetaDETR_Thesis"

# Dataset
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Checkpoint
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Log
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories
for folder in [
    PROJECT_ROOT,
    DATASET_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LOG_DIR
]:
    os.makedirs(folder, exist_ok=True)

print("="*60)
print("Project Configuration")
print("="*60)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Dataset      : {DATASET_DIR}")
print(f"Checkpoint   : {CHECKPOINT_DIR}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Logs         : {LOG_DIR}")
