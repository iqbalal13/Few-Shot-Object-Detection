# ==========================================================
# STEP 29 : AdamW Optimizer (FINAL)
# ==========================================================

import torch.optim as optim

# ----------------------------------------------------------
# Training Hyperparameters
# ----------------------------------------------------------

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("=" * 60)
print("STEP 29 : AdamW Optimizer Ready")
print("=" * 60)

print(f"Optimizer      : {optimizer.__class__.__name__}")
print(f"Learning Rate  : {LEARNING_RATE}")
print(f"Weight Decay   : {WEIGHT_DECAY}")
