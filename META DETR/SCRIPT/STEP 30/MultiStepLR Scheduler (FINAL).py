# ==========================================================
# STEP 30 : MultiStepLR Scheduler (FINAL)
# ==========================================================

from torch.optim.lr_scheduler import MultiStepLR

# ----------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------

NUM_EPOCHS = 25

# Learning rate will decrease at these epochs
MILESTONES = [15, 20]

GAMMA = 0.1

# ----------------------------------------------------------
# Scheduler
# ----------------------------------------------------------

scheduler = MultiStepLR(
    optimizer,
    milestones=MILESTONES,
    gamma=GAMMA
)

print("=" * 60)
print("STEP 30 : MultiStepLR Scheduler Ready")
print("=" * 60)

print(f"Scheduler    : {scheduler.__class__.__name__}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Milestones   : {MILESTONES}")
print(f"Gamma        : {GAMMA}")
print(f"Initial LR   : {optimizer.param_groups[0]['lr']}")
