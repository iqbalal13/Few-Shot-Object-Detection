# ============================================================
# STEP 3 — SEED & REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("=" * 60)
print("REPRODUCIBILITY SETUP")
print("=" * 60)
print("Random seed :", SEED)
print("Status      : READY")
