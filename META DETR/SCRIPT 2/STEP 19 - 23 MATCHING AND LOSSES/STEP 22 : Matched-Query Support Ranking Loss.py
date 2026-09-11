# ==========================================================
# STEP 22: Person-Only Objective
#
# Support branch dan relation module tetap digunakan.
# Wrong semantic support dan ranking loss dihapus.
# ==========================================================

for old_name in (
    "sample_absent_wrong_support",
    "MatchedQuerySupportRankingLoss",
    "support_rank_criterion",
    "compute_combined_training_loss",
):
    globals().pop(old_name, None)

print("=" * 70)
print("STEP 22: PERSON-ONLY OBJECTIVE READY")
print("=" * 70)

print("Loss = focal + 5 * L1 + 2 * GIoU")
print("Support branch: retained")
print("Relation module: retained")
print("Negative semantic support: removed")
print("Support ranking loss: removed")

print("=" * 70)
