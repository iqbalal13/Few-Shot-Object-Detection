
# ============================================================
# STEP 41 — A3-v2 FINAL QC SUMMARY
# Validasi hasil Step 33–40, tanpa training
# ============================================================

required = [
    "report33",
    "report35",
    "report36",
    "report37",
    "report38",
    "report39",
    "report40",
    "results31",
    "QC_COUNTS",
    "QC_REFERENCE",
    "QC_CANDIDATE",
    "SELECTED_A3_V2",
    "qc_save"
]

missing = [
    key for key in required
    if key not in globals()
]

assert not missing, (
    f"Missing A3-v2 QC variables: {missing}"
)

assert QC_REFERENCE == "A3_reference"
assert QC_CANDIDATE == "A3_v2"

assert "A3_176K_reference" in results31
assert "A3_v2_selected" in results31

assert QC_COUNTS[QC_REFERENCE]["GT"] == 7117
assert QC_COUNTS[QC_CANDIDATE]["GT"] == 7117

print("A3-v2 Step 41 dependencies: PASS")
