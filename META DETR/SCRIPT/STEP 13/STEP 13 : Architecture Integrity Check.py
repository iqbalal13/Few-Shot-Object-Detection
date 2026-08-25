# ==========================================================
# STEP 13 : Architecture Integrity Check
# ==========================================================

print("=" * 70)
print("STEP 13 : MODEL ARCHITECTURE CHECK")
print("=" * 70)


# ==========================================================
# Shared Backbone Check
# ==========================================================

assert model.backbone is backbone

# Support / Query encoder tidak boleh membawa
# ResNet backbone masing-masing.
assert not hasattr(
    model.support_encoder,
    "backbone"
), "SupportEncoder masih mempunyai backbone sendiri."

assert not hasattr(
    model.query_encoder,
    "backbone"
), "QueryEncoder masih mempunyai backbone sendiri."


# ==========================================================
# Parameter Count
# ==========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total Parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params:,}"
)

print(
    "Shared Backbone      : ✓"
)

print(
    "Duplicate Backbone   : NONE"
)

print("=" * 70)
print("✓ STEP 13 PASSED")
print("=" * 70)
