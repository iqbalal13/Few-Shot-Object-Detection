# ============================================================
# STEP 19 — BACKBONE OUTPUT SHAPE TEST
# ============================================================

print("=" * 70)
print("STEP 19 — BACKBONE SHAPE TEST")
print("=" * 70)

backbone.eval()

test_input = torch.randn(
    1,
    3,
    CONFIG["input_size"],
    CONFIG["input_size"],
    device=device
)

with torch.no_grad():

    backbone_output = backbone(
        test_input
    )

print("Input shape  :", test_input.shape)
print("Output shape :", backbone_output.shape)

expected_shape = (
    1,
    2048,
    CONFIG["input_size"] // 32,
    CONFIG["input_size"] // 32
)

print("Expected     :", expected_shape)

assert tuple(
    backbone_output.shape
) == expected_shape

assert torch.isfinite(
    backbone_output
).all()

del test_input
del backbone_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSTEP 19 PASSED")
