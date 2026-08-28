# ==========================================================
# V3 ARCHITECTURE + SUPPORT CAUSALITY SANITY
# ==========================================================

import torch


print("=" * 70)
print("V3 ARCHITECTURE SANITY")
print("=" * 70)

model.eval()


dummy_query = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"],
    device=CONFIG["device"]
)

dummy_support_a = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"],
    device=CONFIG["device"]
)

dummy_support_b = torch.randn(
    1,
    3,
    CONFIG["image_size"],
    CONFIG["image_size"],
    device=CONFIG["device"]
)


with torch.inference_mode():

    output_a = model(
        dummy_support_a,
        dummy_query
    )

    output_b = model(
        dummy_support_b,
        dummy_query
    )


print(
    "Logits shape :",
    output_a[
        "pred_logits"
    ].shape
)

print(
    "Boxes shape  :",
    output_a[
        "pred_boxes"
    ].shape
)


assert (
    output_a["pred_logits"].shape
    ==
    (
        1,
        CONFIG["num_queries"],
        1
    )
)

assert (
    output_a["pred_boxes"].shape
    ==
    (
        1,
        CONFIG["num_queries"],
        4
    )
)


logit_delta = (
    output_a["pred_logits"]
    -
    output_b["pred_logits"]
).abs().mean().item()

bbox_delta = (
    output_a["pred_boxes"]
    -
    output_b["pred_boxes"]
).abs().mean().item()


print("-" * 70)

print(
    f"Support-change |Δ logit| : "
    f"{logit_delta:.8f}"
)

print(
    f"Support-change |Δ bbox|  : "
    f"{bbox_delta:.8f}"
)

print("=" * 70)

if (
    logit_delta == 0.0
    and
    bbox_delta == 0.0
):
    raise RuntimeError(
        "V3 support path appears inactive."
    )

print(
    "✓ V3 forward pass OK"
)

print(
    "✓ Support has direct causal path"
)
