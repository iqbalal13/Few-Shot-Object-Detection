# ============================================================
# STEP 28 — SOURCE MODEL / OPTIMIZER HELPERS ONLY
# NO TRAINING
# ============================================================

import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

def set_backbone_bn_eval(model):
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_fresh_source_model():
    return CenterNetResNet101(
        num_classes=CONFIG["num_classes"],
        pretrained_backbone=True
    ).to(device)


def build_source_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )


RUN_TINY_OVERFIT = False
RUN_ENGINEERING_SANITY = False

CONFIG["run_tiny_overfit_diagnostic"] = False
CONFIG["run_engineering_sanity"] = False

assert not RUN_TINY_OVERFIT
assert not RUN_ENGINEERING_SANITY

print("STEP 28 PASSED")
print("Model and optimizer helpers available.")
print("No training executed.")
