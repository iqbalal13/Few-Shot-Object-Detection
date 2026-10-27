
# STEP 28 — B2 optimizer and A2 weight loading utilities
import torch

def set_backbone_bn_eval(model):
    for layer in model.backbone.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()


def build_fresh_source_model():
    return CenterNetResNet101(
        num_classes=CONFIG["num_classes"],
        pretrained_backbone=True
    ).to(device)


def build_source_optimizer(model):
    # Legacy helper for Step 29 compatibility.
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )


def build_b2_optimizer(model):
    new = [
        p for n, p in model.named_parameters()
        if n.startswith("neck.lateral_")
    ]

    gates = [
        model.neck.gate_c3,
        model.neck.gate_c4
    ]

    base = [
        p for n, p in model.named_parameters()
        if not n.startswith("neck.lateral_")
        and not n.startswith("neck.gate_")
    ]

    assert len(new) == 4
    assert len(gates) == 2

    assert (
        len(new) + len(gates) + len(base)
        == len(list(model.parameters()))
    )

    return torch.optim.AdamW(
        [
            {
                "params": base,
                "lr": CONFIG["b2_base_lr"]
            },
            {
                "params": new,
                "lr": CONFIG["b2_lateral_lr"]
            },
            {
                "params": gates,
                "lr": CONFIG["b2_lateral_lr"],
                "weight_decay": 0.0
            }
        ],
        weight_decay=CONFIG["weight_decay"]
    )


def load_a2_into_b2(net, old_state):
    mismatch = net.load_state_dict(
        old_state,
        strict=False
    )

    expected_missing = {
        "neck.lateral_c3.weight",
        "neck.lateral_c3.bias",
        "neck.lateral_c4.weight",
        "neck.lateral_c4.bias",
        "neck.gate_c3",
        "neck.gate_c4"
    }

    assert (
        set(mismatch.missing_keys)
        == expected_missing
    ), mismatch.missing_keys

    assert not mismatch.unexpected_keys, (
        mismatch.unexpected_keys
    )

    assert float(net.neck.gate_c3) == 0.0
    assert float(net.neck.gate_c4) == 0.0

    return mismatch


RUN_ENGINEERING_SANITY = False
CONFIG["run_engineering_sanity"] = False

print("STEP 28 PASS — B2 optimizer and partial-load validator")
