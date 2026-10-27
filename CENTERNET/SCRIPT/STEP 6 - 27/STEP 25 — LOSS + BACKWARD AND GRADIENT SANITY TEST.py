
# STEP 25 — B2 gradient checks
import torch

assert criterion.neighbor_lambda == 0.0

model.train()

for module in model.backbone.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()

images, raw = next(iter(train_loader))
x = images.to(device)

target = build_batch_centernet_targets(
    raw, device
)

optimizer_test = torch.optim.AdamW(
    [{
        "params": [
            model.neck.gate_c3,
            model.neck.gate_c4
        ],
        "lr": 1e-3
    }],
    weight_decay=0.0
)

for iteration in range(2):
    model.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16
    ):
        out = model(x)

    loss = criterion(
        out, target
    )["loss_total"]

    assert bool(torch.isfinite(loss))

    loss.backward()

    g3 = model.neck.gate_c3.grad
    g4 = model.neck.gate_c4.grad

    assert g3 is not None and g4 is not None
    assert bool(torch.isfinite(g3))
    assert bool(torch.isfinite(g4))

    print(
        "iteration", iteration,
        "gate gradients",
        float(g3),
        float(g4)
    )

    if iteration == 1:
        for mod in (
            model.neck.lateral_c3,
            model.neck.lateral_c4
        ):
            assert mod.weight.grad is not None
            assert bool(
                torch.isfinite(mod.weight.grad).all()
            )

    if iteration == 0:
        optimizer_test.step()

        # Ensure nonzero gate in this isolated test.
        with torch.no_grad():
            if abs(model.neck.gate_c3.item()) < 1e-9:
                model.neck.gate_c3.fill_(1e-3)

            if abs(model.neck.gate_c4.item()) < 1e-9:
                model.neck.gate_c4.fill_(1e-3)

# Restore initialization.
# Actual training will separately load A2 checkpoint.
with torch.no_grad():
    model.neck.gate_c3.zero_()
    model.neck.gate_c4.zero_()

model.zero_grad(set_to_none=True)

del optimizer_test, images, raw, x, target, out, loss

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("STEP 25 PASS — gates and lateral paths differentiable")
