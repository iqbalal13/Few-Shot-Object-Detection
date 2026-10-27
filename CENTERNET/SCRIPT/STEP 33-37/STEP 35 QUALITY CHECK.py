
# STEP 35 — Confirm learned B2 gates
import torch

ck = torch.load(
    paths[B2_ARM],
    map_location="cpu",
    weights_only=False
)

net = qc_model(B2_ARM)
probe_images, _ = next(iter(qc_loader()))

net.eval()

with torch.inference_mode():
    f = net.backbone(
        probe_images[:1].to(device)
    )

    base4 = net.neck.deconv1(f["c5"])

    lateral4 = (
        net.neck.gate_c4
        * net.neck.lateral_c4(f["c4"])
    )

    intermediate = net.neck.deconv2(
        base4 + lateral4
    )

    lateral3 = (
        net.neck.gate_c3
        * net.neck.lateral_c3(f["c3"])
    )

ratio4 = float(
    lateral4.float().norm()
    / base4.float().norm().clamp_min(1e-9)
)

ratio3 = float(
    lateral3.float().norm()
    / intermediate.float().norm().clamp_min(1e-9)
)

g3 = float(net.neck.gate_c3)
g4 = float(net.neck.gate_c4)

report35 = {
    "gate_c3": g3,
    "gate_c4": g4,
    "relative_lateral_norm_C3": ratio3,
    "relative_lateral_norm_C4": ratio4,
    "c3_active": abs(g3) > 1e-9 and ratio3 > 1e-9,
    "c4_active": abs(g4) > 1e-9 and ratio4 > 1e-9
}

qc_save(
    "qc35_gate_activation.json",
    report35
)

print(report35)

if (
    not report35["c3_active"]
    or not report35["c4_active"]
):
    print(
        "WARNING: one or both lateral branches "
        "are effectively inactive"
    )

del (
    ck, net, probe_images, f,
    base4, lateral4, intermediate, lateral3
)

torch.cuda.empty_cache()

print("STEP 35 PASS — activation audit performed")
