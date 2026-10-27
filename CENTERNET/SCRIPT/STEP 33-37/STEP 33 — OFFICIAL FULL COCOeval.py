
# STEP 33 — Verify A2 and B2 checkpoint identity
import torch

assert os.path.isfile(paths["A2_control"])
assert os.path.isfile(paths[B2_ARM])

report33 = {}

for arm, path in paths.items():
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert ck["arm"] == arm
    assert ck["additional_exposures"] == 16000
    assert ck["base_checkpoint"] == A2_CKPT
    assert ck["base_source_sha256"] == LOCKED_SHA
    assert ck["gaussian_radius_mode"] == "v1_2_original"
    assert ck["neighbor_lambda"] == 0.0

    keys = set(ck["model_state_dict"])

    if arm == B2_ARM:
        assert "neck.gate_c3" in keys
        assert "neck.gate_c4" in keys

        assert any(
            s.startswith("neck.lateral_c3")
            for s in keys
        )
        assert any(
            s.startswith("neck.lateral_c4")
            for s in keys
        )
    else:
        assert "neck.gate_c3" not in keys
        assert "neck.gate_c4" not in keys

    report33[arm] = {
        "path": path,
        "epoch": ck["epoch"],
        "additional_exposures": ck["additional_exposures"],
        "state_parameters": len(keys)
    }

    del ck

qc_save(
    "qc33_checkpoint_integrity.json",
    report33
)

print(report33)
print("STEP 33 PASS")
