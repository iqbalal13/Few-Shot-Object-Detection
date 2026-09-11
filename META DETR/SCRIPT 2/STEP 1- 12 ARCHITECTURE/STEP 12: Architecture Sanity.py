# ==========================================================
# STEP 12: Architecture Sanity
#
# Memeriksa bentuk tensor dan nilai finite.
# Perubahan akibat support hanya informasi diagnosis.
# Tidak ada syarat bahwa bbox harus berubah.
# ==========================================================

model.eval()

with torch.inference_mode():
    query = torch.randn(
        1, 3, 256, 256,
        device=CONFIG["device"]
    )

    support_a = torch.randn_like(query)
    support_b = torch.randn_like(query)

    decoder_objects, _ = model.encode_query(query)

    prototype_a = model.encode_support(support_a)
    prototype_b = model.encode_support(support_b)

    output_a, extra_a = model.condition_and_predict(
        decoder_objects,
        prototype_a
    )

    output_b, extra_b = model.condition_and_predict(
        decoder_objects,
        prototype_b
    )

    assert output_a["pred_logits"].shape == (
        1, CONFIG["num_queries"], 1
    )

    assert output_a["pred_boxes"].shape == (
        1, CONFIG["num_queries"], 4
    )

    for output in (output_a, output_b):
        assert all(
            torch.isfinite(value).all()
            for value in output.values()
        )

        assert (
            (output["pred_boxes"] >= 0)
            & (output["pred_boxes"] <= 1)
        ).all()

    for name, a, b in (
        ("prototype", prototype_a, prototype_b),
        (
            "relation",
            extra_a["relation_features"],
            extra_b["relation_features"],
        ),
        (
            "logits",
            output_a["pred_logits"],
            output_b["pred_logits"],
        ),
        (
            "bbox",
            output_a["pred_boxes"],
            output_b["pred_boxes"],
        ),
    ):
        delta = (a - b).abs().mean().item()
        print(f"{name}: mean absolute change = {delta:.8f}")

print("STEP 12 PASS: shapes and finite outputs are valid.")
print("Support sensitivity does not prove useful support learning.")

del query, support_a, support_b
del decoder_objects, prototype_a, prototype_b
del output_a, output_b, extra_a, extra_b
del a, b, output, _
