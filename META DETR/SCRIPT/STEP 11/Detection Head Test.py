# ==========================================================
# STEP 11 : Detection Head Test
# ==========================================================

with torch.no_grad():

    class_logits, boxes = detection_head(
        decoder_output
    )

print("="*60)
print("Detection Head Output")
print("="*60)

print("Classification :", class_logits.shape)

print("Bounding Box   :", boxes.shape)
