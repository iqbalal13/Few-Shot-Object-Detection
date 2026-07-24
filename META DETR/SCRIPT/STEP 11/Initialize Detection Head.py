# ==========================================================
# STEP 11 : Initialize Detection Head
# ==========================================================

detection_head = DetectionHead(
    hidden_dim=CONFIG["hidden_dim"],
    num_classes=CONFIG["num_classes"]
).to(CONFIG["device"])

print(detection_head)
