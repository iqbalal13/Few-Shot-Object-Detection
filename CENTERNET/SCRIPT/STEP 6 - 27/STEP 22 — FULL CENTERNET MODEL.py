# ============================================================
# STEP 22 — FULL CENTERNET MODEL
# ============================================================

print("=" * 70)
print("STEP 22 — ASSEMBLE FULL CENTERNET MODEL")
print("=" * 70)


class CenterNetResNet101(nn.Module):

    def __init__(
        self,
        num_classes=80,
        pretrained_backbone=True
    ):
        super().__init__()

        self.backbone = ResNet101Backbone(
            pretrained=pretrained_backbone
        )

        self.neck = CenterNetNeck(
            in_channels=2048,
            hidden_channels=256
        )

        self.heads = CenterNetPredictionHeads(
            in_channels=256,
            head_channels=256,
            num_classes=num_classes
        )

    def forward(self, images):

        features = self.backbone(
            images
        )

        features = self.neck(
            features
        )

        outputs = self.heads(
            features
        )

        return outputs


model = CenterNetResNet101(
    num_classes=CONFIG["num_classes"],
    pretrained_backbone=True
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Model               : CenterNet + ResNet-101")
print("Classes             :", CONFIG["num_classes"])
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

assert trainable_params > 0

print("\nSTEP 22 PASSED")
