# ==========================================================
# STEP 4 : Shared ResNet-101 Backbone
# ==========================================================

class SharedResNet101Backbone(nn.Module):

    def __init__(self):
        super().__init__()


        weights = (

            ResNet101_Weights.DEFAULT

            if CONFIG[
                "backbone_pretrained"
            ]

            else None
        )


        resnet = resnet101(
            weights=weights
        )


        # --------------------------------------------------
        # Keep convolutional backbone only.
        #
        # Output:
        # [B, 2048, H/32, W/32]
        # --------------------------------------------------

        self.body = nn.Sequential(

            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )


        self.out_channels = 2048


    def forward(self, x):

        if x.dim() != 4:

            raise ValueError(
                "Backbone expects [B,3,H,W], "
                f"got {x.shape}"
            )


        return self.body(x)


print("=" * 70)
print("STEP 4 : SHARED RESNET-101 BACKBONE DEFINED")
print("=" * 70)
