# ==========================================================
# STEP 6 : Support Encoder
# ==========================================================

class SupportEncoder(nn.Module):

    def __init__(self, backbone, prototype_extractor):
        super().__init__()

        self.backbone = backbone
        self.prototype = prototype_extractor

    def forward(self, support_image):

        feature = self.backbone(support_image)

        prototype = self.prototype(feature)

        return prototype
