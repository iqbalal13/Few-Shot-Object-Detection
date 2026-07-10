# ==========================================================
# STEP 7 : Query Encoder
# ==========================================================

class QueryEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def forward(self, query_image):

        feature = self.backbone(query_image)

        return feature
