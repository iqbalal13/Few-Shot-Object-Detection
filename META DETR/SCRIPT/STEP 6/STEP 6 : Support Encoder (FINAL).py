# ==========================================================
# STEP 6 : Support Encoder (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class SupportEncoder(nn.Module):

    def __init__(
        self,
        prototype_extractor
    ):
        super().__init__()

        self.prototype_extractor = \
            prototype_extractor

    def forward(
        self,
        support_feature
    ):

        # support_feature:
        # [B,256,H,W]

        support_tokens, prototype = \
            self.prototype_extractor(
                support_feature
            )

        return support_tokens, prototype


support_encoder = SupportEncoder(
    prototype_extractor
).to(CONFIG["device"])


print("=" * 70)
print("STEP 6 : Support Encoder Ready")
print("=" * 70)
