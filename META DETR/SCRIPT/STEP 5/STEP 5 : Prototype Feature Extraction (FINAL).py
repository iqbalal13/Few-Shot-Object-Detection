# ==========================================================
# STEP 5 : Prototype Feature Extraction (FINAL)
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeFeatureExtractor(nn.Module):

    def __init__(self, hidden_dim=256):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feature_map):

        # --------------------------------------------------
        # feature_map:
        # [B, 256, H, W]
        # --------------------------------------------------

        B, C, H, W = feature_map.shape

        # --------------------------------------------------
        # Preserve spatial support information
        #
        # [B,256,H,W]
        # ->
        # [B,H*W,256]
        # --------------------------------------------------

        support_tokens = (
            feature_map
            .flatten(2)
            .transpose(1, 2)
        )

        support_tokens = self.norm(
            support_tokens
        )

        # --------------------------------------------------
        # Global class prototype
        #
        # Average spatial support tokens
        # --------------------------------------------------

        prototype = support_tokens.mean(
            dim=1
        )

        # Normalize prototype representation
        prototype = F.normalize(
            prototype,
            p=2,
            dim=-1
        )

        return support_tokens, prototype


prototype_extractor = PrototypeFeatureExtractor(
    hidden_dim=CONFIG["hidden_dim"]
).to(CONFIG["device"])


print("=" * 70)
print("STEP 5 : Prototype Feature Extractor Ready")
print("=" * 70)
