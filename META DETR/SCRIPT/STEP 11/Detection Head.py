# ==========================================================
# STEP 11 : Detection Head
# ==========================================================

import torch.nn as nn

class DetectionHead(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_classes=1
    ):

        super().__init__()

        # Classification Head
        self.class_head = nn.Linear(
            hidden_dim,
            num_classes + 1
        )

        # Bounding Box Head
        self.box_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, 4),

            nn.Sigmoid()

        )

    def forward(self, decoder_output):

        class_logits = self.class_head(decoder_output)

        bbox = self.box_head(decoder_output)

        return class_logits, bbox
