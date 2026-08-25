# ==========================================================
# STEP 11 : Detection Head (FINAL)
# ==========================================================

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers
    ):
        super().__init__()

        layers = []

        for i in range(num_layers):

            in_dim = (
                input_dim
                if i == 0
                else hidden_dim
            )

            out_dim = (
                output_dim
                if i == num_layers - 1
                else hidden_dim
            )

            layers.append(
                nn.Linear(
                    in_dim,
                    out_dim
                )
            )

            if i < num_layers - 1:
                layers.append(
                    nn.ReLU()
                )

        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.layers(x)


class DetectionHead(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_output_classes=CONFIG["num_output_classes"]
    ):
        super().__init__()

        # ==================================================
        # Classification
        # ==================================================

        self.class_head = nn.Linear(
            hidden_dim,
            num_output_classes
        )

        # ==================================================
        # Bounding Box Regression
        #
        # output:
        # cx, cy, w, h normalized [0,1]
        # ==================================================

        self.box_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3
        )

    def forward(self, decoder_output):

        class_logits = self.class_head(
            decoder_output
        )

        boxes = torch.sigmoid(
            self.box_head(
                decoder_output
            )
        )

        return class_logits, boxes


# ==========================================================
# Initialize
# ==========================================================

detection_head = DetectionHead().to(
    CONFIG["device"]
)

print("=" * 70)
print("STEP 11 : Detection Head Ready")
print("=" * 70)
print(detection_head)
