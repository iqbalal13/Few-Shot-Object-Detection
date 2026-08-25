# ==========================================================
# STEP 3 : Simplified Meta-DETR Configuration (FINAL)
# ==========================================================

import torch
import random
import numpy as np

CONFIG = {

    # ======================================================
    # Model
    # ======================================================

    "backbone": "resnet101",

    # Feature dimension used by DETR Transformer
    "hidden_dim": 256,

    # DETR object queries
    "num_queries": 100,

    # Transformer attention heads
    "num_heads": 8,

    # Transformer depth
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,

    # Feed-forward dimension
    "dim_feedforward": 2048,

    # Transformer dropout
    "dropout": 0.1,


    # ======================================================
    # COCO Classes
    # ======================================================

    # Number of foreground COCO classes
    "num_classes": 80,

    # Additional class used for unmatched DETR queries
    "num_output_classes": 81,

    # Index of background / no-object class
    "no_object_index": 80,


    # ======================================================
    # Meta-Learning Episode
    # ======================================================

    # Simplified Meta-DETR uses one target class per episode
    "episode_way": 1,

    # Base meta-training begins with 1 support example
    # K-shot will later become 1 / 3 / 5 on CCTV
    "base_support_shot": 1,

    # Support representation
    "support_feature_type": "class_specific",

    # Support-query conditioning
    "support_conditioning": "cross_attention",


    # ======================================================
    # Positional Encoding
    # ======================================================

    "position_embedding": "sine",


    # ======================================================
    # Image
    # ======================================================

    "image_size": 640,


    # ======================================================
    # Reproducibility
    # ======================================================

    "seed": 42,


    # ======================================================
    # Device
    # ======================================================

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ==========================================================
# Reproducibility
# ==========================================================

SEED = CONFIG["seed"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# Validation
# ==========================================================

assert CONFIG["hidden_dim"] % CONFIG["num_heads"] == 0

assert (
    CONFIG["num_output_classes"]
    ==
    CONFIG["num_classes"] + 1
)

assert (
    CONFIG["no_object_index"]
    ==
    CONFIG["num_classes"]
)


# ==========================================================
# Display Configuration
# ==========================================================

print("=" * 70)
print("Simplified Meta-DETR Configuration (FINAL)")
print("=" * 70)

for key, value in CONFIG.items():
    print(f"{key:25s}: {value}")

print("=" * 70)
