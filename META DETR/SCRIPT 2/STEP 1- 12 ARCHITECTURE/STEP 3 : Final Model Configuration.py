# ==========================================================
# STEP 3 : Final Model Configuration
#
# LOCKED ARCHITECTURE
#
# ResNet-101
# Support Prototype Encoder
# Query Feature Encoder
# Transformer Encoder : 3
# Transformer Decoder : 6
# Object Queries       : 100
# Residual Support-Object Relation
# Support-Conditioned Binary Detection Head
# ==========================================================

CONFIG = {

    "seed": SEED,
    "device": DEVICE,

    # ------------------------------------------------------
    # INPUT / BACKBONE
    # ------------------------------------------------------

    "image_size": 640,

    "backbone": "resnet101",

    "backbone_out_channels": 2048,

    "backbone_pretrained": True,

    # ------------------------------------------------------
    # TRANSFORMER
    #
    # 3 encoder layers dipilih berdasarkan diagnosis
    # notebook lama / STEP 34.
    # ------------------------------------------------------

    "hidden_dim": 256,

    "num_queries": 100,

    "num_heads": 8,

    "num_encoder_layers": 3,

    "num_decoder_layers": 6,

    "dim_feedforward": 2048,

    # STEP 34 menggunakan no-dropout dan memberikan
    # hasil terbaik dari diagnosis sebelumnya.
    "dropout": 0.0,

    # ------------------------------------------------------
    # POSITIONAL ENCODING
    # ------------------------------------------------------

    "position_embedding": "sine_2d",

    "position_temperature": 10000,

    "position_normalize": True,

    "position_scale": (
        2.0 * math.pi
    ),

    # ------------------------------------------------------
    # SUPPORT PROTOTYPE
    # ------------------------------------------------------

    "support_hidden_dim": 512,

    # ------------------------------------------------------
    # SUPPORT-OBJECT RELATION
    # ------------------------------------------------------

    "relation_hidden_dim": 512,

    "relation_output_dim": 256,

    # ------------------------------------------------------
    # SUPPORT-CONDITIONED BINARY CLASSIFIER
    #
    # One logit per query:
    #
    # object matches support category
    # vs
    # background / no-object
    # ------------------------------------------------------

    "foreground_prior_prob": 0.05,

    "initial_logit_scale": 5.0,

    # ------------------------------------------------------
    # SOURCE META-TRAINING
    # ------------------------------------------------------

    "source_num_categories": 80,

    "source_episode_way": 1,

    "source_support_shot": 1,

    # ------------------------------------------------------
    # TARGET CCTV
    # ------------------------------------------------------

    "target_category": "person",

    "target_shots": [1, 3, 5],

    # ------------------------------------------------------
    # DETECTION LOSS
    # ------------------------------------------------------

    "focal_alpha": 0.25,

    "focal_gamma": 2.0,

    "loss_bbox_weight": 5.0,

    "loss_giou_weight": 2.0,

    # ------------------------------------------------------
    # HUNGARIAN MATCHER
    # ------------------------------------------------------

    "matcher_class_cost": 1.0,

    "matcher_bbox_cost": 5.0,

    "matcher_giou_cost": 2.0,
}


# ==========================================================
# CONFIGURATION CHECKS
# ==========================================================

assert (
    CONFIG["hidden_dim"] % 4 == 0
)

assert (
    CONFIG["hidden_dim"]
    % CONFIG["num_heads"]
    == 0
)

assert (
    CONFIG["relation_output_dim"]
    == CONFIG["hidden_dim"]
)

assert (
    CONFIG["num_encoder_layers"]
    == 3
)

assert (
    CONFIG["num_decoder_layers"]
    == 6
)

assert (
    CONFIG["source_num_categories"]
    == 80
)


print("=" * 70)
print("STEP 3 : FINAL MODEL CONFIGURATION READY")
print("=" * 70)

print(
    "Backbone        :",
    CONFIG["backbone"]
)

print(
    "Input size      :",
    CONFIG["image_size"]
)

print(
    "Hidden dim      :",
    CONFIG["hidden_dim"]
)

print(
    "Object queries  :",
    CONFIG["num_queries"]
)

print(
    "Encoder layers  :",
    CONFIG["num_encoder_layers"]
)

print(
    "Decoder layers  :",
    CONFIG["num_decoder_layers"]
)

print(
    "Attention heads :",
    CONFIG["num_heads"]
)

print(
    "FFN dim         :",
    CONFIG["dim_feedforward"]
)

print(
    "Dropout         :",
    CONFIG["dropout"]
)

print(
    "Source classes  :",
    CONFIG["source_num_categories"]
)

print(
    "Episode way     :",
    CONFIG["source_episode_way"]
)

print(
    "Support shot    :",
    CONFIG["source_support_shot"]
)

print("=" * 70)
