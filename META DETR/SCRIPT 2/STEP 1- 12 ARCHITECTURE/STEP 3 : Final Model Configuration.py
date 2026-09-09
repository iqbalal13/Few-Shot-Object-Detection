# ==========================================================
# STEP 3 : Final Model Configuration
# FINAL CLEAN ARCHITECTURE — LOCKED
# ==========================================================

CONFIG = {

    # ------------------------------------------------------
    # Reproducibility / device
    # ------------------------------------------------------

    "seed":
        SEED,

    "device":
        DEVICE,


    # ------------------------------------------------------
    # Input / Backbone
    # ------------------------------------------------------

    "image_size":
        640,

    "backbone":
        "resnet101",

    "backbone_out_channels":
        2048,

    "backbone_pretrained":
        True,


    # ------------------------------------------------------
    # Transformer
    # ------------------------------------------------------

    "hidden_dim":
        256,

    "num_queries":
        100,

    "num_heads":
        8,

    "num_encoder_layers":
        6,

    "num_decoder_layers":
        6,

    "dim_feedforward":
        2048,

    "dropout":
        0.1,


    # ------------------------------------------------------
    # Positional Encoding
    # ------------------------------------------------------

    "position_embedding":
        "sine_2d",

    "position_temperature":
        10000,

    "position_normalize":
        True,

    "position_scale":
        2.0 * math.pi,


    # ------------------------------------------------------
    # Support Prototype
    # ------------------------------------------------------

    "support_hidden_dim":
        512,


    # ------------------------------------------------------
    # Support-Object Relation
    # ------------------------------------------------------

    "relation_hidden_dim":
        512,

    "relation_output_dim":
        256,


    # ------------------------------------------------------
    # Explicit support classifier
    # ------------------------------------------------------

    "foreground_prior_prob":
        0.05,

    "initial_logit_scale":
        5.0,


    # ------------------------------------------------------
    # Detection loss — locked for later steps
    # ------------------------------------------------------

    "focal_alpha":
        0.25,

    "focal_gamma":
        2.0,

    "loss_bbox_weight":
        5.0,

    "loss_giou_weight":
        2.0,


    # ------------------------------------------------------
    # Hungarian Matcher — locked
    # ------------------------------------------------------

    "matcher_class_cost":
        1.0,

    "matcher_bbox_cost":
        5.0,

    "matcher_giou_cost":
        2.0,


    # ------------------------------------------------------
    # Matched-query support ranking
    # ------------------------------------------------------

    "support_rank_margin":
        0.10,

    "support_rank_weight":
        1.0,
}


# ==========================================================
# BASIC CONFIG ASSERTIONS
# ==========================================================

assert CONFIG["hidden_dim"] % 2 == 0

assert (
    CONFIG["hidden_dim"]
    %
    CONFIG["num_heads"]
    ==
    0
)

assert (
    CONFIG["relation_output_dim"]
    ==
    CONFIG["hidden_dim"]
)


print("=" * 70)
print("STEP 3 : FINAL MODEL CONFIGURATION")
print("=" * 70)

print("Backbone        :", CONFIG["backbone"])
print("Input Size      :", CONFIG["image_size"])
print("Hidden Dim      :", CONFIG["hidden_dim"])
print("Object Queries  :", CONFIG["num_queries"])
print(
    "Encoder Layers  :",
    CONFIG["num_encoder_layers"]
)
print(
    "Decoder Layers  :",
    CONFIG["num_decoder_layers"]
)
print("Attention Heads :", CONFIG["num_heads"])
print("FFN Dim         :", CONFIG["dim_feedforward"])
print("Dropout         :", CONFIG["dropout"])

print("-" * 70)

print(
    "Position        :",
    CONFIG["position_embedding"]
)

print(
    "Foreground Prior:",
    CONFIG["foreground_prior_prob"]
)

print(
    "Initial Scale   :",
    CONFIG["initial_logit_scale"]
)

print("=" * 70)
