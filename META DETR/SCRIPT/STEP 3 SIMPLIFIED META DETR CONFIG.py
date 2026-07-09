# ==========================================================
# STEP 3 : Simplified Meta-DETR Configuration
# ==========================================================

CONFIG = {

    # Backbone
    "backbone": "resnet101",

    # Transformer
    "hidden_dim": 256,
    "num_queries": 100,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,

    # Dataset
    "num_classes": 1,

    # Image
    "image_size": 800,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"

}

print("="*60)
print("Simplified Meta-DETR Configuration")
print("="*60)

for k,v in CONFIG.items():
    print(f"{k:25} : {v}")
