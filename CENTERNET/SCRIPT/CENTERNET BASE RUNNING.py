# =========================================================
# INSTALL & IMPORT
# =========================================================

import torch
import torch.nn as nn
import torchvision.models as models


# =========================================================
# CEK GPU
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device yang digunakan:", device)


# =========================================================
# VANILLA TRANSFORMER ENCODER
# =========================================================

class VanillaTransformerEncoder(nn.Module):

    """
    Transformer encoder ringan untuk
    memperbaiki feature map hasil backbone.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # x = [B, C, H, W]

        B, C, H, W = x.shape

        # =====================================================
        # UBAH FEATURE MAP MENJADI SEQUENCE
        # =====================================================

        x = x.flatten(2).transpose(1, 2)

        # Shape:
        # [B, H*W, C]

        # =====================================================
        # TRANSFORMER ENCODING
        # =====================================================

        x = self.transformer(x)

        # =====================================================
        # KEMBALIKAN KE FEATURE MAP
        # =====================================================

        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


# =========================================================
# CONTRASTIVE EMBEDDING HEAD
# =========================================================

class ContrastiveEmbeddingHead(nn.Module):

    """
    Branch embedding untuk contrastive learning.
    """

    def __init__(self, in_channels=256, embedding_dim=128):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.embedding(x)


# =========================================================
# CENTERNET HEAD
# =========================================================

class CenterNetHead(nn.Module):

    """
    Head utama CenterNet:
    - Heatmap prediction
    - Width-height regression
    - Offset regression
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # =====================================================
        # HEATMAP HEAD
        # =====================================================

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        # =====================================================
        # WIDTH-HEIGHT HEAD
        # =====================================================

        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)
        )

        # =====================================================
        # OFFSET HEAD
        # =====================================================

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)
        )

    def forward(self, x):

        heatmap = self.heatmap_head(x)

        wh = self.wh_head(x)

        offset = self.offset_head(x)

        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset
        }


# =========================================================
# MODIFIED CENTERNET FSOD
# =========================================================

class ModifiedCenterNetFSOD(nn.Module):

    """
    Modified CenterNet untuk
    Few-Shot Object Detection.
    """

    def __init__(
        self,
        num_base_classes,
        num_novel_classes
    ):
        super().__init__()

        # =====================================================
        # BACKBONE RESNET101
        # =====================================================

        backbone = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        )

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        # =====================================================
        # FEATURE REDUCTION
        # =====================================================

        self.feature_reduce = nn.Conv2d(
            1024,
            256,
            kernel_size=1
        )

        # =====================================================
        # VANILLA TRANSFORMER
        # =====================================================

        self.transformer = VanillaTransformerEncoder(
            embed_dim=256,
            num_heads=4,
            ff_dim=512,
            num_layers=2
        )

        # =====================================================
        # CONTRASTIVE EMBEDDING
        # =====================================================

        self.embedding_head = ContrastiveEmbeddingHead(
            in_channels=256,
            embedding_dim=128
        )

        # =====================================================
        # BASE HEAD
        # =====================================================

        self.base_head = CenterNetHead(
            in_channels=256,
            num_classes=num_base_classes
        )

        # =====================================================
        # NOVEL HEAD
        # =====================================================

        self.novel_head = CenterNetHead(
            in_channels=256,
            num_classes=num_novel_classes
        )

    # =========================================================
    # FREEZE BASE HEAD
    # =========================================================

    def freeze_base_head(self):

        for param in self.base_head.parameters():
            param.requires_grad = False

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, x):

        # =====================================================
        # STEP 1 - BACKBONE
        # =====================================================

        features = self.backbone(x)

        print("Shape backbone output:", features.shape)

        # =====================================================
        # STEP 2 - FEATURE REDUCTION
        # =====================================================

        features = self.feature_reduce(features)

        print("Shape reduced feature:", features.shape)

        # =====================================================
        # STEP 3 - TRANSFORMER
        # =====================================================

        refined_features = self.transformer(features)

        print("Shape transformer output:", refined_features.shape)

        # =====================================================
        # STEP 4 - EMBEDDING
        # =====================================================

        embeddings = self.embedding_head(refined_features)

        print("Shape embedding:", embeddings.shape)

        # =====================================================
        # STEP 5 - BASE HEAD
        # =====================================================

        base_outputs = self.base_head(refined_features)

        # =====================================================
        # STEP 6 - NOVEL HEAD
        # =====================================================

        novel_outputs = self.novel_head(refined_features)

        return {
            "features": refined_features,
            "embeddings": embeddings,
            "base_outputs": base_outputs,
            "novel_outputs": novel_outputs
        }


# =========================================================
# MEMBUAT MODEL
# =========================================================

model = ModifiedCenterNetFSOD(
    num_base_classes=60,
    num_novel_classes=20
).to(device)

print("\nModel berhasil dibuat!\n")


# =========================================================
# DUMMY TENSOR
# =========================================================

"""
DUMMY TENSOR BUKAN DATASET ASLI.

Ini hanya gambar random palsu untuk mengetes:
- apakah model bisa jalan
- apakah transformer error
- apakah output shape benar
- apakah GPU kuat

Shape:
-------
[2, 3, 512, 512]

Artinya:
- batch size = 2 gambar
- RGB = 3 channel
- ukuran gambar = 512x512
"""

dummy_input = torch.randn(2, 3, 512, 512).to(device)

print("Dummy tensor berhasil dibuat!")
print("Shape dummy tensor:", dummy_input.shape)


# =========================================================
# FORWARD PASS TEST
# =========================================================

print("\nMulai forward pass...\n")

outputs = model(dummy_input)

print("\n====================================")
print("FORWARD PASS BERHASIL!")
print("====================================\n")


# =========================================================
# OUTPUT SHAPE
# =========================================================

print("Shape Embedding:")
print(outputs["embeddings"].shape)

print("\nShape Base Heatmap:")
print(outputs["base_outputs"]["heatmap"].shape)

print("\nShape Novel Heatmap:")
print(outputs["novel_outputs"]["heatmap"].shape)

print("\n====================================")
print("MODEL BERHASIL DIJALANKAN")
print("====================================")
