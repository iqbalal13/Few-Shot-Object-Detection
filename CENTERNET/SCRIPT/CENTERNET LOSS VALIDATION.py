# =========================================================
# MODIFIED CENTERNET FSOD
# FULL FORWARD + LOSS VALIDATION
# =========================================================

"""
TUJUAN CODING INI:
------------------
1. Menjalankan Modified CenterNet
2. Menggunakan dummy tensor
3. Menghitung:
   - heatmap loss
   - width-height loss
   - offset loss
4. Mengecek apakah model bisa belajar
5. Mengecek backward pass berhasil

CATATAN:
---------
INI BELUM TRAINING DATASET ASLI.

Masih tahap:
- validasi arsitektur
- validasi loss
- validasi training pipeline awal
"""

# =========================================================
# IMPORT
# =========================================================

import torch
import torch.nn as nn
import torchvision.models as models


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device yang digunakan:", device)


# =========================================================
# VANILLA TRANSFORMER ENCODER
# =========================================================

class VanillaTransformerEncoder(nn.Module):

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

        B, C, H, W = x.shape

        # convert ke sequence
        x = x.flatten(2).transpose(1, 2)

        # transformer
        x = self.transformer(x)

        # kembali ke feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


# =========================================================
# CONTRASTIVE EMBEDDING HEAD
# =========================================================

class ContrastiveEmbeddingHead(nn.Module):

    def __init__(self, in_channels=256, embedding_dim=128):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.embedding(x)


# =========================================================
# CENTERNET HEAD
# =========================================================

class CenterNetHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # heatmap
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        # width-height
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)
        )

        # offset
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

    def __init__(
        self,
        num_base_classes,
        num_novel_classes
    ):
        super().__init__()

        # =====================================================
        # RESNET101 BACKBONE
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
        # TRANSFORMER
        # =====================================================

        self.transformer = VanillaTransformerEncoder(
            embed_dim=256,
            num_heads=4,
            ff_dim=512,
            num_layers=2
        )

        # =====================================================
        # EMBEDDING
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

    def forward(self, x):

        # backbone
        features = self.backbone(x)

        # reduce channel
        features = self.feature_reduce(features)

        # transformer refinement
        refined_features = self.transformer(features)

        # embedding
        embeddings = self.embedding_head(refined_features)

        # base prediction
        base_outputs = self.base_head(refined_features)

        # novel prediction
        novel_outputs = self.novel_head(refined_features)

        return {
            "features": refined_features,
            "embeddings": embeddings,
            "base_outputs": base_outputs,
            "novel_outputs": novel_outputs
        }


# =========================================================
# HEATMAP FOCAL LOSS
# =========================================================

class HeatmapFocalLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pos_inds = target.eq(1).float()

        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = (
            torch.log(pred + 1e-6)
            * torch.pow(1 - pred, 2)
            * pos_inds
        )

        neg_loss = (
            torch.log(1 - pred + 1e-6)
            * torch.pow(pred, 2)
            * neg_weights
            * neg_inds
        )

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()

        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


# =========================================================
# MODEL
# =========================================================

model = ModifiedCenterNetFSOD(
    num_base_classes=60,
    num_novel_classes=20
).to(device)

print("\nModel berhasil dibuat!\n")


# =========================================================
# DUMMY INPUT
# =========================================================

dummy_input = torch.randn(2, 3, 512, 512).to(device)

print("Dummy tensor berhasil dibuat!")
print("Shape input:", dummy_input.shape)


# =========================================================
# FORWARD PASS
# =========================================================

outputs = model(dummy_input)

print("\nForward pass berhasil!\n")


# =========================================================
# DUMMY TARGET
# =========================================================

"""
TARGET PALSU UNTUK TEST LOSS

Belum menggunakan dataset asli.
"""

# heatmap target
heatmap_target = torch.zeros(2, 60, 32, 32).to(device)

# kasih center object palsu
heatmap_target[:, :, 16, 16] = 1

# width-height target
wh_target = torch.randn(2, 2, 32, 32).to(device)

# offset target
offset_target = torch.randn(2, 2, 32, 32).to(device)

print("Dummy target berhasil dibuat!\n")


# =========================================================
# LOSS FUNCTION
# =========================================================

heatmap_loss_fn = HeatmapFocalLoss()

wh_loss_fn = nn.L1Loss()

offset_loss_fn = nn.L1Loss()


# =========================================================
# HITUNG LOSS
# =========================================================

heatmap_loss = heatmap_loss_fn(
    outputs["base_outputs"]["heatmap"],
    heatmap_target
)

wh_loss = wh_loss_fn(
    outputs["base_outputs"]["wh"],
    wh_target
)

offset_loss = offset_loss_fn(
    outputs["base_outputs"]["offset"],
    offset_target
)

total_loss = (
    heatmap_loss
    + wh_loss
    + offset_loss
)

# =========================================================
# PRINT LOSS
# =========================================================

print("====================================")
print("HASIL LOSS VALIDATION")
print("====================================\n")

print("Heatmap Loss:", heatmap_loss.item())

print("WH Loss:", wh_loss.item())

print("Offset Loss:", offset_loss.item())

print("Total Loss:", total_loss.item())


# =========================================================
# BACKWARD PASS
# =========================================================

total_loss.backward()

print("\n====================================")
print("BACKWARD PASS BERHASIL!")
print("MODEL SUDAH BISA BELAJAR")
print("====================================")
