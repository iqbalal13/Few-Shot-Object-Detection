# ==========================================================
# META-DETR V3
# DIRECT PROTOTYPE RELATION CONDITIONING
#
# Goals:
# H1 -> remove easy support-bypass path
# H2 -> learn discriminative support prototype
#
# Keep unchanged:
# - ResNet101 backbone
# - Query Encoder
# - Position Encoding
# - Transformer Encoder
# - Detection Head
# - 1-way 1-shot episodic protocol
# - Hungarian / Focal / L1 / GIoU
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=" * 70)
print("META-DETR V3 : BUILDING")
print("=" * 70)


# ==========================================================
# V3 SUPPORT PROTOTYPE
#
# V2:
# LayerNorm every spatial token
# -> mean 400 tokens
#
# V3:
# Global pooling
# -> learnable projection
# -> L2-normalized prototype
# ==========================================================

class V3SupportEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.projector = nn.Sequential(

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            )
        )


    def forward(
        self,
        support_feature
    ):

        # support_feature:
        # [B,256,H,W]

        pooled = (
            self.pool(
                support_feature
            )
            .flatten(1)
        )

        # [B,256]

        prototype = (
            self.projector(
                pooled
            )
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=-1
        )

        return prototype


# ==========================================================
# V3 QUERY <-> PROTOTYPE RELATION CONDITIONER
#
# IMPORTANT:
# No:
#     query + attended_support
#
# Instead representation is explicitly constructed from:
#
#     query * prototype
#     |query - prototype|
#
# Therefore support is structurally involved.
# ==========================================================

class V3PrototypeRelationConditioner(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.query_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.prototype_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.LayerNorm(
                hidden_dim
            )
        )


    def forward(
        self,
        query_tokens,
        prototype
    ):

        # query_tokens:
        # [B, HW, 256]
        #
        # prototype:
        # [B,256]

        q = self.query_projection(
            query_tokens
        )

        p = self.prototype_projection(
            prototype
        )

        q = F.normalize(
            q,
            p=2,
            dim=-1
        )

        p = F.normalize(
            p,
            p=2,
            dim=-1
        )

        p = p.unsqueeze(1)

        # ----------------------------------------------
        # Explicit support-query relations
        # ----------------------------------------------

        product_relation = (
            q * p
        )

        distance_relation = (
            torch.abs(
                q - p
            )
        )

        relation = torch.cat(
            [
                product_relation,
                distance_relation
            ],
            dim=-1
        )

        guided_query = self.fusion(
            relation
        )

        return guided_query


# ==========================================================
# V3 TRANSFORMER DECODER
#
# Object queries ALSO receive prototype.
# ==========================================================

class V3TransformerDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG[
            "num_decoder_layers"
        ],
        num_queries=CONFIG["num_queries"],
        dim_feedforward=CONFIG[
            "dim_feedforward"
        ],
        dropout=CONFIG["dropout"]
    ):
        super().__init__()


        # ----------------------------------------------
        # Standard DETR learned object queries
        # ----------------------------------------------

        self.object_queries = nn.Embedding(
            num_queries,
            hidden_dim
        )


        # ----------------------------------------------
        # Prototype -> object-query conditioning
        # ----------------------------------------------

        self.prototype_to_query = (
            nn.Sequential(

                nn.Linear(
                    hidden_dim,
                    hidden_dim
                ),

                nn.GELU(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim
                )
            )
        )

        self.query_norm = nn.LayerNorm(
            hidden_dim
        )


        # ----------------------------------------------
        # Transformer decoder
        # ----------------------------------------------

        decoder_layer = (
            nn.TransformerDecoderLayer(

                d_model=hidden_dim,

                nhead=num_heads,

                dim_feedforward=
                    dim_feedforward,

                dropout=dropout,

                activation="relu",

                batch_first=True,

                norm_first=False
            )
        )

        self.decoder = nn.TransformerDecoder(

            decoder_layer=
                decoder_layer,

            num_layers=
                num_layers,

            norm=nn.LayerNorm(
                hidden_dim
            )
        )


    def forward(
        self,
        memory,
        prototype
    ):

        batch_size = (
            memory.shape[0]
        )

        base_queries = (
            self.object_queries
            .weight
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1
            )
        )


        # ----------------------------------------------
        # Direct prototype condition
        # ----------------------------------------------

        condition = (
            self.prototype_to_query(
                prototype
            )
            .unsqueeze(1)
        )


        # ----------------------------------------------
        # Multiplicative + additive conditioning
        # ----------------------------------------------

        gate = (
            1.0
            +
            torch.tanh(
                condition
            )
        )

        conditioned_queries = (
            base_queries
            *
            gate
            +
            condition
        )

        conditioned_queries = (
            self.query_norm(
                conditioned_queries
            )
        )


        output = self.decoder(

            tgt=
                conditioned_queries,

            memory=
                memory
        )

        return output


# ==========================================================
# V3 FULL MODEL
# ==========================================================

class SimplifiedMetaDETRV3(
    nn.Module
):

    def __init__(
        self,
        backbone,
        support_encoder,
        query_encoder,
        prototype_conditioner,
        position_encoding,
        transformer_encoder,
        transformer_decoder,
        detection_head
    ):
        super().__init__()

        self.backbone = backbone

        self.support_encoder = (
            support_encoder
        )

        self.query_encoder = (
            query_encoder
        )

        self.prototype_conditioner = (
            prototype_conditioner
        )

        self.position_encoding = (
            position_encoding
        )

        self.transformer_encoder = (
            transformer_encoder
        )

        self.transformer_decoder = (
            transformer_decoder
        )

        self.detection_head = (
            detection_head
        )


    def forward(
        self,
        support_image,
        query_image
    ):

        # ==============================================
        # Shared backbone
        # ==============================================

        support_feature_map = (
            self.backbone(
                support_image
            )
        )

        query_feature_map = (
            self.backbone(
                query_image
            )
        )


        # ==============================================
        # Support prototype
        # ==============================================

        prototype = (
            self.support_encoder(
                support_feature_map
            )
        )


        # ==============================================
        # Query spatial tokens
        # ==============================================

        (
            query_tokens,
            spatial_shape
        ) = self.query_encoder(
            query_feature_map
        )


        # ==============================================
        # Support/query batch compatibility
        # ==============================================

        if (
            prototype.shape[0] == 1
            and
            query_tokens.shape[0] > 1
        ):

            prototype = prototype.expand(
                query_tokens.shape[0],
                -1
            )


        if (
            prototype.shape[0]
            !=
            query_tokens.shape[0]
        ):

            raise ValueError(
                "Prototype/query batch mismatch: "
                f"{prototype.shape[0]} vs "
                f"{query_tokens.shape[0]}"
            )


        # ==============================================
        # V3 DIRECT RELATION CONDITIONING
        # ==============================================

        guided_query = (
            self.prototype_conditioner(

                query_tokens=
                    query_tokens,

                prototype=
                    prototype
            )
        )


        # ==============================================
        # Positional encoding
        # ==============================================

        query_position = (
            self.position_encoding(
                query_feature_map
            )
        )

        assert (
            query_position.shape
            ==
            guided_query.shape
        )


        # ==============================================
        # Transformer encoder
        # ==============================================

        encoder_input = (
            guided_query
            +
            query_position
        )

        memory = (
            self.transformer_encoder(
                encoder_input
            )
        )


        # ==============================================
        # Prototype-conditioned decoder
        # ==============================================

        decoder_output = (
            self.transformer_decoder(

                memory=
                    memory,

                prototype=
                    prototype
            )
        )


        # ==============================================
        # Detection head
        # ==============================================

        (
            pred_logits,
            pred_boxes
        ) = self.detection_head(
            decoder_output
        )


        return {

            "pred_logits":
                pred_logits,

            "pred_boxes":
                pred_boxes
        }


# ==========================================================
# BUILD V3 COMPONENTS
# ==========================================================

support_encoder = (
    V3SupportEncoder(

        hidden_dim=
            CONFIG["hidden_dim"],

        dropout=
            CONFIG["dropout"]
    )
    .to(
        CONFIG["device"]
    )
)


prototype_conditioner = (
    V3PrototypeRelationConditioner(

        hidden_dim=
            CONFIG["hidden_dim"],

        dropout=
            CONFIG["dropout"]
    )
    .to(
        CONFIG["device"]
    )
)


transformer_decoder = (
    V3TransformerDecoder()
    .to(
        CONFIG["device"]
    )
)


# ==========================================================
# UPDATE CONFIG LABEL
# ==========================================================

CONFIG[
    "support_conditioning"
] = "prototype_relation"


# ==========================================================
# BUILD MODEL
# ==========================================================

model = SimplifiedMetaDETRV3(

    backbone=
        backbone,

    support_encoder=
        support_encoder,

    query_encoder=
        query_encoder,

    prototype_conditioner=
        prototype_conditioner,

    position_encoding=
        position_encoding,

    transformer_encoder=
        transformer_encoder,

    transformer_decoder=
        transformer_decoder,

    detection_head=
        detection_head

).to(
    CONFIG["device"]
)


print("=" * 70)
print("META-DETR V3 READY")
print("=" * 70)

print(
    "Model        :",
    model.__class__.__name__
)

print(
    "Conditioning :",
    CONFIG[
        "support_conditioning"
    ]
)

print("=" * 70)
