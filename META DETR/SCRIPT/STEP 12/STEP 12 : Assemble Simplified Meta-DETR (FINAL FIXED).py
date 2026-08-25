# ==========================================================
# STEP 12 : Assemble Simplified Meta-DETR (FINAL FIXED)
# ==========================================================

import torch
import torch.nn as nn


class SimplifiedMetaDETR(nn.Module):

    def __init__(
        self,
        backbone,
        support_encoder,
        query_encoder,
        cross_attention,
        position_encoding,
        transformer_encoder,
        transformer_decoder,
        detection_head
    ):
        super().__init__()

        # ==================================================
        # ONE Shared Backbone
        # ==================================================

        self.backbone = backbone

        # ==================================================
        # Meta-Learning Components
        # ==================================================

        self.support_encoder = support_encoder
        self.query_encoder = query_encoder
        self.cross_attention = cross_attention

        # ==================================================
        # DETR Components
        # ==================================================

        self.position_encoding = position_encoding
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.detection_head = detection_head


    def forward(
        self,
        support_image,
        query_image
    ):

        # ==================================================
        # Shared ResNet-101 Backbone
        #
        # support_feature_map:
        # [Bs, 256, H, W]
        #
        # query_feature_map:
        # [Bq, 256, H, W]
        # ==================================================

        support_feature_map = self.backbone(
            support_image
        )

        query_feature_map = self.backbone(
            query_image
        )


        # ==================================================
        # Support Encoder
        #
        # support_tokens:
        # [Bs, Hs*Ws, 256]
        #
        # prototype:
        # [Bs, 256]
        # ==================================================

        support_tokens, prototype = self.support_encoder(
            support_feature_map
        )


        # ==================================================
        # Query Encoder
        #
        # query_tokens:
        # [Bq, Hq*Wq, 256]
        #
        # spatial_shape:
        # (Hq, Wq)
        # ==================================================

        query_tokens, spatial_shape = self.query_encoder(
            query_feature_map
        )


        # ==================================================
        # Support Batch Expansion
        #
        # Useful later when one support example conditions
        # multiple query images.
        # ==================================================

        if (
            support_tokens.shape[0] == 1
            and query_tokens.shape[0] > 1
        ):

            support_tokens = support_tokens.expand(
                query_tokens.shape[0],
                -1,
                -1
            )

            prototype = prototype.expand(
                query_tokens.shape[0],
                -1
            )


        # ==================================================
        # Batch Consistency
        # ==================================================

        if (
            support_tokens.shape[0]
            != query_tokens.shape[0]
        ):

            raise ValueError(
                "Support and query batch dimensions are incompatible: "
                f"support={support_tokens.shape[0]}, "
                f"query={query_tokens.shape[0]}"
            )


        if (
            prototype.shape[0]
            != query_tokens.shape[0]
        ):

            raise ValueError(
                "Prototype and query batch dimensions "
                "are incompatible."
            )


        # ==================================================
        # Query Token Sanity Check
        # ==================================================

        expected_query_tokens = (
            spatial_shape[0]
            *
            spatial_shape[1]
        )

        if (
            query_tokens.shape[1]
            != expected_query_tokens
        ):

            raise RuntimeError(
                "Query token count does not match "
                "query spatial shape."
            )


        # ==================================================
        # Prototype-Guided Cross Attention
        #
        # Q   = query_tokens
        #
        # K,V = support spatial tokens
        #       + global prototype
        #
        # Output:
        # [B, Hq*Wq, 256]
        # ==================================================

        guided_query = self.cross_attention(

            query_tokens=query_tokens,

            support_tokens=support_tokens,

            prototype=prototype
        )


        # ==================================================
        # 2D Positional Encoding
        #
        # IMPORTANT:
        # positional encoding receives FEATURE MAP,
        # not flattened query tokens.
        # ==================================================

        query_position = self.position_encoding(
            query_feature_map
        )


        if (
            query_position.shape
            != guided_query.shape
        ):

            raise RuntimeError(
                "Position encoding shape does not match guided query: "
                f"position={query_position.shape}, "
                f"guided={guided_query.shape}"
            )


        # ==================================================
        # Transformer Encoder Input
        # ==================================================

        encoder_input = (
            guided_query
            +
            query_position
        )


        # ==================================================
        # Transformer Encoder
        # ==================================================

        memory = self.transformer_encoder(
            encoder_input
        )


        # ==================================================
        # Transformer Decoder
        # +
        # Learnable Object Queries
        # ==================================================

        decoder_output = self.transformer_decoder(
            memory
        )


        # ==================================================
        # Detection Head
        # ==================================================

        pred_logits, pred_boxes = self.detection_head(
            decoder_output
        )


        # ==================================================
        # DETR-style Output
        # ==================================================

        return {

            "pred_logits":
                pred_logits,

            "pred_boxes":
                pred_boxes
        }


# ==========================================================
# Build Final Simplified Meta-DETR
# ==========================================================

model = SimplifiedMetaDETR(

    backbone=backbone,

    support_encoder=support_encoder,

    query_encoder=query_encoder,

    cross_attention=cross_attention,

    position_encoding=position_encoding,

    transformer_encoder=transformer_encoder,

    transformer_decoder=transformer_decoder,

    detection_head=detection_head

).to(
    CONFIG["device"]
)


print("=" * 70)
print("STEP 12 : Simplified Meta-DETR Assembled")
print("=" * 70)

print(
    "Model :",
    model.__class__.__name__
)

print(
    "Device:",
    CONFIG["device"]
)

print("=" * 70)
