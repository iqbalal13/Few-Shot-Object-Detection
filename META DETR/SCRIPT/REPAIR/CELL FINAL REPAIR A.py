# ==========================================================
# CELL FINAL REPAIR A
# EXPLICIT SUPPORT-DEPENDENT CLASSIFICATION
#
# FINAL REPAIR CANDIDATE
#
# LOCKED DIAGNOSIS:
# H1 -> detector output almost ignores support identity
# H2 -> representation improved in V3.2
# H3 -> coarse localization improved, precise localization
#       still weak
#
# REPAIR:
# foreground logit MUST explicitly use:
#
# decoder object representation <-> support prototype
#
# No decoder-only classification path.
#
# Bounding-box mechanism remains unchanged.
#
# IMPORTANT:
# - starts from untouched official V3 `model`
# - DOES NOT continue from v32_model
# - does NOT overwrite `model`
# ==========================================================

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=" * 70)
print("CELL FINAL REPAIR A : BUILD FINAL MODEL")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert (
    "model" in globals()
), (
    "Official V3 `model` is missing."
)

assert (
    model.__class__.__name__
    ==
    "SimplifiedMetaDETRV3"
), (
    "Expected SimplifiedMetaDETRV3 as source model."
)


# ==========================================================
# EXPLICIT SUPPORT-MATCH HEAD
#
# Classification:
#
# decoder output
#      ↓
# trainable projection
#      ↓
# L2 normalize
#      ×
# support prototype
#      ↓
# L2 normalize
#      ↓
# cosine similarity
#      ↓
# fixed scale + learnable bias
#      ↓
# foreground logit
#
# CRITICAL:
# There is NO Linear(decoder_output)->class logit bypass.
# ==========================================================

class ExplicitSupportMatchHead(nn.Module):

    def __init__(
        self,
        box_head,
        hidden_dim=CONFIG["hidden_dim"],
        prior_prob=CONFIG[
            "classification_prior_prob"
        ],
        support_scale=5.0
    ):
        super().__init__()


        # --------------------------------------------------
        # Maps decoder objects into support-prototype space
        # --------------------------------------------------

        self.query_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )


        # Start as identity.
        with torch.no_grad():

            nn.init.eye_(
                self.query_projection.weight
            )


        # --------------------------------------------------
        # Fixed scale.
        #
        # Intentionally NOT learnable so training cannot
        # collapse support dependence by driving scale -> 0.
        # --------------------------------------------------

        self.register_buffer(

            "support_scale",

            torch.tensor(
                float(support_scale),
                dtype=torch.float32
            )
        )


        # --------------------------------------------------
        # Same foreground prior concept as previous head
        # --------------------------------------------------

        prior_bias = -math.log(

            (1.0 - prior_prob)
            /
            prior_prob
        )


        self.class_bias = nn.Parameter(

            torch.tensor(
                [prior_bias],
                dtype=torch.float32
            )
        )


        # --------------------------------------------------
        # SAME bbox head.
        # Localization mechanism is not replaced here.
        # --------------------------------------------------

        self.box_head = box_head


    def forward(
        self,
        decoder_output,
        prototype
    ):

        # decoder_output:
        # [B, Q, D]
        #
        # prototype:
        # [B, D]


        # --------------------------------------------------
        # Query/object representation
        # --------------------------------------------------

        query_metric = (

            self.query_projection(
                decoder_output
            )
        )


        query_metric = F.normalize(

            query_metric,

            p=2,

            dim=-1
        )


        # --------------------------------------------------
        # IMPORTANT:
        # use support prototype DIRECTLY.
        #
        # No learnable support projection that could collapse
        # all support classes into the same representation.
        # --------------------------------------------------

        support_metric = F.normalize(

            prototype,

            p=2,

            dim=-1
        )


        support_metric = (

            support_metric
            .unsqueeze(1)
        )


        # --------------------------------------------------
        # Explicit cosine support/object relation
        #
        # [B,Q,D] × [B,1,D]
        # -> [B,Q,1]
        # --------------------------------------------------

        support_similarity = (

            query_metric
            *
            support_metric
        ).sum(

            dim=-1,

            keepdim=True
        )


        # --------------------------------------------------
        # FOREGROUND LOGIT
        #
        # Cannot be calculated without support prototype.
        # --------------------------------------------------

        pred_logits = (

            self.support_scale
            *
            support_similarity
            +
            self.class_bias
        )


        # --------------------------------------------------
        # Class-agnostic box regression unchanged
        # --------------------------------------------------

        pred_boxes = torch.sigmoid(

            self.box_head(
                decoder_output
            )
        )


        return (
            pred_logits,
            pred_boxes
        )


# ==========================================================
# FINAL META-DETR MODEL
#
# Same V3 pipeline until detection head.
# ==========================================================

class SimplifiedMetaDETRFinal(nn.Module):

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

        # ==================================================
        # Shared backbone
        # ==================================================

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


        # ==================================================
        # Support prototype
        # ==================================================

        prototype = (

            self.support_encoder(
                support_feature_map
            )
        )


        # ==================================================
        # Query spatial representation
        # ==================================================

        (
            query_tokens,
            spatial_shape
        ) = (

            self.query_encoder(
                query_feature_map
            )
        )


        # ==================================================
        # Batch compatibility
        # ==================================================

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


        # ==================================================
        # SAME V3 prototype relation conditioning
        # ==================================================

        guided_query = (

            self.prototype_conditioner(

                query_tokens=
                    query_tokens,

                prototype=
                    prototype
            )
        )


        # ==================================================
        # Positional encoding
        # ==================================================

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


        # ==================================================
        # Transformer encoder
        # ==================================================

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


        # ==================================================
        # SAME V3 prototype-conditioned decoder
        # ==================================================

        decoder_output = (

            self.transformer_decoder(

                memory=
                    memory,

                prototype=
                    prototype
            )
        )


        # ==================================================
        # FINAL EXPLICIT SUPPORT-MATCH HEAD
        # ==================================================

        (
            pred_logits,
            pred_boxes
        ) = (

            self.detection_head(

                decoder_output,

                prototype
            )
        )


        return {

            "pred_logits":
                pred_logits,

            "pred_boxes":
                pred_boxes
        }


# ==========================================================
# BUILD FINAL MODEL
#
# Fresh copy of official untouched V3.
# NOT v32_model.
# ==========================================================

final_detection_head = (

    ExplicitSupportMatchHead(

        box_head=
            copy.deepcopy(
                model
                .detection_head
                .box_head
            ),

        hidden_dim=
            CONFIG["hidden_dim"],

        prior_prob=
            CONFIG[
                "classification_prior_prob"
            ],

        support_scale=
            5.0
    )
)


final_model = SimplifiedMetaDETRFinal(

    backbone=
        copy.deepcopy(
            model.backbone
        ),

    support_encoder=
        copy.deepcopy(
            model.support_encoder
        ),

    query_encoder=
        copy.deepcopy(
            model.query_encoder
        ),

    prototype_conditioner=
        copy.deepcopy(
            model.prototype_conditioner
        ),

    position_encoding=
        copy.deepcopy(
            model.position_encoding
        ),

    transformer_encoder=
        copy.deepcopy(
            model.transformer_encoder
        ),

    transformer_decoder=
        copy.deepcopy(
            model.transformer_decoder
        ),

    detection_head=
        final_detection_head

).to(
    CONFIG["device"]
)


final_model.zero_grad(
    set_to_none=True
)


# ==========================================================
# STRUCTURAL CHECKS
# ==========================================================

assert (
    not hasattr(
        final_model.detection_head,
        "class_head"
    )
), (
    "Decoder-only class_head still exists."
)


assert (
    hasattr(
        final_model.detection_head,
        "query_projection"
    )
)


# ==========================================================
# PARAMETER SUMMARY
# ==========================================================

total_params = sum(

    p.numel()

    for p in final_model.parameters()
)


trainable_params = sum(

    p.numel()

    for p in final_model.parameters()

    if p.requires_grad
)


print("-" * 70)

print(
    "Model               :",
    final_model.__class__.__name__
)

print(
    "Classification      : explicit support cosine match"
)

print(
    "Decoder-only bypass : NONE"
)

print(
    "Support projection  : DIRECT prototype"
)

print(
    "Support scale       :",
    float(
        final_model
        .detection_head
        .support_scale
        .item()
    )
)

print(
    "BBox head           : unchanged"
)

print(
    "Starts from         : official untouched V3"
)

print(
    "Continues V3.2      : NO"
)

print(
    "Total Parameters    :",
    f"{total_params:,}"
)

print(
    "Trainable Parameters:",
    f"{trainable_params:,}"
)

print("=" * 70)
print("✓ FINAL REPAIR MODEL READY")
print("=" * 70)
