# ==========================================================
# STEP 11 : Final Simplified Meta-DETR Model
#
# CLEAN FINAL ARCHITECTURE
# ==========================================================


class SimplifiedMetaDETRFinalClean(
    nn.Module
):

    def __init__(
        self,
        backbone,
        support_encoder,
        query_encoder,
        transformer_encoder,
        transformer_decoder,
        relation_module,
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

        self.transformer_encoder = (
            transformer_encoder
        )

        self.transformer_decoder = (
            transformer_decoder
        )

        self.relation_module = (
            relation_module
        )

        self.detection_head = (
            detection_head
        )


    # ======================================================
    # SUPPORT BRANCH
    # ======================================================

    def encode_support(
        self,
        support_image
    ):

        support_feature_map = (

            self.backbone(
                support_image
            )
        )


        prototype = (

            self.support_encoder(
                support_feature_map
            )
        )


        return prototype


    # ======================================================
    # QUERY BRANCH
    #
    # IMPORTANT:
    # no support enters this branch.
    # ======================================================

    def encode_query(
        self,
        query_image,
        query_padding_mask=None
    ):

        query_feature_map = (

            self.backbone(
                query_image
            )
        )


        (
            query_tokens,
            query_pos,
            mask_flat,
            spatial_shape

        ) = (

            self.query_encoder(

                query_feature_map,

                padding_mask=
                    query_padding_mask
            )
        )


        memory = (

            self.transformer_encoder(

                src=
                    query_tokens,

                pos=
                    query_pos,

                key_padding_mask=
                    mask_flat
            )
        )


        decoder_objects = (

            self.transformer_decoder(

                memory=
                    memory,

                memory_pos=
                    query_pos,

                memory_key_padding_mask=
                    mask_flat
            )
        )


        return (

            decoder_objects,

            {
                "memory":
                    memory,

                "query_position":
                    query_pos,

                "padding_mask":
                    mask_flat,

                "spatial_shape":
                    spatial_shape
            }
        )


    # ======================================================
    # SUPPORT CONDITIONING + PREDICTION
    # ======================================================

    def condition_and_predict(
        self,
        decoder_objects,
        support_prototype
    ):

        B = decoder_objects.shape[0]


        # Allow one prototype to condition a larger
        # query batch if ever needed.

        if (
            support_prototype.shape[0]
            ==
            1

            and

            B > 1
        ):

            support_prototype = (

                support_prototype.expand(
                    B,
                    -1
                )
            )


        if (
            support_prototype.shape[0]
            !=
            B
        ):

            raise ValueError(
                "Support/query batch mismatch."
            )


        relation_features = (

            self.relation_module(

                decoder_objects,

                support_prototype
            )
        )


        (
            pred_logits,
            pred_boxes,
            support_similarity

        ) = (

            self.detection_head(

                relation_features,

                support_prototype
            )
        )


        outputs = {

            "pred_logits":
                pred_logits,

            "pred_boxes":
                pred_boxes
        }


        extras = {

            "decoder_objects":
                decoder_objects,

            "support_prototype":
                support_prototype,

            "relation_features":
                relation_features,

            "support_similarity":
                support_similarity
        }


        return (
            outputs,
            extras
        )


    # ======================================================
    # FEATURE-AWARE FORWARD
    #
    # Later used by support-ranking loss.
    # ======================================================

    def forward_with_features(
        self,
        support_image,
        query_image,
        query_padding_mask=None
    ):

        support_prototype = (

            self.encode_support(
                support_image
            )
        )


        (
            decoder_objects,
            query_extras

        ) = (

            self.encode_query(

                query_image,

                query_padding_mask=
                    query_padding_mask
            )
        )


        (
            outputs,
            relation_extras

        ) = (

            self.condition_and_predict(

                decoder_objects,

                support_prototype
            )
        )


        extras = {

            **query_extras,

            **relation_extras
        }


        return (
            outputs,
            extras
        )


    # ======================================================
    # STANDARD FORWARD
    # ======================================================

    def forward(
        self,
        support_image,
        query_image,
        query_padding_mask=None
    ):

        outputs, _ = (

            self.forward_with_features(

                support_image=
                    support_image,

                query_image=
                    query_image,

                query_padding_mask=
                    query_padding_mask
            )
        )


        return outputs


# ==========================================================
# BUILD MODEL
# ==========================================================

backbone = (
    SharedResNet101Backbone()
)


support_encoder = (
    SupportPrototypeEncoder()
)


query_encoder = (
    QueryFeatureEncoder()
)


transformer_encoder = (
    DETRTransformerEncoder()
)


transformer_decoder = (
    DETRObjectDecoder()
)


relation_module = (
    SupportObjectRelationModule()
)


detection_head = (
    JointSupportConditionedDetectionHead()
)


model = SimplifiedMetaDETRFinalClean(

    backbone=
        backbone,

    support_encoder=
        support_encoder,

    query_encoder=
        query_encoder,

    transformer_encoder=
        transformer_encoder,

    transformer_decoder=
        transformer_decoder,

    relation_module=
        relation_module,

    detection_head=
        detection_head

).to(
    CONFIG["device"]
)


# ==========================================================
# PARAMETER SUMMARY
# ==========================================================

total_params = sum(

    p.numel()

    for p in model.parameters()
)


trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)


print("=" * 70)
print("STEP 11 : FINAL CLEAN MODEL READY")
print("=" * 70)

print(
    "Model      :",
    model.__class__.__name__
)

print(
    "Parameters :",
    f"{total_params:,}"
)

print(
    "Trainable  :",
    f"{trainable_params:,}"
)

print(
    "Device     :",
    CONFIG["device"]
)

print("=" * 70)
