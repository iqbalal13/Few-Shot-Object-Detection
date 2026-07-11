# ==========================================================
# STEP 12 : Assemble Simplified Meta-DETR
# ==========================================================

class SimplifiedMetaDETR(nn.Module):

    def __init__(
        self,
        backbone,
        prototype_extractor,
        support_encoder,
        query_encoder,
        cross_attention,
        transformer_encoder,
        transformer_decoder,
        detection_head
    ):
        super().__init__()

        self.backbone = backbone
        self.prototype_extractor = prototype_extractor
        self.support_encoder = support_encoder
        self.query_encoder = query_encoder
        self.cross_attention = cross_attention
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.detection_head = detection_head

    def forward(
        self,
        support_image,
        query_image
    ):

        prototype = self.support_encoder(
            support_image
        )

        query_feature = self.query_encoder(
            query_image
        )

        attended = self.cross_attention(
            prototype,
            query_feature
        )

        encoder_output = self.transformer_encoder(
            attended
        )

        decoder_output = self.transformer_decoder(
            encoder_output
        )

        class_logits, boxes = self.detection_head(
            decoder_output
        )

        return class_logits, boxes
