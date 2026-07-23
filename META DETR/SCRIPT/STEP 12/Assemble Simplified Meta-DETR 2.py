def forward(
    self,
    support_image,
    query_image=None
):

    # ==================================================
    # Base Training Mode
    # ==================================================
    if query_image is None:
        query_image = support_image

    # ==================================================
    # Support Encoder
    # ==================================================
    prototype = self.support_encoder(
        support_image
    )

    # ==================================================
    # Query Encoder
    # ==================================================
    query_feature = self.query_encoder(
        query_image
    )

    # ==================================================
    # Cross Attention
    # ==================================================
    attended = self.cross_attention(
        prototype,
        query_feature
    )

    # ==================================================
    # Transformer Encoder
    # ==================================================
    encoder_output = self.transformer_encoder(
        attended
    )

    # ==================================================
    # Transformer Decoder
    # ==================================================
    decoder_output = self.transformer_decoder(
        encoder_output
    )

    # ==================================================
    # Detection Head
    # ==================================================
    class_logits, boxes = self.detection_head(
        decoder_output
    )

    return class_logits, boxes
