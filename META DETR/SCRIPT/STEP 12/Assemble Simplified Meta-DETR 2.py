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
