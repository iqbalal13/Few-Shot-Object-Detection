model = SimplifiedMetaDETR(

    backbone,

    prototype_extractor,

    support_encoder,

    query_encoder,

    cross_attention,

    transformer_encoder,

    transformer_decoder,

    detection_head

).to(CONFIG["device"])
