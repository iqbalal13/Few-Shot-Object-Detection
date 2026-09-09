# ==========================================================
# STEP 9 : Support-Object Relation Module
#
# D = decoder objects
# P = support prototype
#
# D' = normalize(Wd(D))
# P' = normalize(Wp(P))
#
# R = [
#     D',
#     P',
#     D' * P',
#     abs(D' - P')
# ]
#
# R -> relation MLP -> Z
#
# NO raw decoder-output bypass.
# ==========================================================


class SupportObjectRelationModule(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=
            CONFIG[
                "hidden_dim"
            ],
        relation_hidden_dim=
            CONFIG[
                "relation_hidden_dim"
            ],
        dropout=
            CONFIG[
                "dropout"
            ]
    ):
        super().__init__()


        self.decoder_projection = (
            nn.Linear(

                hidden_dim,

                hidden_dim,

                bias=False
            )
        )


        self.support_projection = (
            nn.Linear(

                hidden_dim,

                hidden_dim,

                bias=False
            )
        )


        relation_input_dim = (
            hidden_dim * 4
        )


        self.relation_mlp = (
            nn.Sequential(

                nn.Linear(
                    relation_input_dim,
                    relation_hidden_dim
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    relation_hidden_dim,
                    hidden_dim
                ),

                nn.LayerNorm(
                    hidden_dim
                )
            )
        )


    def forward(
        self,
        decoder_objects,
        support_prototype
    ):

        # decoder_objects:
        # [B,Q,D]
        #
        # support_prototype:
        # [B,D]


        if decoder_objects.dim() != 3:
            raise ValueError(
                "decoder_objects must be [B,Q,D]"
            )


        if support_prototype.dim() != 2:
            raise ValueError(
                "support_prototype must be [B,D]"
            )


        if (
            support_prototype.shape[0]
            !=
            decoder_objects.shape[0]
        ):
            raise ValueError(
                "Support/query batch mismatch."
            )


        d_metric = F.normalize(

            self.decoder_projection(
                decoder_objects
            ),

            p=2,

            dim=-1
        )


        p_metric = F.normalize(

            self.support_projection(
                support_prototype
            ),

            p=2,

            dim=-1
        )


        p_expand = (

            p_metric
            .unsqueeze(1)

            .expand(
                -1,
                decoder_objects.shape[1],
                -1
            )
        )


        product = (
            d_metric
            *
            p_expand
        )


        difference = (

            d_metric
            -
            p_expand

        ).abs()


        relation_input = torch.cat(

            (
                d_metric,
                p_expand,
                product,
                difference
            ),

            dim=-1
        )


        relation_features = (

            self.relation_mlp(
                relation_input
            )
        )


        return relation_features


print("=" * 70)
print("STEP 9 : SUPPORT-OBJECT RELATION MODULE DEFINED")
print("=" * 70)
