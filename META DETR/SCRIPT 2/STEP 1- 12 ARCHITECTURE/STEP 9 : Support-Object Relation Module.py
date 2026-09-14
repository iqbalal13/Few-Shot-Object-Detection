# ==========================================================
# STEP 9 : Residual Support-Object Relation Module
#
# D = decoder object representation
# P = support prototype
#
# relation_delta = f(D, P)
#
# Z = LayerNorm(
#       D + tanh(relation_delta)
#     )
#
# BOTH classification and bbox use Z.
# ==========================================================

class SupportObjectRelationModule(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=CONFIG[
            "hidden_dim"
        ],
        relation_hidden_dim=CONFIG[
            "relation_hidden_dim"
        ],
        dropout=CONFIG[
            "dropout"
        ],
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

        self.relation_mlp = nn.Sequential(

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
        )

        self.fusion_norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )

    def forward(
        self,
        decoder_objects,
        support_prototype
    ):

        if decoder_objects.dim() != 3:
            raise ValueError(
                "decoder_objects must be "
                "[B,Q,D]."
            )

        if support_prototype.dim() != 2:
            raise ValueError(
                "support_prototype must be "
                "[B,D]."
            )

        if (
            decoder_objects.shape[0]
            !=
            support_prototype.shape[0]
        ):
            raise ValueError(
                "Support/query batch mismatch."
            )

        if (
            decoder_objects.shape[-1]
            !=
            support_prototype.shape[-1]
        ):
            raise ValueError(
                "Support/decoder hidden "
                "dimension mismatch."
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
                difference,
            ),
            dim=-1
        )

        relation_delta = torch.tanh(
            self.relation_mlp(
                relation_input
            )
        )

        relation_features = (
            self.fusion_norm(
                decoder_objects
                +
                relation_delta
            )
        )

        return relation_features


print("=" * 70)
print("STEP 9 : RESIDUAL SUPPORT-OBJECT RELATION DEFINED")
print("=" * 70)

print(
    "Z = LayerNorm("
    "D + tanh(Relation(D,P)))"
)
