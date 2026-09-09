# ==========================================================
# STEP 10 : Joint Support-Conditioned Detection Head
#
# Z -> explicit support classifier
# Z -> bbox regression
#
# NO decoder-only classification bypass
# NO decoder-only bbox bypass
# ==========================================================


class BoxMLP(nn.Module):

    def __init__(
        self,
        hidden_dim=
            CONFIG[
                "hidden_dim"
            ]
    ):
        super().__init__()


        self.layers = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                4
            )
        )


    def forward(self, x):
        return self.layers(x)


class JointSupportConditionedDetectionHead(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=
            CONFIG[
                "hidden_dim"
            ],
        prior_prob=
            CONFIG[
                "foreground_prior_prob"
            ],
        initial_scale=
            CONFIG[
                "initial_logit_scale"
            ]
    ):
        super().__init__()


        # --------------------------------------------------
        # Metric space used by final classifier.
        # --------------------------------------------------

        self.object_metric_projection = (
            nn.Linear(

                hidden_dim,

                hidden_dim,

                bias=False
            )
        )


        self.support_metric_projection = (
            nn.Linear(

                hidden_dim,

                hidden_dim,

                bias=False
            )
        )


        # Start both metric transforms as identity.

        with torch.no_grad():

            nn.init.eye_(
                self
                .object_metric_projection
                .weight
            )

            nn.init.eye_(
                self
                .support_metric_projection
                .weight
            )


        # --------------------------------------------------
        # Positive learnable scale:
        #
        # scale = 1 + softplus(raw_scale)
        #
        # Initialize effective scale to 5.
        # --------------------------------------------------

        target_softplus = (
            float(initial_scale)
            -
            1.0
        )


        raw_scale_init = math.log(

            math.expm1(
                target_softplus
            )
        )


        self.raw_logit_scale = (
            nn.Parameter(

                torch.tensor(
                    raw_scale_init,
                    dtype=torch.float32
                )
            )
        )


        # --------------------------------------------------
        # Learnable foreground bias initialized from
        # prior probability = 0.05.
        # --------------------------------------------------

        prior_bias = math.log(

            prior_prob
            /
            (
                1.0
                -
                prior_prob
            )
        )


        self.class_bias = nn.Parameter(

            torch.tensor(
                [prior_bias],
                dtype=torch.float32
            )
        )


        # --------------------------------------------------
        # Bbox is ALSO predicted from relation feature Z.
        # --------------------------------------------------

        self.box_head = BoxMLP(
            hidden_dim=hidden_dim
        )


    def get_logit_scale(self):

        return (

            1.0

            +
            F.softplus(
                self.raw_logit_scale
            )
        )


    def compute_support_similarity(
        self,
        relation_features,
        support_prototype
    ):

        z_metric = F.normalize(

            self.object_metric_projection(
                relation_features
            ),

            p=2,

            dim=-1
        )


        p_metric = F.normalize(

            self.support_metric_projection(
                support_prototype
            ),

            p=2,

            dim=-1
        )


        similarity = (

            z_metric

            *
            p_metric.unsqueeze(1)

        ).sum(

            dim=-1,

            keepdim=True
        )


        return similarity


    def forward(
        self,
        relation_features,
        support_prototype
    ):

        similarity = (

            self.compute_support_similarity(

                relation_features,

                support_prototype
            )
        )


        pred_logits = (

            self.get_logit_scale()

            *
            similarity

            +
            self.class_bias
        )


        pred_boxes = torch.sigmoid(

            self.box_head(
                relation_features
            )
        )


        return (
            pred_logits,
            pred_boxes,
            similarity
        )


print("=" * 70)
print("STEP 10 : JOINT SUPPORT-CONDITIONED HEAD DEFINED")
print("=" * 70)
