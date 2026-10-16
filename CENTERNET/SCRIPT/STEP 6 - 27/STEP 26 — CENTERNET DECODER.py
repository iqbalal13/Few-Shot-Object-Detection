# ============================================================
# STEP 26 — CENTERNET DECODER
# ============================================================

print("=" * 70)
print("STEP 26 — BUILD CENTERNET DECODER")
print("=" * 70)


def heatmap_local_nms(
    heatmap,
    kernel=3
):

    padding = (
        kernel - 1
    ) // 2

    local_max = F.max_pool2d(
        heatmap,
        kernel_size=kernel,
        stride=1,
        padding=padding
    )

    keep = (
        local_max == heatmap
    ).float()

    return heatmap * keep


def gather_feature(
    feature,
    indices
):

    channels = feature.size(
        2
    )

    indices = indices.unsqueeze(
        2
    ).expand(
        indices.size(0),
        indices.size(1),
        channels
    )

    return feature.gather(
        1,
        indices
    )


def topk_centers(
    scores,
    K=100
):
    """
    scores:
        [B, C, H, W]
    """

    batch, classes, height, width = \
        scores.shape

    K = min(
        K,
        height * width
    )

    topk_scores, topk_indices = \
        torch.topk(
            scores.view(
                batch,
                classes,
                -1
            ),
            K
        )

    topk_indices = (
        topk_indices
        % (height * width)
    )

    topk_y = (
        topk_indices // width
    ).float()

    topk_x = (
        topk_indices % width
    ).float()

    flattened_scores = \
