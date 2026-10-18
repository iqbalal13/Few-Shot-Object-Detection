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
        topk_scores.view(
            batch,
            -1
        )

    final_scores, final_indices = \
        torch.topk(
            flattened_scores,
            K
        )

    final_classes = (
        final_indices // K
    ).long()

    topk_indices = gather_feature(
        topk_indices.view(
            batch,
            -1,
            1
        ),
        final_indices
    ).squeeze(2)

    topk_y = gather_feature(
        topk_y.view(
            batch,
            -1,
            1
        ),
        final_indices
    ).squeeze(2)

    topk_x = gather_feature(
        topk_x.view(
            batch,
            -1,
            1
        ),
        final_indices
    ).squeeze(2)

    return (
        final_scores,
        topk_indices,
        final_classes,
        topk_y,
        topk_x
    )


def decode_centernet(
    outputs,
    K=100,
    score_threshold=None,
    output_stride=None
):

    if score_threshold is None:

        score_threshold = CONFIG[
            "score_threshold"
        ]

    if output_stride is None:

        output_stride = CONFIG[
            "output_stride"
        ]

    heatmap = torch.sigmoid(
        outputs["heatmap"]
    )

    heatmap = heatmap_local_nms(
        heatmap
    )

    (
        scores,
        indices,
        classes,
        ys,
        xs
    ) = topk_centers(
        heatmap,
        K=K
    )

    wh = transpose_and_gather_feature(
        outputs["wh"],
        indices
    )

    offset = transpose_and_gather_feature(
        outputs["offset"],
        indices
    )

    # WH should represent positive object size.
    wh = torch.clamp(
        wh,
        min=0.0
    )

    xs = (
        xs
        + offset[..., 0]
    )

    ys = (
        ys
        + offset[..., 1]
    )

    half_w = (
        wh[..., 0]
        / 2.0
    )

    half_h = (
        wh[..., 1]
        / 2.0
    )

    x1 = (
        xs - half_w
    ) * output_stride

    y1 = (
        ys - half_h
    ) * output_stride

    x2 = (
        xs + half_w
    ) * output_stride

    y2 = (
        ys + half_h
    ) * output_stride

    boxes = torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=-1
    )

    results = []

    for batch_index in range(
        boxes.size(0)
    ):

        keep = (
            scores[batch_index]
            >= score_threshold
        )

        batch_boxes = boxes[
            batch_index
        ][keep]

        batch_scores = scores[
            batch_index
        ][keep]

        batch_classes = classes[
            batch_index
        ][keep]

        # Clip coordinates to preprocessed image size
        batch_boxes[:, 0::2] = \
            batch_boxes[:, 0::2].clamp(
                0,
                CONFIG["input_size"] - 1
            )

        batch_boxes[:, 1::2] = \
            batch_boxes[:, 1::2].clamp(
                0,
                CONFIG["input_size"] - 1
            )

        results.append(
            {
                "boxes": batch_boxes,
                "scores": batch_scores,
                "labels": batch_classes
            }
        )

    return results


print(
    "Primary score threshold :",
    CONFIG["score_threshold"]
)

print(
    "Primary IoU threshold   :",
    CONFIG["iou_threshold"]
)

print("Top-K default            : 100")

print("\nSTEP 26 PASSED")

def inspect_raw_centernet_decode(outputs, K=100):
    """Diagnostic helper; does not change decode_centernet output."""
    with torch.no_grad():
        hm=heatmap_local_nms(outputs["heatmap"].float().sigmoid())
        _,inds,_,_,_=topk_centers(hm,K=K)
        wh=transpose_and_gather_feature(outputs["wh"].float(),inds)
        off=transpose_and_gather_feature(outputs["offset"].float(),inds)
        return {
            "candidate_count": int(wh.shape[0]*wh.shape[1]),
            "nonpositive_wh": int((wh<=0).any(-1).sum().item()),
            "offset_outside_0_1": int(((off<0)|(off>=1)).any(-1).sum().item()),
            "wh_min": float(wh.min().item()),
            "wh_median": float(wh.median().item()),
            "offset_min": float(off.min().item()),
            "offset_max": float(off.max().item()),
        }
