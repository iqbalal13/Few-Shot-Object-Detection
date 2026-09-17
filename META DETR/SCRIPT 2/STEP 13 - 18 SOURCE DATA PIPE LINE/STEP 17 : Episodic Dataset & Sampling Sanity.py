# ==========================================================
# STEP 17 — FULL REPLACEMENT
# Letterbox / bbox / mask sanity
# ==========================================================

import matplotlib.pyplot as plt

from matplotlib.patches import (
    Rectangle
)


def display_tensor_image(
    tensor
):

    array = (
        tensor
        .detach()
        .cpu()
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )

    array = (
        array
        *
        np.asarray(
            IMAGENET_STD
        )
        +
        np.asarray(
            IMAGENET_MEAN
        )
    )

    return np.clip(
        array,
        0.0,
        1.0,
    )


train_dataset.set_epoch(
    0
)


seen_classes = set()


for index in range(
    min(
        160,
        len(
            train_dataset
        ),
    )
):

    episode = (
        train_dataset[
            index
        ]
    )

    semantic_label = int(
        episode[
            'episode_class'
        ].item()
    )

    seen_classes.add(
        semantic_label
    )

    support_target = (
        episode[
            'support_target'
        ]
    )

    query_target = (
        episode[
            'query_target'
        ]
    )

    assert (
        support_target[
            'semantic_label'
        ].item()
        ==
        semantic_label
    )

    assert (
        query_target[
            'semantic_label'
        ].item()
        ==
        semantic_label
    )

    assert (
        support_target[
            'image_id'
        ].item()
        !=
        query_target[
            'image_id'
        ].item()
    )

    assert tuple(
        episode[
            'support_image'
        ].shape[-2:]
    ) == (
        CONFIG['image_size'],
        CONFIG['image_size'],
    )

    assert tuple(
        episode[
            'query_image'
        ].shape[-2:]
    ) == (
        CONFIG['image_size'],
        CONFIG['image_size'],
    )

    for mask_key in (
        'support_padding_mask',
        'query_padding_mask',
    ):

        mask = (
            episode[
                mask_key
            ]
        )

        assert (
            mask.dtype
            ==
            torch.bool
        )

        assert tuple(
            mask.shape
        ) == (
            CONFIG['image_size'],
            CONFIG['image_size'],
        )

        assert bool(
            (
                ~mask
            ).any()
        )

    assert (
        len(
            query_target[
                'boxes'
            ]
        )
        >
        0
    )

    assert bool(
        (
            query_target[
                'labels'
            ]
            ==
            0
        ).all()
    )

    boxes = (
        query_target[
            'boxes'
        ]
    )

    assert torch.isfinite(
        boxes
    ).all()

    assert bool(
        (
            boxes[
                :,
                2:
            ]
            >
            0
        ).all()
    )

    xy_min = (
        boxes[
            :,
            :2
        ]
        -
        boxes[
            :,
            2:
        ]
        /
        2
    )

    xy_max = (
        boxes[
            :,
            :2
        ]
        +
        boxes[
            :,
            2:
        ]
        /
        2
    )

    assert bool(
        (
            xy_min
            >=
            -1e-6
        ).all()
    )

    assert bool(
        (
            xy_max
            <=
            1.0
            +
            1e-6
        ).all()
    )


print(
    'Semantic classes observed in first sanity window:',
    len(
        seen_classes
    ),
)


# ==========================================================
# Visualize four distinct classes
# ==========================================================

visual_indices = []

used_labels = set()


for index, label in enumerate(
    train_dataset
    .episode_labels
):

    if label not in used_labels:

        visual_indices.append(
            index
        )

        used_labels.add(
            label
        )

    if len(
        visual_indices
    ) >= 4:
        break


for index in visual_indices:

    episode = (
        train_dataset[
            index
        ]
    )

    semantic_label = int(
        episode[
            'episode_class'
        ].item()
    )

    category_name = (
        CATEGORY_NAMES[
            semantic_label
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12,
            6,
        ),
    )

    axes[0].imshow(
        display_tensor_image(
            episode[
                'support_image'
            ]
        )
    )

    axes[0].set_title(
        f'Support letterbox: {category_name}'
    )

    axes[1].imshow(
        display_tensor_image(
            episode[
                'query_image'
            ]
        )
    )

    (
        height,
        width,

    ) = episode[
        'query_image'
    ].shape[-2:]

    for (
        cx,
        cy,
        w,
        h,

    ) in episode[
        'query_target'
    ][
        'boxes'
    ].tolist():

        axes[1].add_patch(
            Rectangle(
                (
                    (
                        cx
                        -
                        w / 2
                    )
                    *
                    width,

                    (
                        cy
                        -
                        h / 2
                    )
                    *
                    height,
                ),

                w
                *
                width,

                h
                *
                height,

                fill=False,
                linewidth=2,
            )
        )

    axes[1].set_title(
        f'Query letterbox: all {category_name} GT'
    )

    for axis in axes:
        axis.axis(
            'off'
        )

    plt.tight_layout()
    plt.show()
    plt.close(
        fig
    )


print('=' * 70)
print(
    'STEP 17 PASS : LETTERBOX + BBOX + '
    'MASK SANITY VALID'
)
print('=' * 70)
