# ==========================================================
# STEP 15 — FULL REPLACEMENT
# Aspect-ratio-preserving letterbox
# ==========================================================

from PIL import Image

from torchvision.transforms import (
    functional as TF
)

from torchvision.transforms import (
    InterpolationMode
)


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# Mean-color padding => approximately zero after normalization.
IMAGENET_MEAN_FILL = tuple(
    int(
        round(
            value * 255.0
        )
    )
    for value
    in IMAGENET_MEAN
)


class LetterboxTransform:

    def __init__(
        self,
        size=CONFIG['image_size'],
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        fill=IMAGENET_MEAN_FILL,
    ):

        self.size = int(
            size
        )

        self.mean = list(
            mean
        )

        self.std = list(
            std
        )

        self.fill = tuple(
            fill
        )

        if self.size <= 0:
            raise ValueError(
                'Letterbox size must be positive.'
            )

    def __call__(
        self,
        image,
    ):

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                'LetterboxTransform expects PIL.Image.'
            )

        image = image.convert(
            'RGB'
        )

        (
            original_width,
            original_height,

        ) = image.size

        if (
            original_width <= 0
            or
            original_height <= 0
        ):
            raise ValueError(
                'Invalid image size.'
            )

        scale = min(
            self.size
            /
            float(
                original_width
            ),

            self.size
            /
            float(
                original_height
            ),
        )

        resized_width = max(
            1,
            int(
                round(
                    original_width
                    *
                    scale
                )
            ),
        )

        resized_height = max(
            1,
            int(
                round(
                    original_height
                    *
                    scale
                )
            ),
        )

        resized_width = min(
            resized_width,
            self.size,
        )

        resized_height = min(
            resized_height,
            self.size,
        )

        resized = TF.resize(
            image,
            [
                resized_height,
                resized_width,
            ],
            interpolation=
                InterpolationMode.BILINEAR,
            antialias=True,
        )

        pad_left = (
            self.size
            -
            resized_width
        ) // 2

        pad_top = (
            self.size
            -
            resized_height
        ) // 2

        pad_right = (
            self.size
            -
            resized_width
            -
            pad_left
        )

        pad_bottom = (
            self.size
            -
            resized_height
            -
            pad_top
        )

        canvas = Image.new(
            'RGB',
            (
                self.size,
                self.size,
            ),
            color=
                self.fill,
        )

        canvas.paste(
            resized,
            (
                pad_left,
                pad_top,
            )
        )

        tensor = TF.to_tensor(
            canvas
        )

        tensor = TF.normalize(
            tensor,
            mean=
                self.mean,
            std=
                self.std,
        )

        # False = image content
        # True  = letterbox padding
        padding_mask = torch.ones(
            (
                self.size,
                self.size,
            ),
            dtype=torch.bool,
        )

        padding_mask[
            pad_top:
                pad_top + resized_height,

            pad_left:
                pad_left + resized_width,

        ] = False

        # Actual factors after integer resizing.
        scale_x = (
            resized_width
            /
            float(
                original_width
            )
        )

        scale_y = (
            resized_height
            /
            float(
                original_height
            )
        )

        meta = {

            'original_width':
                int(
                    original_width
                ),

            'original_height':
                int(
                    original_height
                ),

            'resized_width':
                int(
                    resized_width
                ),

            'resized_height':
                int(
                    resized_height
                ),

            'pad_left':
                int(
                    pad_left
                ),

            'pad_top':
                int(
                    pad_top
                ),

            'pad_right':
                int(
                    pad_right
                ),

            'pad_bottom':
                int(
                    pad_bottom
                ),

            'scale_x':
                float(
                    scale_x
                ),

            'scale_y':
                float(
                    scale_y
                ),

            'canvas_size':
                int(
                    self.size
                ),
        }

        return {

            'image':
                tensor,

            'padding_mask':
                padding_mask,

            'meta':
                meta,
        }


support_transform = (
    LetterboxTransform(
        size=
            CONFIG['image_size']
    )
)

query_transform = (
    LetterboxTransform(
        size=
            CONFIG['image_size']
    )
)


# ==========================================================
# SANITY
# ==========================================================

assert len(IMAGENET_MEAN) == 3
assert len(IMAGENET_STD) == 3

_test_image = Image.new(
    'RGB',
    (
        1600,
        900,
    ),
    color=(
        128,
        128,
        128,
    ),
)

_test_output = (
    query_transform(
        _test_image
    )
)

assert tuple(
    _test_output[
        'image'
    ].shape
) == (
    3,
    CONFIG['image_size'],
    CONFIG['image_size'],
)

assert tuple(
    _test_output[
        'padding_mask'
    ].shape
) == (
    CONFIG['image_size'],
    CONFIG['image_size'],
)

assert (
    _test_output[
        'padding_mask'
    ].dtype
    ==
    torch.bool
)

assert bool(
    (
        ~_test_output[
            'padding_mask'
        ]
    ).any()
)

assert bool(
    _test_output[
        'padding_mask'
    ].any()
)


print('=' * 70)
print('STEP 15 : LETTERBOX TRANSFORMS READY')
print('=' * 70)

print('Canvas size      :', CONFIG['image_size'])
print('Aspect ratio     : preserved')
print('Padding          : center padding')
print('Padding mask     : enabled')
print('Padding fill RGB :', IMAGENET_MEAN_FILL)
print('Normalization    : ImageNet')
print('Geometry aug.    : disabled for now')

print('=' * 70)


del _test_image
del _test_output
