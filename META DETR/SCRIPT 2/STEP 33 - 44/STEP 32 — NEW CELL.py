# ==========================================================
# STEP 32 — NEW CELL
# FINAL INFERENCE-TIME BENCHMARK
#
# FINAL METRIC:
#   mean milliseconds / image
#
# LOCK:
#   batch = 1
#   query = 640x640
#   same GPU
#   support prototype CACHED
#   support encoding excluded
#   disk I/O excluded
#   host->GPU transfer excluded
#   20 warm-up iterations
#   >=100 queries / preferably full CCTV-Test
# ==========================================================

import time


@torch.no_grad()
def cache_single_support_prototype(
    target_model,
    support_images,
    support_padding_masks,
):

    target_model.eval()


    device = CONFIG[
        'device'
    ]


    support_images = (
        support_images
        .to(
            device,
            non_blocking=True,
        )
    )


    support_padding_masks = (
        support_padding_masks
        .to(
            device,
            non_blocking=True,
        )
    )


    if (
        support_images.shape[0]
        !=
        1
    ):

        raise ValueError(
            'This helper is for ONE already-selected '
            'support prototype. For 3-shot/5-shot, '
            'build the final aggregated prototype in '
            'the CCTV adaptation pipeline first, then '
            'pass that cached prototype directly to '
            'benchmark_cached_support_inference().'
        )


    prototype = (
        target_model.encode_support(

            support_images,

            support_padding_mask=
                support_padding_masks,
        )
    )


    return (
        prototype
        .detach()
    )


@torch.no_grad()
def forward_with_cached_support(
    target_model,
    support_prototype,
    query_images,
    query_padding_masks,
):

    (
        decoder_objects,
        _,

    ) = target_model.encode_query(

        query_images,

        query_padding_mask=
            query_padding_masks,
    )


    (
        outputs,
        _,

    ) = target_model.condition_and_predict(

        decoder_objects,

        support_prototype,
    )


    return outputs


@torch.no_grad()
def benchmark_cached_support_inference(
    target_model,
    support_prototype,
    query_loader,
    warmup_iterations=20,
    min_images=100,
    max_images=None,
):

    target_model.eval()


    device = CONFIG[
        'device'
    ]


    support_prototype = (
        support_prototype
        .detach()
        .to(
            device
        )
    )


    warmup_iterations = int(
        warmup_iterations
    )


    min_images = int(
        min_images
    )


    if warmup_iterations < 0:

        raise ValueError(
            'warmup_iterations must be >= 0.'
        )


    if min_images < 1:

        raise ValueError(
            'min_images must be >= 1.'
        )


    # ======================================================
    # GET ONE QUERY FOR WARM-UP
    #
    # Data loading and H->D copy occur OUTSIDE timer.
    # ======================================================

    warmup_batch = next(
        iter(
            query_loader
        )
    )


    warmup_query = (
        warmup_batch[
            'query_images'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )


    warmup_mask = (
        warmup_batch[
            'query_padding_masks'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )


    if (
        warmup_query.shape[0]
        !=
        1
    ):

        raise ValueError(
            'Inference benchmark requires batch size 1.'
        )


    # ======================================================
    # WARM-UP
    # ======================================================

    for _ in range(
        warmup_iterations
    ):

        _ = (
            forward_with_cached_support(

                target_model,

                support_prototype,

                warmup_query,

                warmup_mask,
            )
        )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    # ======================================================
    # MEASURE
    # ======================================================

    timings_ms = []


    for batch in query_loader:

        query_images = (
            batch[
                'query_images'
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        query_padding_masks = (
            batch[
                'query_padding_masks'
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        if (
            query_images.shape[0]
            !=
            1
        ):

            raise ValueError(
                'Inference benchmark requires '
                'batch size 1.'
            )


        # H->D transfer above is excluded from timing.

        if torch.cuda.is_available():
            torch.cuda.synchronize()


        start_time = (
            time.perf_counter()
        )


        _ = (
            forward_with_cached_support(

                target_model,

                support_prototype,

                query_images,

                query_padding_masks,
            )
        )


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        end_time = (
            time.perf_counter()
        )


        elapsed_ms = (
            end_time
            -
            start_time
        ) * 1000.0


        timings_ms.append(
            float(
                elapsed_ms
            )
        )


        if (
            max_images is not None
            and
            len(
                timings_ms
            )
            >=
            int(
                max_images
            )
        ):

            break


    if (
        len(
            timings_ms
        )
        <
        min_images
    ):

        raise RuntimeError(
            'Inference benchmark used too few images: '
            f'{len(timings_ms)} < {min_images}.'
        )


    timings_ms = np.asarray(
        timings_ms,
        dtype=np.float64,
    )


    result = {

        # FINAL metric
        'mean_ms_per_image':
            float(
                timings_ms.mean()
            ),

        # Diagnostic metadata only
        'num_images':
            int(
                len(
                    timings_ms
                )
            ),

        'std_ms':
            float(
                timings_ms.std()
            ),

        'median_ms':
            float(
                np.median(
                    timings_ms
                )
            ),

        'warmup_iterations':
            warmup_iterations,

        'batch_size':
            1,

        'image_size':
            CONFIG[
                'image_size'
            ],

        'support_prototype_cached':
            True,

        'device':
            str(
                device
            ),

        'gpu':
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else
                'CPU'
            ),
    }


    return result


print('=' * 70)
print('STEP 32 : INFERENCE BENCHMARK READY')
print('=' * 70)

print(
    'Final metric : mean ms/image'
)

print(
    'Batch        : 1'
)

print(
    'Input        : 640x640'
)

print(
    'Warm-up      : 20'
)

print(
    'Support      : cached prototype'
)

print(
    'Timing scope : query encoder + transformer + '
    'relation + detection head'
)

print(
    'Excluded     : disk I/O + support encoding + '
    'host-to-GPU copy'
)

print('=' * 70)
