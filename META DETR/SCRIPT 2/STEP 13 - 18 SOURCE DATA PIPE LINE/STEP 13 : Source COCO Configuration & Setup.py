# ==========================================================
# STEP 13 : COCO-80 Source Dataset Setup
#
# Dataset:
# MS COCO 2017
#
# Stage 1:
# all 80 instance categories are available
# for episodic support-conditioned training.
# ==========================================================

import os
import sys
import time
import zipfile
import subprocess
import urllib.request
import urllib.error
import http.client


assert (
    "CONFIG" in globals()
), "Run STEP 3 first."

assert (
    "COCO_DIR" in globals()
), "Run STEP 2 first."


COCO_CONFIG = {

    # COCO has 80 instance categories.
    "num_source_categories": 80,

    # Episodic task is 1-way.
    "episode_way": 1,

    # One support crop per source episode.
    "support_shot": 1,

    # Initial full source-training plan.
    "num_train_episodes": 8000,

    "num_val_episodes": 800,

    "image_size": (
        CONFIG[
            "image_size"
        ]
    ),

    "min_bbox_size": 2.0,

    "batch_size": 1,

    "num_workers": 2,

    "pin_memory": (
        torch.cuda.is_available()
    ),

    "seed": (
        CONFIG[
            "seed"
        ]
    ),

    "train_images": (
        "train2017"
    ),

    "val_images": (
        "val2017"
    ),

    "train_annotation": (
        "annotations/"
        "instances_train2017.json"
    ),

    "val_annotation": (
        "annotations/"
        "instances_val2017.json"
    ),
}


assert (
    COCO_CONFIG[
        "num_source_categories"
    ]
    ==
    CONFIG[
        "source_num_categories"
    ]
)


# ==========================================================
# DOWNLOAD SOURCES
# ==========================================================

COCO_DOWNLOAD_BASES = (

    (
        "https://s3.us-east-1.amazonaws.com/"
        "images.cocodataset.org/"
    ),

    (
        "https://s3.amazonaws.com/"
        "images.cocodataset.org/"
    ),
)


COCO_ASSETS = (

    (
        "zips/train2017.zip",
        "train2017",
        118287,
    ),

    (
        "zips/val2017.zip",
        "val2017",
        5000,
    ),

    (
        "annotations/"
        "annotations_trainval2017.zip",
        None,
        None,
    ),
)


def count_coco_images(
    folder
):

    if not os.path.isdir(
        folder
    ):
        return 0

    with os.scandir(
        folder
    ) as entries:

        return sum(

            entry.is_file()
            and
            entry.name.lower().endswith(
                ".jpg"
            )
            and
            entry.stat().st_size > 0

            for entry in entries
        )


def coco_asset_is_complete(
    root,
    asset
):

    _, folder, expected_count = (
        asset
    )

    if folder is not None:

        return (
            count_coco_images(
                os.path.join(
                    root,
                    folder
                )
            )
            >=
            expected_count
        )

    return all(

        os.path.isfile(
            os.path.join(
                root,
                COCO_CONFIG[key]
            )
        )
        and
        os.path.getsize(
            os.path.join(
                root,
                COCO_CONFIG[key]
            )
        )
        > 0

        for key in (
            "train_annotation",
            "val_annotation",
        )
    )


def coco_root_is_complete(
    root
):

    return all(
        coco_asset_is_complete(
            root,
            asset
        )
        for asset in COCO_ASSETS
    )


def download_coco_zip(
    relative_url,
    archive_path
):

    # Already complete.
    if zipfile.is_zipfile(
        archive_path
    ):

        print(
            "Using existing ZIP:",
            os.path.basename(
                archive_path
            )
        )

        return

    part_path = (
        archive_path
        +
        ".part"
    )

    # A completed .part file can be promoted.
    if zipfile.is_zipfile(
        part_path
    ):

        os.replace(
            part_path,
            archive_path
        )

        return

    last_error = None

    for base_url in (
        COCO_DOWNLOAD_BASES
    ):

        url = (
            base_url
            +
            relative_url
        )

        for attempt in range(
            1,
            4
        ):

            offset = (
                os.path.getsize(
                    part_path
                )
                if os.path.isfile(
                    part_path
                )
                else 0
            )

            headers = {
                "User-Agent":
                    "MetaDETR-COCO/1.0"
            }

            if offset > 0:
                headers[
                    "Range"
                ] = (
                    f"bytes={offset}-"
                )

            print()
            print(
                "Downloading:",
                relative_url,
                f"| attempt {attempt}/3"
            )
            print(
                "URL:",
                url
            )

            request = (
                urllib.request.Request(
                    url,
                    headers=headers
                )
            )

            try:

                with urllib.request.urlopen(
                    request,
                    timeout=60

                ) as response:

                    status = (
                        response.getcode()
                    )

                    if status == 206:

                        content_range = (
                            response
                            .headers
                            .get(
                                "Content-Range",
                                ""
                            )
                        )

                        if not (
                            content_range
                            .startswith(
                                f"bytes {offset}-"
                            )
                        ):
                            raise RuntimeError(
                                "Server resume offset "
                                "does not match."
                            )

                        mode = (
                            "ab"
                            if offset > 0
                            else "wb"
                        )

                    elif status == 200:

                        # Server starts from byte zero.
                        offset = 0
                        mode = "wb"

                    else:

                        raise RuntimeError(
                            "Unexpected HTTP "
                            f"status {status}"
                        )

                    length = (
                        response
                        .headers
                        .get(
                            "Content-Length"
                        )
                    )

                    expected_bytes = (
                        int(length)
                        if length
                        else None
                    )

                    received = 0

                    last_report = (
                        time.monotonic()
                    )

                    with open(
                        part_path,
                        mode
                    ) as output_file:

                        while True:

                            chunk = (
                                response.read(
                                    4
                                    *
                                    1024
                                    *
                                    1024
                                )
                            )

                            if not chunk:
                                break

                            output_file.write(
                                chunk
                            )

                            received += len(
                                chunk
                            )

                            if (
                                time.monotonic()
                                -
                                last_report
                                >=
                                5
                            ):

                                downloaded_gib = (
                                    offset
                                    +
                                    received
                                ) / (
                                    1024 ** 3
                                )

                                print(
                                    "\rDownloaded: "
                                    f"{downloaded_gib:.2f} GiB",
                                    end="",
                                    flush=True
                                )

                                last_report = (
                                    time.monotonic()
                                )

                    print()

                    if (
                        expected_bytes
                        is not None
                        and
                        received
                        !=
                        expected_bytes
                    ):

                        raise (
                            http.client
                            .IncompleteRead(
                                b"",
                                expected_bytes
                                -
                                received
                            )
                        )

                if not zipfile.is_zipfile(
                    part_path
                ):

                    if os.path.exists(
                        part_path
                    ):
                        os.remove(
                            part_path
                        )

                    raise zipfile.BadZipFile(
                        "Downloaded file "
                        "is not a valid ZIP."
                    )

                os.replace(
                    part_path,
                    archive_path
                )

                return

            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                zipfile.BadZipFile,
            ) as error:

                last_error = error

                print(
                    "Download interrupted:",
                    repr(error)
                )

                if (
                    isinstance(
                        error,
                        urllib.error.HTTPError
                    )
                    and
                    error.code == 416
                    and
                    os.path.isfile(
                        part_path
                    )
                ):
                    os.remove(
                        part_path
                    )

    raise RuntimeError(
        "Failed to download "
        f"{relative_url}: "
        f"{last_error}"
    )


# ==========================================================
# FIND EXISTING COCO INSTALLATION FIRST
# ==========================================================

candidates = list(
    dict.fromkeys([

        COCO_DIR,

        globals().get(
            "COCO_ROOT",
            COCO_DIR
        ),

        # Old notebook location.
        (
            "/content/"
            "MetaDETR_PersonOnly/"
            "datasets/coco"
        ),

        "/content/datasets/coco",

        (
            "/content/"
            "MetaDETR_Final_Clean/"
            "datasets/coco"
        ),
    ])
)


COCO_ROOT = next(

    (
        root

        for root in candidates

        if coco_root_is_complete(
            root
        )
    ),

    COCO_DIR
)


os.makedirs(
    COCO_ROOT,
    exist_ok=True
)

print(
    "COCO root:",
    COCO_ROOT
)


# ==========================================================
# DOWNLOAD / EXTRACT ONLY MISSING ASSETS
# ==========================================================

for asset in COCO_ASSETS:

    relative_url, _, _ = (
        asset
    )

    filename = (
        os.path.basename(
            relative_url
        )
    )

    if coco_asset_is_complete(
        COCO_ROOT,
        asset
    ):

        print(
            "Already available:",
            filename
        )

        continue

    archive_path = (
        os.path.join(
            COCO_ROOT,
            filename
        )
    )

    download_coco_zip(
        relative_url,
        archive_path
    )

    print(
        "Extracting:",
        filename
    )

    root_abs = os.path.realpath(
        COCO_ROOT
    )

    with zipfile.ZipFile(
        archive_path
    ) as archive:

        # Protect against path traversal.
        for member in (
            archive.infolist()
        ):

            destination = (
                os.path.realpath(
                    os.path.join(
                        root_abs,
                        member.filename
                    )
                )
            )

            if (
                os.path.commonpath(
                    [
                        root_abs,
                        destination
                    ]
                )
                !=
                root_abs
            ):

                raise RuntimeError(
                    "Invalid ZIP path: "
                    +
                    member.filename
                )

        archive.extractall(
            root_abs
        )

    if not coco_asset_is_complete(
        COCO_ROOT,
        asset
    ):

        raise RuntimeError(
            "Extraction incomplete: "
            +
            filename
        )

    os.remove(
        archive_path
    )

    print(
        "Extraction complete:",
        filename
    )


# ==========================================================
# PATHS USED BY FOLLOWING STEPS
# ==========================================================

TRAIN_IMAGE_DIR = (
    os.path.join(
        COCO_ROOT,
        COCO_CONFIG[
            "train_images"
        ]
    )
)

VAL_IMAGE_DIR = (
    os.path.join(
        COCO_ROOT,
        COCO_CONFIG[
            "val_images"
        ]
    )
)

TRAIN_ANN_PATH = (
    os.path.join(
        COCO_ROOT,
        COCO_CONFIG[
            "train_annotation"
        ]
    )
)

VAL_ANN_PATH = (
    os.path.join(
        COCO_ROOT,
        COCO_CONFIG[
            "val_annotation"
        ]
    )
)


assert coco_root_is_complete(
    COCO_ROOT
), "COCO dataset is incomplete."


# ==========================================================
# PYCOCOTOOLS
# ==========================================================

try:
    from pycocotools.coco import COCO

except ImportError:

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pycocotools",
    ])

    from pycocotools.coco import COCO


# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 70)
print("STEP 13 : COCO-80 SOURCE DATA READY")
print("=" * 70)

print(
    "COCO root          :",
    COCO_ROOT
)

print(
    "Source categories  :",
    COCO_CONFIG[
        "num_source_categories"
    ]
)

print(
    "Episode way        :",
    COCO_CONFIG[
        "episode_way"
    ]
)

print(
    "Support shot       :",
    COCO_CONFIG[
        "support_shot"
    ]
)

print(
    "Train episodes     :",
    COCO_CONFIG[
        "num_train_episodes"
    ]
)

print(
    "Validation episodes:",
    COCO_CONFIG[
        "num_val_episodes"
    ]
)

print(
    "Train images       :",
    count_coco_images(
        TRAIN_IMAGE_DIR
    )
)

print(
    "Validation images  :",
    count_coco_images(
        VAL_IMAGE_DIR
    )
)

print("=" * 70)
