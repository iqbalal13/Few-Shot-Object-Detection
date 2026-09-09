# ==========================================================
# STEP 13 : Source COCO Configuration & Setup
# ROBUST DOWNLOAD VERSION
# ==========================================================

import os
import sys
import zipfile
import subprocess
import shutil

import torch


# ==========================================================
# DEPENDENCY CHECK
# ==========================================================

assert "CONFIG" in globals(), (
    "Run STEP 3 first."
)

assert "COCO_DIR" in globals(), (
    "Run STEP 2 first."
)


# ==========================================================
# SOURCE CONFIGURATION
# ==========================================================

COCO_CONFIG = {

    "num_classes":
        80,

    "episode_way":
        1,

    "support_shot":
        1,

    "num_train_episodes":
        8000,

    "num_val_episodes":
        800,

    "image_size":
        CONFIG["image_size"],

    "min_bbox_size":
        2.0,

    "batch_size":
        1,

    "shuffle":
        False,

    "num_workers":
        2,

    "pin_memory":
        torch.cuda.is_available(),

    "seed":
        CONFIG["seed"],

    "train_images":
        "train2017",

    "val_images":
        "val2017",

    "train_annotation":
        "annotations/instances_train2017.json",

    "val_annotation":
        "annotations/instances_val2017.json",
}


assert COCO_CONFIG["num_classes"] == 80
assert COCO_CONFIG["episode_way"] == 1
assert COCO_CONFIG["support_shot"] == 1
assert COCO_CONFIG["batch_size"] == 1
assert COCO_CONFIG["shuffle"] is False
assert COCO_CONFIG["num_val_episodes"] % 80 == 0


# ==========================================================
# POSSIBLE COCO ROOTS
# ==========================================================

LEGACY_COCO_ROOT = (
    "/content/datasets/coco"
)


def coco_root_is_complete(root):

    required = [

        os.path.join(
            root,
            COCO_CONFIG["train_images"]
        ),

        os.path.join(
            root,
            COCO_CONFIG["val_images"]
        ),

        os.path.join(
            root,
            COCO_CONFIG["train_annotation"]
        ),

        os.path.join(
            root,
            COCO_CONFIG["val_annotation"]
        ),
    ]

    return all(
        os.path.exists(path)
        for path in required
    )


# ==========================================================
# CHOOSE ROOT
# ==========================================================

if coco_root_is_complete(
    COCO_DIR
):

    COCO_ROOT = COCO_DIR

    print(
        "✓ Using clean-project COCO."
    )


elif coco_root_is_complete(
    LEGACY_COCO_ROOT
):

    COCO_ROOT = (
        LEGACY_COCO_ROOT
    )

    print(
        "✓ Reusing existing /content/datasets/coco."
    )


else:

    # Use legacy-style runtime path.
    # This keeps dataset separate from model/checkpoint project.
    COCO_ROOT = (
        LEGACY_COCO_ROOT
    )

    os.makedirs(
        COCO_ROOT,
        exist_ok=True
    )


    # ======================================================
    # COCO DOWNLOAD URLS
    #
    # Use HTTP first because the current runtime has
    # SSL certificate mismatch with the HTTPS COCO endpoint.
    # ======================================================

    downloads = {

        "train2017.zip":
            (
                "http://images.cocodataset.org/"
                "zips/train2017.zip"
            ),

        "val2017.zip":
            (
                "http://images.cocodataset.org/"
                "zips/val2017.zip"
            ),

        "annotations_trainval2017.zip":
            (
                "http://images.cocodataset.org/"
                "annotations/"
                "annotations_trainval2017.zip"
            ),
    }


    markers = {

        "train2017.zip":
            os.path.join(
                COCO_ROOT,
                "train2017"
            ),

        "val2017.zip":
            os.path.join(
                COCO_ROOT,
                "val2017"
            ),

        "annotations_trainval2017.zip":
            os.path.join(
                COCO_ROOT,
                "annotations",
                "instances_train2017.json"
            ),
    }


    # ======================================================
    # DOWNLOAD HELPER
    # ======================================================

    def download_with_wget(
        url,
        output_path
    ):

        # Remove partial/corrupt previous download.
        if os.path.exists(
            output_path
        ):

            if not zipfile.is_zipfile(
                output_path
            ):

                print(
                    "Removing incomplete ZIP:",
                    output_path
                )

                os.remove(
                    output_path
                )


        # Already valid.
        if (
            os.path.exists(
                output_path
            )
            and
            zipfile.is_zipfile(
                output_path
            )
        ):

            print(
                "✓ Valid ZIP already exists"
            )

            return


        print(
            "Downloading..."
        )


        normal_cmd = [

            "wget",

            "-c",

            "-O",
            output_path,

            url
        ]


        result = subprocess.run(
            normal_cmd
        )


        # --------------------------------------------------
        # Runtime-specific SSL fallback.
        #
        # Only used if normal wget fails.
        # --------------------------------------------------

        if result.returncode != 0:

            print(
                "Normal download failed."
            )

            print(
                "Retrying COCO download "
                "with certificate check disabled..."
            )


            if os.path.exists(
                output_path
            ):

                os.remove(
                    output_path
                )


            fallback_cmd = [

                "wget",

                "--no-check-certificate",

                "-c",

                "-O",
                output_path,

                url
            ]


            subprocess.run(

                fallback_cmd,

                check=True
            )


        # --------------------------------------------------
        # Validate downloaded ZIP
        # --------------------------------------------------

        if not zipfile.is_zipfile(
            output_path
        ):

            raise RuntimeError(

                "Downloaded file is not a valid ZIP: "
                f"{output_path}"
            )


        print(
            "✓ Download finished and ZIP verified"
        )


    # ======================================================
    # DOWNLOAD + EXTRACT
    # ======================================================

    for (
        filename,
        url
    ) in downloads.items():

        print("-" * 70)

        print(
            filename
        )


        marker = (
            markers[
                filename
            ]
        )


        if os.path.exists(
            marker
        ):

            print(
                "✓ Already extracted"
            )

            continue


        zip_path = os.path.join(
            COCO_ROOT,
            filename
        )


        download_with_wget(

            url,
            zip_path
        )


        print(
            "Extracting..."
        )


        with zipfile.ZipFile(

            zip_path,

            "r"

        ) as archive:

            archive.extractall(
                COCO_ROOT
            )


        print(
            "✓ Extraction finished"
        )


        # Remove ZIP after successful extraction
        # to save Colab disk space.

        os.remove(
            zip_path
        )


# ==========================================================
# REQUIRED PATHS
# ==========================================================

TRAIN_IMAGE_DIR = os.path.join(

    COCO_ROOT,

    COCO_CONFIG[
        "train_images"
    ]
)


VAL_IMAGE_DIR = os.path.join(

    COCO_ROOT,

    COCO_CONFIG[
        "val_images"
    ]
)


TRAIN_ANN_PATH = os.path.join(

    COCO_ROOT,

    COCO_CONFIG[
        "train_annotation"
    ]
)


VAL_ANN_PATH = os.path.join(

    COCO_ROOT,

    COCO_CONFIG[
        "val_annotation"
    ]
)


required_paths = {

    "Train Images":
        TRAIN_IMAGE_DIR,

    "Val Images":
        VAL_IMAGE_DIR,

    "Train Annotation":
        TRAIN_ANN_PATH,

    "Val Annotation":
        VAL_ANN_PATH,
}


# ==========================================================
# VERIFY STRUCTURE
# ==========================================================

print("=" * 70)
print("VERIFYING COCO STRUCTURE")
print("=" * 70)


for (
    name,
    path
) in required_paths.items():

    exists = os.path.exists(
        path
    )


    print(

        f"{name:20s}: "
        f"{'✓ FOUND' if exists else '✗ MISSING'}"
    )


    assert exists, (
        f"{name} missing: {path}"
    )


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
        "pycocotools"
    ])


    from pycocotools.coco import COCO


print("=" * 70)
print("STEP 13 : SOURCE COCO SETUP READY")
print("=" * 70)

print(
    "COCO Root          :",
    COCO_ROOT
)

print(
    "Train Episodes     :",
    COCO_CONFIG[
        "num_train_episodes"
    ]
)

print(
    "Validation Episodes:",
    COCO_CONFIG[
        "num_val_episodes"
    ]
)

print(
    "Classes            :",
    COCO_CONFIG[
        "num_classes"
    ]
)

print(
    "Episode Way        :",
    COCO_CONFIG[
        "episode_way"
    ]
)

print(
    "Support Shot       :",
    COCO_CONFIG[
        "support_shot"
    ]
)

print("=" * 70)
