# ==========================================================
# STEP 2 : Project Paths & Research Protocol
# FINAL CLEAN NOTEBOOK
# ==========================================================

PROJECT_ROOT = (
    "/content/MetaDETR_Final_Clean"
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets"
)

COCO_DIR = os.path.join(
    DATASET_DIR,
    "coco"
)

CCTV_DIR = os.path.join(
    DATASET_DIR,
    "cctv"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)


for path in [
    PROJECT_ROOT,
    DATASET_DIR,
    COCO_DIR,
    CCTV_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LOG_DIR
]:
    os.makedirs(
        path,
        exist_ok=True
    )


# ==========================================================
# LOCKED RESEARCH PROTOCOL
# ==========================================================

RESEARCH_PROTOCOL = {

    "task":
        "Few-Shot Cross-Domain Person Detection",

    "source_domain":
        "MS COCO",

    "target_domain":
        "CCTV",

    "source_train_split":
        "COCO-Train",

    "source_validation_split":
        "COCO-Val",

    "source_test_split":
        "UNUSED",

    "source_episode":
        "1-way episodic across 80 COCO classes",

    "target_semantic_class":
        "person",

    "target_setting":
        "same-class cross-domain few-shot adaptation",

    "shot_definition":
        "1 fully annotated CCTV frame = 1 shot",

    "nested_shots": {
        1: ["A"],
        3: ["A", "B", "C"],
        5: ["A", "B", "C", "D", "E"]
    },

    "target_train":
        "gradient allowed",

    "target_val":
        "no gradient; model selection only",

    "target_test":
        "final evaluation only"
}


print("=" * 70)
print("STEP 2 : PROJECT & PROTOCOL READY")
print("=" * 70)

print("Project Root :", PROJECT_ROOT)
print("Task         :", RESEARCH_PROTOCOL["task"])
print(
    "Source       :",
    RESEARCH_PROTOCOL["source_domain"]
)
print(
    "Target       :",
    RESEARCH_PROTOCOL["target_domain"]
)
print(
    "Source setup :",
    RESEARCH_PROTOCOL["source_episode"]
)
print(
    "Target setup :",
    RESEARCH_PROTOCOL["target_setting"]
)
print("=" * 70)
