# ==========================================================
# STEP 2 — FULL REPLACEMENT
# ==========================================================

PROJECT_ROOT = '/content/MetaDETR_Simplified_Final'

DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
COCO_DIR = os.path.join(DATASET_DIR, 'coco')
CCTV_DIR = os.path.join(DATASET_DIR, 'cctv')

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
COCO80_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'coco80_meta')
PERSON_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'coco_person')
CCTV_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'cctv_fewshot')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

for path in (
    PROJECT_ROOT,
    DATASET_DIR,
    COCO_DIR,
    CCTV_DIR,
    CHECKPOINT_DIR,
    COCO80_CHECKPOINT_DIR,
    PERSON_CHECKPOINT_DIR,
    CCTV_CHECKPOINT_DIR,
    OUTPUT_DIR,
    LOG_DIR,
):
    os.makedirs(path, exist_ok=True)


RESEARCH_PROTOCOL = {

    'task':
        'Cross-Domain Few-Shot Person Detection',

    'model':
        'Simplified Meta-DETR-inspired',

    'stage_1':
        'COCO-80 multi-class episodic '
        'support-conditioned meta-training',

    'stage_1_domain':
        'MS COCO',

    'stage_1_categories':
        80,

    'stage_1_episode_way':
        1,

    'stage_1_support_shot':
        1,

    'stage_1_detection_target':
        'all eligible instances in the query '
        'that match the support category',

    'stage_1_output_semantics':
        'support-match/objectness + bounding box; '
        'NOT 80-way classification',

    # Source readiness gates
    'source_gate_1':
        'stable unseen COCO-Val 80-class '
        'episodic generalization',

    'source_gate_2':
        'COCO-Val person-only AP50/Precision/Recall gate',

    # Fallback only
    'stage_2':
        'COCO-person specialization — FALLBACK ONLY',

    'stage_2_trigger':
        'run only if generic COCO-80 checkpoint passes '
        'the generic gate but person-specific validation '
        'is weak',

    'stage_2_semantic_class':
        'person',

    # CCTV
    'stage_3':
        'CCTV few-shot cross-domain adaptation',

    'target_domain':
        'CCTV',

    'target_semantic_class':
        'person',

    'target_shots':
        [1, 3, 5],

    'shot_definition':
        '1 annotated person instance = 1 shot',

    'nested_support_sets':
        True,

    'nested_shots': {
        1: ['A'],
        3: ['A', 'B', 'C'],
        5: ['A', 'B', 'C', 'D', 'E'],
    },

    'target_initialization':
        'each 1/3/5-shot experiment starts independently '
        'from the same final source checkpoint; normally '
        'the stable COCO-80 checkpoint, or the COCO-person '
        'fallback checkpoint only when the person gate '
        'requires it',

    'target_split_policy':
        'support/train, validation, and test must be '
        'separated by sequence/session/camera/time block '
        'where possible; do not randomly split adjacent '
        'CCTV frames',

    'target_val':
        'no final reporting; hyperparameter/model '
        'selection only',

    'target_test':
        'final evaluation only',

    # Locked final thesis metrics
    'final_metrics': {

        'AP50':
            'IoU >= 0.50; primary accuracy metric',

        'Precision':
            'score >= 0.50 and IoU >= 0.50',

        'Recall':
            'score >= 0.50 and IoU >= 0.50',

        'Inference_Time':
            'mean ms/image; batch size 1; 640x640; '
            'same GPU; support prototype cached',
    },

    'removed_metric':
        'NCAcc',
}


print('=' * 70)
print('STEP 2 : LOCKED RESEARCH PROTOCOL READY')
print('=' * 70)

print('Model        :', RESEARCH_PROTOCOL['model'])
print('Stage 1      :', RESEARCH_PROTOCOL['stage_1'])
print('Source gate  :', RESEARCH_PROTOCOL['source_gate_2'])
print('Stage 2      :', RESEARCH_PROTOCOL['stage_2'])
print('Stage 3      :', RESEARCH_PROTOCOL['stage_3'])
print('Target shots :', RESEARCH_PROTOCOL['target_shots'])
print('Shot unit    :', RESEARCH_PROTOCOL['shot_definition'])
print('Final metrics:', list(RESEARCH_PROTOCOL['final_metrics'].keys()))
print('Project root :', PROJECT_ROOT)

print('=' * 70)
