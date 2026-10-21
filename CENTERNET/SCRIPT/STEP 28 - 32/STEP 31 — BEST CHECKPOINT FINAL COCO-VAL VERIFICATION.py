
# ============================================================
# STEP 31 — VERIFY B1 CHECKPOINTS AND LOAD BEST (NO TRAINING)
# ============================================================
import os
import torch

assert os.path.isfile(LAST_CHECKPOINT)
assert os.path.isfile(BEST_CHECKPOINT)
assert os.path.isfile(milestone30(20))
last31 = torch.load(LAST_CHECKPOINT, map_location='cpu', weights_only=False)
assert last31['experiment_name'] == CONFIG['experiment_name']
assert last31['source_manifest_sha256'] == source_manifest_sha256
assert last31['source_images_total'] == 160000
assert int(last31['epoch']) == 20
assert len(last31['history']) == 20
assert all(h['train_stats']['batches'] == 1000 for h in last31['history'])
for key, value in {
    'gaussian_radius_mode': 'v1_3_corrected',
    'wh_loss_type': 'hybrid_abs_relative',
    'wh_absolute_fraction': 0.75,
    'wh_relative_fraction': 0.25,
    'wh_relative_eps': 2.0,
    'wh_relative_beta': 0.20,
    'wh_relative_scale': 10.0
}.items():
    assert last31['config'][key] == value, f'LAST mismatch: {key}'

checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu', weights_only=False)
assert checkpoint['experiment_name'] == CONFIG['experiment_name']
assert checkpoint['source_manifest_sha256'] == source_manifest_sha256
assert checkpoint['source_images_total'] == 160000
assert int(checkpoint['epoch']) in (5, 10, 15, 20)
assert checkpoint['config']['wh_loss_type'] == 'hybrid_abs_relative'
assert checkpoint['config']['gaussian_radius_mode'] == 'v1_3_corrected'
assert abs(float(checkpoint['val_stats']['official_map']) - float(last31['best_map'])) < 1e-9

if 'best_model' in globals():
    del best_model
    torch.cuda.empty_cache()
best_model = CenterNetResNet101(
    num_classes=80, pretrained_backbone=False).to(device)
best_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
best_model.eval()

if 'source_val_loader' not in globals():
    source_val_loader, source_val_generator = build_val_loader(SEED + 3301, encoded=True)
final_source_metrics = evaluate_centernet(
    best_model, source_val_loader, max_batches=None,
    score_threshold=CONFIG['score_threshold'],
    iou_threshold=CONFIG['iou_threshold'], compute_official_ap=False)
print('STEP 31 PASS — best epoch:', checkpoint['epoch'])
print('Best official mAP:', checkpoint['val_stats']['official_map'])
print('Geometry50:', final_source_metrics['geometry50'])
print('Fixed 160K:', milestone30(20))
