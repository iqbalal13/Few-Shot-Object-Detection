
# ============================================================
# STEP 30 — B1 ON A2, FRESH 160K
# EXACT A2 MANIFEST; ONLY GAUSSIAN TARGET V1.3 IS MODIFIED.
# ============================================================
import os, json, hashlib, random, time, math, gc
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Sampler
from google.colab import drive

EPOCHS, IMAGES_PER_EPOCH, STEPS_PER_EPOCH = 20, 8000, 1000
EVAL_EPOCHS, SCHEDULER_TMAX = (5, 10, 15, 20), 25
MANIFEST_SEED = int(SEED) + int(CONFIG['source_manifest_seed_offset'])
RUN_NAME = 'CenterNet_COCO80_B1_V13_ON_A2_FRESH_160K_BS8'
LOCKED_SHA = '9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f'

assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert len(train_dataset) == 118287 and len(val_dataset) == 5000
assert not train_dataset.augment and not val_dataset.augment
assert CONFIG['experiment_name'] == RUN_NAME
assert CONFIG['gaussian_radius_mode'] == 'v1_3_corrected'
assert CONFIG['gaussian_min_overlap'] == 0.7
assert CONFIG['wh_loss_type'] == 'hybrid_abs_relative'
assert (CONFIG['wh_absolute_fraction'], CONFIG['wh_relative_fraction']) == (0.75, 0.25)
assert (CONFIG['wh_relative_eps'], CONFIG['wh_relative_beta'],
        CONFIG['wh_relative_scale']) == (2.0, 0.20, 10.0)
assert criterion.wh_loss_type == 'hybrid_abs_relative'
assert (criterion.wh_absolute_fraction, criterion.wh_relative_fraction) == (0.75, 0.25)
assert (criterion.hm_weight, criterion.wh_weight, criterion.offset_weight) == (1., .1, 1.)
assert CONFIG['batch_size'] == 8 and CONFIG['max_objects'] == 128
assert CONFIG['source_images_total'] == 160000
assert CONFIG['source_epochs'] == 20 and CONFIG['source_images_per_epoch'] == 8000
assert CONFIG['source_images_per_class_per_epoch'] == 100
assert CONFIG['train_horizontal_flip_p'] == 0.0
assert not RUN_TINY_OVERFIT and not RUN_ENGINEERING_SANITY
assert int(gaussian_radius(12, 12, mode='v1_2_original')) == 1
assert int(gaussian_radius(12, 12, mode='v1_3_corrected')) == 0
assert int(gaussian_radius(12, 12)) == 0

# A2 files are NEVER overwritten by B1.
drive.mount('/content/drive', force_remount=False)
ROOT30 = '/content/drive/MyDrive'
A2_DIR30 = os.path.join(ROOT30, 'CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8')
A2_MANIFEST = os.path.join(A2_DIR30, 'balanced160k_manifest.jsonl')
assert os.path.isfile(A2_MANIFEST), 'Original A2 manifest missing; do not regenerate silently.'
with open(A2_MANIFEST, 'rb') as f:
    manifest_bytes = f.read()
source_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
assert source_manifest_sha256 == LOCKED_SHA, 'A2 manifest SHA256 mismatch'
source_manifest = [json.loads(line) for line in manifest_bytes.splitlines() if line]
assert len(source_manifest) == 160000

# Verify all manifest indices and per-epoch quotas.
image_ids30 = list(map(int, train_dataset.image_ids))
assert len(image_ids30) == len(train_dataset)
for epoch in range(1, EPOCHS + 1):
    rows = source_manifest[(epoch-1)*8000:epoch*8000]
    assert len(rows) == 8000
    quota = Counter(r['selected_category_id'] for r in rows)
    assert len(quota) == 80 and set(quota.values()) == {100}
    for j, r in enumerate(rows):
        assert int(r['epoch']) == epoch and int(r['position']) == j
        assert image_ids30[int(r['dataset_index'])] == int(r['image_id'])

FULL_CHECKPOINT_DIR = os.path.join(ROOT30, RUN_NAME)
os.makedirs(FULL_CHECKPOINT_DIR, exist_ok=True)
BEST_CHECKPOINT = os.path.join(FULL_CHECKPOINT_DIR, 'centernet_coco80_best.pth')
LAST_CHECKPOINT = os.path.join(FULL_CHECKPOINT_DIR, 'centernet_coco80_last.pth')
MANIFEST_FILE = os.path.join(FULL_CHECKPOINT_DIR, 'balanced160k_manifest.jsonl')
HISTORY_FILE = os.path.join(FULL_CHECKPOINT_DIR, 'training_history.json')

def milestone30(epoch):
    assert epoch in EVAL_EPOCHS
    return os.path.join(FULL_CHECKPOINT_DIR, f'epoch{epoch}_{epoch*8}k.pth')

def atomic_save30(obj, path):
    temp = path + '.tmp'
    torch.save(obj, temp)
    os.replace(temp, path)

if os.path.isfile(MANIFEST_FILE):
    with open(MANIFEST_FILE, 'rb') as f:
        assert f.read() == manifest_bytes, 'B1 manifest content changed'
else:
    temp = MANIFEST_FILE + '.tmp'
    with open(temp, 'wb') as f:
        f.write(manifest_bytes)
    os.replace(temp, MANIFEST_FILE)

class BalancedManifestSampler(Sampler):
    def __init__(self, rows):
        self.by_epoch = {e: [] for e in range(1, EPOCHS+1)}
        for r in rows:
            self.by_epoch[int(r['epoch'])].append(int(r['dataset_index']))
        assert all(len(v) == 8000 for v in self.by_epoch.values())
        self.epoch = 1

    def set_epoch(self, epoch):
        assert 1 <= epoch <= EPOCHS
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self.by_epoch[self.epoch])

    def __len__(self):
        return IMAGES_PER_EPOCH

resume = None
if os.path.isfile(LAST_CHECKPOINT):
    resume = torch.load(LAST_CHECKPOINT, map_location='cpu', weights_only=False)
    assert resume['experiment_name'] == RUN_NAME
    assert resume['source_manifest_sha256'] == LOCKED_SHA
    assert resume['source_images_total'] == 160000
    assert 1 <= int(resume['epoch']) <= 20
    assert len(resume['history']) == int(resume['epoch'])
    for key in ('gaussian_radius_mode', 'gaussian_min_overlap', 'wh_loss_type',
                'wh_absolute_fraction', 'wh_relative_fraction', 'wh_relative_eps',
                'wh_relative_beta', 'wh_relative_scale', 'hm_loss_weight',
                'wh_loss_weight', 'offset_loss_weight', 'learning_rate',
                'weight_decay', 'batch_size', 'source_epochs',
                'source_images_per_epoch', 'source_manifest_seed_offset',
                'input_size', 'output_stride', 'num_classes',
                'train_horizontal_flip_p'):
        assert resume['config'][key] == CONFIG[key], f'Resume mismatch: {key}'
    assert os.path.isfile(BEST_CHECKPOINT) or int(resume['epoch']) < 5
elif os.path.isfile(BEST_CHECKPOINT):
    raise RuntimeError('BEST exists but LAST is missing; inspect B1 folder.')

start_epoch = int(resume['epoch']) if resume is not None else 0
if start_epoch == EPOCHS:
    print('B1 already completed: no optimizer updates.')
else:
    sampler30 = BalancedManifestSampler(source_manifest)
    sampler30.set_epoch(start_epoch + 1)
    full_train_loader, full_train_generator = build_train_loader(
        MANIFEST_SEED, sampler=sampler30, encoded=True)
    source_val_loader, source_val_generator = build_val_loader(
        SEED + 3301, encoded=True)
    assert len(full_train_loader) == STEPS_PER_EPOCH
    assert len(source_val_loader) == 625

    # Confirm that actual DataLoader workers produce V1.3 encoded targets.
    _, smoke_raw, smoke_encoded = next(iter(full_train_loader))
    assert smoke_encoded['heatmap'].shape == (8, 80, 160, 160)
    for j in range(2):
        expected = build_centernet_target(smoke_raw[j], gaussian_radius_mode='v1_3_corrected')
        for key in ('heatmap', 'wh', 'offset', 'indices', 'mask'):
            assert torch.equal(smoke_encoded[key][j], expected[key]), f'Worker mismatch: {key}'
    del smoke_raw, smoke_encoded
    print('B1 worker target check: PASSED')

    if resume is None:
        random.seed(MANIFEST_SEED)
        np.random.seed(MANIFEST_SEED)
        torch.manual_seed(MANIFEST_SEED)
        torch.cuda.manual_seed_all(MANIFEST_SEED)

    model30 = CenterNetResNet101(
        num_classes=80, pretrained_backbone=(resume is None)).to(device)
    optimizer30 = build_source_optimizer(model30)
    scheduler30 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer30, T_max=SCHEDULER_TMAX)
    history30, best_map30, best_val_loss30 = [], -1.0, float('inf')

    if resume is not None:
        model30.load_state_dict(resume['model_state_dict'], strict=True)
        optimizer30.load_state_dict(resume['optimizer_state_dict'])
        scheduler30.load_state_dict(resume['scheduler_state_dict'])
        assert scheduler30.last_epoch == start_epoch
        history30 = list(resume['history'])
        best_map30 = float(resume['best_map'])
        best_val_loss30 = float(resume['best_val_loss'])
        rng = resume['rng_state']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch_cpu'].cpu())
        torch.cuda.set_rng_state_all(rng['torch_cuda'])
        print('Resume B1 from epoch', start_epoch)
    else:
        print('Fresh B1 initialization from ImageNet ResNet-101')

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        sampler30.set_epoch(epoch)
        full_train_generator.manual_seed(MANIFEST_SEED + epoch)
        t0 = time.perf_counter()
        train_result = train_centernet_epoch(model30, full_train_loader, optimizer30)
        assert train_result['batches'] == STEPS_PER_EPOCH
        duration = time.perf_counter() - t0
        validation, selected = None, False

        if epoch in EVAL_EPOCHS:
            validation = evaluate_centernet(
                model30, source_val_loader, max_batches=None,
                score_threshold=CONFIG['score_threshold'],
                iou_threshold=CONFIG['iou_threshold'],
                compute_official_ap=True)
            current_map = float(validation['official_map'])
            current_loss = float(validation['loss'])
            assert math.isfinite(current_map) and math.isfinite(current_loss)
            selected = (current_map > best_map30 + 1e-8
                        or (abs(current_map - best_map30) <= 1e-8
                            and current_loss < best_val_loss30))
            print(f'epoch={epoch} | mAP={current_map:.6f} '
                  f'AP50={float(validation["official_ap50"]):.6f} '
                  f'AP75={float(validation["official_ap75"]):.6f}')

        history30.append({
            'epoch': epoch,
            'exposures': epoch * 8000,
            'train_stats': train_result,
            'val_stats': validation,
            'train_seconds': duration,
            'lr_before_scheduler_step': optimizer30.param_groups[0]['lr']
        })

        if epoch in EVAL_EPOCHS:
            fixed = {
                'model_state_dict': model30.state_dict(), 'epoch': epoch,
                'experiment_name': RUN_NAME,
                'source_manifest_sha256': LOCKED_SHA,
                'source_images_total': 160000,
                'actual_images_seen_at_checkpoint': epoch * 8000,
                'val_stats': validation, 'config': dict(CONFIG)
            }
            atomic_save30(fixed, milestone30(epoch))
            if selected:
                best_map30, best_val_loss30 = current_map, current_loss
                atomic_save30(fixed, BEST_CHECKPOINT)
                print('New B1 best checkpoint:', epoch)

        scheduler30.step()
        atomic_save30({
            'model_state_dict': model30.state_dict(),
            'optimizer_state_dict': optimizer30.state_dict(),
            'scheduler_state_dict': scheduler30.state_dict(),
            'epoch': epoch, 'experiment_name': RUN_NAME,
            'source_manifest_sha256': LOCKED_SHA,
            'source_images_total': 160000,
            'best_map': best_map30,
            'best_val_loss': best_val_loss30,
            'history': history30, 'config': dict(CONFIG),
            'rng_state': {
                'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all()
            }
        }, LAST_CHECKPOINT)
        temp = HISTORY_FILE + '.tmp'
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump(history30, f, indent=2, allow_nan=False)
        os.replace(temp, HISTORY_FILE)
        print(f'B1 epoch {epoch:02d}/{EPOCHS} '
              f'loss={train_result["loss"]:.5f} elapsed={duration/60:.1f} min')

    del model30, optimizer30, scheduler30
    gc.collect()
    torch.cuda.empty_cache()

last30 = torch.load(LAST_CHECKPOINT, map_location='cpu', weights_only=False)
assert last30['epoch'] == EPOCHS and len(last30['history']) == EPOCHS
assert last30['experiment_name'] == RUN_NAME
assert last30['source_manifest_sha256'] == LOCKED_SHA
assert last30['config']['gaussian_radius_mode'] == 'v1_3_corrected'
assert last30['config']['wh_loss_type'] == 'hybrid_abs_relative'
assert last30['config']['wh_absolute_fraction'] == 0.75
assert last30['config']['wh_relative_fraction'] == 0.25
assert all(row['train_stats']['batches'] == 1000 for row in last30['history'])
assert os.path.isfile(BEST_CHECKPOINT)
assert all(os.path.isfile(milestone30(e)) for e in EVAL_EPOCHS)
if 'source_val_loader' not in globals():
    source_val_loader, source_val_generator = build_val_loader(SEED + 3301, encoded=True)
print('STEP 30 PASS | B1 hybrid WH + Gaussian V1.3 | 160K exposures')
print('B1 checkpoints:', FULL_CHECKPOINT_DIR)
print('B1 best mAP:', last30['best_map'])
