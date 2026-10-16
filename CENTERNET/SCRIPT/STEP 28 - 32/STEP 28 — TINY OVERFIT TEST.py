# ============================================================
# STEP 28 — INDEPENDENT TINY-OVERFIT DIAGNOSTIC + HELPERS
# For Balanced80K V1.3. Does NOT touch main source checkpoint.
# ============================================================
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

print('=' * 70)
print('STEP 28 — TINY OVERFIT & SOURCE HELPERS')
print('=' * 70)
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
device = torch.device('cuda')

# Original helpers used by subsequent steps.
def set_backbone_bn_eval(model):
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def build_fresh_source_model():
    return CenterNetResNet101(
        num_classes=CONFIG['num_classes'],
        pretrained_backbone=True
    ).to(device)

def build_source_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

# Separate switch: does not trigger Step 29 short training.
RUN_TINY_OVERFIT = bool(CONFIG.get('run_tiny_overfit_diagnostic', True))
RUN_ENGINEERING_SANITY = False
CONFIG['run_engineering_sanity'] = False

if RUN_TINY_OVERFIT:
    # Do NOT use augmented train_dataset here: the same 8 images must
    # have exactly the same pixels + boxes on each optimization step.
    tiny_source = CenterNetCOCODataset(
        CONFIG['train_images'], CONFIG['train_annotations'],
        input_size=CONFIG['input_size'],
        augment=False, flip_probability=0.0
    )
    tiny_indices = []
    for idx, image_id in enumerate(tiny_source.image_ids):
        anns = tiny_source.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        if 1 <= len(anns) <= 15:
            tiny_indices.append(idx)
        if len(tiny_indices) == 8:
            break
    assert len(tiny_indices) == 8
    tiny_loader = DataLoader(
        Subset(tiny_source, tiny_indices),
        batch_size=8, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=centernet_collate_fn
    )
    tiny_images, tiny_targets = next(iter(tiny_loader))
    tiny_images = tiny_images.to(device)
    tiny_encoded = build_batch_centernet_targets(tiny_targets, device=device)

    torch.manual_seed(SEED + 2801)
    torch.cuda.manual_seed_all(SEED + 2801)
    tiny_model = build_fresh_source_model()
    tiny_optimizer = build_source_optimizer(tiny_model)

    TINY_STEPS = 200
    tiny_loss_history = []
    for i in range(1, TINY_STEPS + 1):
        tiny_model.train()
        set_backbone_bn_eval(tiny_model)
        tiny_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            tiny_out = tiny_model(tiny_images)
        # Criterion computes loss in FP32 as in original training.
        tiny_losses = criterion(tiny_out, tiny_encoded)
        tiny_loss = tiny_losses['loss_total']
        if not bool(torch.isfinite(tiny_loss)):
            raise FloatingPointError(f'Tiny loss invalid at step {i}')
        tiny_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            tiny_model.parameters(), max_norm=CONFIG['grad_clip_norm'],
            error_if_nonfinite=True
        )
        tiny_optimizer.step()
        tiny_loss_history.append(float(tiny_loss.detach().cpu()))
        if i == 1 or i % 25 == 0:
            print(f'Tiny step {i:03}/{TINY_STEPS}: loss={tiny_loss_history[-1]:.4f}')

    first = float(np.mean(tiny_loss_history[:20]))
    last = float(np.mean(tiny_loss_history[-20:]))
    print(f'Tiny mean loss first20={first:.5f} last20={last:.5f}')
    print(f'Tiny loss reduction={(1 - last/max(first, 1e-8))*100:.1f}%')
    if last >= first * 0.5:
        print('WARNING: tiny-overfit improvement <50%. Investigate targets, loss and gradients.')
    else:
        print('Tiny-overfit loss-reduction diagnostic PASSED.')
    del tiny_model, tiny_optimizer, tiny_images, tiny_encoded, tiny_out, tiny_losses, tiny_loss
    del tiny_loader, tiny_source, tiny_targets
    torch.cuda.empty_cache()
else:
    print('Tiny-overfit diagnostic skipped; helpers still available.')

print('STEP 28 PASSED (source training remains separate)')
