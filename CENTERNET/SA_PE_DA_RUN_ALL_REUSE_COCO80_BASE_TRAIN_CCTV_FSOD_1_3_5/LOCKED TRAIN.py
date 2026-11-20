%%writefile sape_da/locked_train.py
"""LOCKED COCO80 Stage1 (exact one-epoch option) & Stage2 (FiLM K-shots).

Stage1 saves full optimizer/scaler/model progress; COCO-Val 5000 runs ONCE later
outside the fixed one-epoch optimizer-step budget. Checkpoint deliberately cannot load old
one-class SA-PE-DA weights.
"""
from __future__ import annotations
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .model import SAPEDACenterNet
from .data import COCO80Dataset, PersonDetectionDataset
from .shared import resolve_support_ids
from .train import seed_all, center_loss
from .evaluate import evaluate_person

ARCHITECTURE = 'SA-PE-DA-COCO80-LOCKED-v1'


def _atomic_torch_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    torch.save(obj,tmp);os.replace(tmp,path)


def _atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');os.replace(tmp,path)


def load_locked_checkpoint(path,model,optimizer=None,scaler=None):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if ckpt.get('architecture')!=ARCHITECTURE:
        raise RuntimeError('Wrong checkpoint architecture: expected LOCKED COCO-80, NOT prior person-only version')
    if ckpt['model']['base.hm.2.weight'].shape[0]!=80:
        raise RuntimeError('Checkpoint is not 80-class CenterNet')
    model.load_state_dict(ckpt['model'],strict=True)
    if optimizer is not None and 'optimizer' in ckpt:optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:scaler.load_state_dict(ckpt['scaler'])
    return ckpt


def save_locked_checkpoint(path,model,phase,step,metadata,optimizer=None,scaler=None):
    obj={'architecture':ARCHITECTURE,'phase':phase,'step':int(step),
         'model':model.state_dict(),'metadata':metadata}
    if optimizer is not None:obj['optimizer']=optimizer.state_dict()
    if scaler is not None:obj['scaler']=scaler.state_dict()
    _atomic_torch_save(obj,path)
    return path


def _gpu(device):
    if device.type!='cuda':
        raise RuntimeError('Training budget requires GPU. Enable GPU A100 in Colab Runtime settings.')


def fit_source_budget(model,annotation_json,image_root,ordered_ids,manifest_hash,save_dir,
                      max_train_seconds=3600,batch_size=1,workers=2,
                      fusion_lr=1e-4,head_lr=1e-5,checkpoint_every_seconds=600,
                      enable_amp=True, max_optimizer_steps=None):
    """Exactly one source-training budget for ALL 80 classes, NOT per category.

    Uses real monotonic training time; saves final checkpoint and resumable
    rolling checkpoint (including AdamW/scaler). Sample exposure and category
    coverage are counted from batches the model ACTUALLY sees.
    """
    if len(ordered_ids)!=8000 or len(set(ordered_ids))!=8000:
        raise ValueError('Stage1 source requires 8,000 unique shared COCO image IDs')
    if max_optimizer_steps is not None:
        if batch_size!=1 or max_optimizer_steps!=len(ordered_ids):
            raise ValueError('One-epoch run requires batch_size=1 and exactly len(ordered_ids) optimizer steps')
    seed_all(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');_gpu(device)
    model.to(device);model.freeze_phase('source')
    groups=[{'params':list(model.fusion.parameters()),'lr':fusion_lr},
        {'params':[p for head in (model.base.hm,model.base.wh,model.base.reg)
                   for p in head.parameters()], 'lr':head_lr}]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=enable_amp)
    root=Path(save_dir);root.mkdir(parents=True,exist_ok=True)
    last=root/'source80_last_resumable.pth'
    final=root/'source80_final.pth'
    complete=root/'source80_training_complete.json'
    if complete.exists() and final.exists():
        ck=load_locked_checkpoint(final,model)
        if ck['metadata'].get('manifest_hash')!=manifest_hash:raise RuntimeError('Checkpoint manifest hash differs')
        if ck['metadata'].get('max_optimizer_steps')!=max_optimizer_steps:
            raise RuntimeError('Completed checkpoint was produced using another step budget')
        print('SKIP Stage1: completed COCO-80 checkpoint loaded from Google Drive.')
        return ck['metadata']
    elapsed_base=0.;steps=0;cursor=0;passes=0;exposures=[0]*80
    unique_seen=set()
    if last.exists():
        ck=load_locked_checkpoint(last,model,optimizer,scaler)
        meta=ck['metadata']
        if meta.get('manifest_hash')!=manifest_hash:raise RuntimeError('Source manifest changed; cannot resume')
        if meta.get('max_optimizer_steps')!=max_optimizer_steps:
            raise RuntimeError('Resumable checkpoint has a different step budget')
        elapsed_base=float(meta['elapsed_train_seconds']);steps=int(ck['step'])
        cursor=int(meta['cursor']);passes=int(meta['completed_passes'])
        exposures=[int(x) for x in meta['class_instances_exposed']]
        unique_seen=set(int(x) for x in meta['unique_image_ids_seen'])
        print(f'RESUME Stage1: {elapsed_base:.1f}s used, step={steps}, cursor={cursor}')
    ds=COCO80Dataset(annotation_json,image_root,ordered_ids,train=True)
    started=time.monotonic();last_saved=elapsed_base
    while True:
        if max_optimizer_steps is not None and steps>=max_optimizer_steps:break
        elapsed=elapsed_base+time.monotonic()-started
        if elapsed>=max_train_seconds:break
        if cursor>=len(ds):cursor=0;passes+=1
        remaining=Subset(ds,range(cursor,len(ds)))
        loader=DataLoader(remaining,batch_size=batch_size,shuffle=False,num_workers=workers,
                          pin_memory=True,persistent_workers=workers>0)
        model.train();iterator=iter(loader)
        for batch in iterator:
            if max_optimizer_steps is not None and steps>=max_optimizer_steps:break
            elapsed=elapsed_base+time.monotonic()-started
            if elapsed>=max_train_seconds:break
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v
                   for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=enable_amp):
                outputs=model(batch['image'])
                loss,stats=center_loss(outputs,batch)
            if not math.isfinite(stats['total']):raise FloatingPointError(f'Non-finite source loss at step {steps}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            scaler.step(optimizer);scaler.update()
            steps+=1;cursor+=len(batch['image'])
            new_counts=batch['class_counts'].sum(0).detach().cpu().tolist()
            exposures=[a+int(b) for a,b in zip(exposures,new_counts)]
            unique_seen.update(int(x) for x in batch['image_id'].detach().cpu().tolist())
            if steps%100==0:
                elapsed=elapsed_base+time.monotonic()-started
                print(f'COCO-80 step={steps} elapsed={elapsed:.1f}s images_unique={len(unique_seen)} '
                      f'classes_seen={sum(x>0 for x in exposures)}/80 loss={stats["total"]:.4f}')
            elapsed=elapsed_base+time.monotonic()-started
            if elapsed-last_saved>=checkpoint_every_seconds:
                meta={'elapsed_train_seconds':elapsed,'cursor':cursor,'completed_passes':passes,
                      'class_instances_exposed':exposures,'unique_image_ids_seen':sorted(unique_seen),
                      'manifest_hash':manifest_hash,'training_budget_seconds':max_train_seconds,
                      'optimizer_updates':steps,'source_classes':80,
                      'max_optimizer_steps':max_optimizer_steps}
                save_locked_checkpoint(last,model,'source',steps,meta,optimizer,scaler)
                elapsed_base=elapsed;started=time.monotonic();last_saved=elapsed
                print('Resumable Stage1 checkpoint saved:',last)
            if elapsed>=max_train_seconds:break
        if cursor>=len(ds):cursor=0;passes+=1
        del iterator, loader
    final_elapsed=elapsed_base+time.monotonic()-started
    meta={'elapsed_train_seconds':final_elapsed,'cursor':cursor,
      'completed_passes':passes,'class_instances_exposed':exposures,
      'unique_image_ids_seen':sorted(unique_seen),'source_classes_seen':sum(x>0 for x in exposures),
      'manifest_hash':manifest_hash,'training_budget_seconds':max_train_seconds,
      'optimizer_updates':steps,'source_classes':80,
                      'max_optimizer_steps':max_optimizer_steps}
    # Never mark the one-epoch run complete if interrupted before 8,000 updates.
    if max_optimizer_steps is not None:
        if steps!=max_optimizer_steps or len(unique_seen)!=len(ordered_ids):
            save_locked_checkpoint(last,model,'source',steps,meta,optimizer,scaler)
            raise RuntimeError(f'One epoch incomplete: {steps}/{max_optimizer_steps} steps and '
                               f'{len(unique_seen)}/{len(ordered_ids)} unique images. '
                               'Resume Stage 1 from the saved rolling checkpoint.')
    # Final checkpoint is the state at the end of exactly one epoch (not validation-best).
    save_locked_checkpoint(final,model,'source',steps,meta)
    save_locked_checkpoint(last,model,'source',steps,meta,optimizer,scaler)
    _atomic_json(meta,complete)
    print(f'Stage1 complete: {final_elapsed:.1f}s, {steps} steps, '
          f'{len(unique_seen)} unique images; {meta["source_classes_seen"]}/80 COCO classes actually exposed.')
    if meta['source_classes_seen']!=80:
        print('WARNING: not all 80 classes were seen during optimizer steps; disclose and reconsider run.')
    return meta


def fit_target_ap50(model,train_json,train_images,val_json,val_images,
                    support_manifest,shots,seed,save_root,source_checkpoint,
                    steps=100,valid_every=20,lr=1e-3,reg_lambda=1e-3):
    """128-param FiLM adaptation; ALL 100 CCTV-Val images for checkpoint AP50 selection."""
    seed_all(seed)
    ids=resolve_support_ids(train_json,support_manifest,shots)
    root=Path(save_root)/f'shot_{shots}'/f'seed_{seed}'
    root.mkdir(parents=True,exist_ok=True)
    bestfile=root/'target_best.pth';done=root/'complete.json'
    if bestfile.exists() and done.exists():
        print('SKIP completed target:',bestfile)
        return json.loads(done.read_text())
    source=load_locked_checkpoint(source_checkpoint,model)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');_gpu(device)
    model.to(device);n=model.freeze_phase('target')
    assert n==128,f'Expected 128 FiLM-only trainable parameters; got {n}'
    ds=PersonDetectionDataset(train_json,train_images,selected_ids=ids,train=True)
    loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=0)
    iterator=iter(loader)
    optimizer=torch.optim.AdamW([model.adapter.g,model.adapter.b],lr=lr,weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda')
    history=[];best_ap=-1.;best_step=0;start_step=1
    support_record={'shots':shots,'seed':seed,'support_image_ids':ids,
         'support_files':json.loads(Path(support_manifest).read_text())['shots'][str(shots)]}
    support_file=root/'support_manifest_used.json'
    if support_file.exists():
        previous=json.loads(support_file.read_text())
        if previous.get('support_files')!=support_record['support_files']:
            raise RuntimeError('CCTV support manifest changed; cannot reuse/resume target checkpoint')
    else:_atomic_json(support_record,support_file)
    historyfile=root/'history.json'
    rolling=root/'target_last_resumable.pth'
    if rolling.exists():
        old=load_locked_checkpoint(rolling,model,optimizer,scaler)
        oldmeta=old['metadata']
        if old.get('phase')!='target' or int(oldmeta.get('shot',-1))!=shots or int(oldmeta.get('seed',-1))!=seed:
            raise RuntimeError('Incorrect rolling target checkpoint for selected shot/seed')
        if oldmeta.get('support_files')!=support_record['support_files']:
            raise RuntimeError('Rolling checkpoint support filenames have changed')
        start_step=int(old['step'])+1
        if historyfile.exists():
            history=json.loads(historyfile.read_text())
        best_ap=float(oldmeta.get('best_ap50',-1.))
        best_step=int(oldmeta.get('best_step',0))
        print(f'RESUME CCTV {shots}-shot seed={seed} from step={start_step}; '
              f'best_val_AP50={best_ap:.4f}')
    for step in range(start_step,steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            pred=model(batch['image'])
            # CCTV supervision ONLY for person channel 0; do not penalize 79 unrelated classes.
            pred_person={**pred,'hm':pred['hm'][:,0:1]}
            loss,stats=center_loss(pred_person,batch,film=model.adapter,reg_lambda=reg_lambda)
        if not math.isfinite(stats['total']):raise FloatingPointError(f'Nonfinite target loss at step {step}')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([model.adapter.g,model.adapter.b],5.)
        scaler.step(optimizer);scaler.update()
        if step%valid_every==0 or step==steps:
            # Source head 80 retained, person decoder selects exactly channel 0.
            metric=evaluate_person(model,val_json,val_images,
              result_path=root/f'val_predictions_step{step:03d}.json',
              score_threshold=.001,ncacc_threshold=.5,progress_every=0)
            ap=float(metric['AP50'])
            history.append({'step':step,'train':stats,'val_AP50':ap,
                            'val_AP50_95':metric['AP50:95'],'val_NCAcc50':metric['NCAcc50']})
            if ap>best_ap:
                best_ap=ap;best_step=step
                save_locked_checkpoint(bestfile,model,'target',step,
                  {'shot':shots,'seed':seed,'val_AP50':ap,'best_step':step,
                   'source_checkpoint':str(source_checkpoint)})
            # Persist complete optimizer+AMP state only after validation selection,
            # enabling restart at most ~20 steps back rather than restarting shot.
            save_locked_checkpoint(rolling,model,'target',step,
              {'shot':shots,'seed':seed,'source_checkpoint':str(source_checkpoint),
               'support_manifest':str(support_manifest),
               'support_files':support_record['support_files'],
               'val_AP50':ap,'best_ap50':best_ap,'best_step':best_step},
               optimizer=optimizer,scaler=scaler)
            _atomic_json(history,historyfile)
            print(f'CCTV {shots}-shot seed={seed} step={step}/{steps} val_AP50={ap:.4f} best={best_ap:.4f}')
    summary={'shot':shots,'seed':seed,'source_checkpoint':str(source_checkpoint),
             'best_step':best_step,'best_val_AP50':best_ap,'optimizer_steps':steps,
             'support_ids':ids}
    _atomic_json(summary,done)
    return summary
