%%writefile sape_da/train.py
"""Two-phase training and CenterNet Objects-as-Points losses for SA-PE-DA.
Source: fusion only; target: FiLM only. Original backbone, neck and heads frozen.
"""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .model import SAPEDACenterNet
from .data import PersonDetectionDataset, sample_cctv_support_ids


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _gather_at_indices(fmap,indices):
    b,c,h,w=fmap.shape
    fmap=fmap.permute(0,2,3,1).contiguous().view(b,-1,c)
    return fmap.gather(dim=1,index=indices.unsqueeze(-1).expand(-1,-1,c))


def focal_heatmap_loss(logits,gt):
    """Penalty-reduced CenterNet/CornerNet heatmap focal loss.
    Full precision accumulation to avoid unstable logarithms with AMP.
    """
    pred=logits.float().sigmoid().clamp(1e-4,1-1e-4)
    gt=gt.float()
    pos=(gt==1).float();neg=(gt<1).float()
    pos_loss=(pred.log()*((1-pred)**2)*pos).sum()
    neg_loss=((1-pred).log()*(pred**2)*((1-gt)**4)*neg).sum()
    n=pos.sum()
    return -(pos_loss+neg_loss)/n if n.item()>0 else -neg_loss


def center_loss(pred,batch,film=None,reg_lambda=0.0):
    hm=focal_heatmap_loss(pred['hm'],batch['hm'])
    mask=batch['mask'].float().unsqueeze(-1)
    n=mask.sum().clamp(min=1)
    wh_pred=_gather_at_indices(pred['wh'],batch['ind'])
    off_pred=_gather_at_indices(pred['reg'],batch['ind'])
    wh=(torch.abs(wh_pred-batch['wh'])*mask).sum()/n
    off=(torch.abs(off_pred-batch['reg'])*mask).sum()/n
    regularizer=film.regularizer() if (film is not None and reg_lambda>0) else hm.new_zeros(())
    total=hm+0.1*wh+off+reg_lambda*regularizer
    return total,{'total':float(total.detach()),'hm':float(hm.detach()),
                  'wh':float(wh.detach()),'off':float(off.detach()),
                  'regularizer':float(regularizer.detach())}


def save_phase_checkpoint(path,model,phase,epoch,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'phase':phase,'epoch':epoch,
                'metadata':extra or {},'architecture':'SA-PE-DA CenterNet ResDCN101'},path)
    return path


def load_phase_checkpoint(path,model):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if obj.get('architecture')!='SA-PE-DA CenterNet ResDCN101':
        raise ValueError('Not a compatible SA-PE-DA checkpoint')
    model.load_state_dict(obj['model'],strict=True)
    return obj


def _loader(ds,batch_size,shuffle,workers):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),drop_last=False,
                      persistent_workers=(workers>0))


def run_epoch(model,loader,device,optimizer=None,amp=True,reg_lambda=0.0,
              max_batches=None,grad_accum=1):
    train=optimizer is not None
    model.train(train)
    scaler=torch.amp.GradScaler('cuda',enabled=(amp and device.type=='cuda' and train)) if train else None
    totals=[]
    if train:optimizer.zero_grad(set_to_none=True)
    steps = min(len(loader),max_batches) if max_batches else len(loader)
    if steps==0:raise ValueError('Empty loader')
    for iteration,batch in enumerate(loader):
        if max_batches is not None and iteration>=max_batches:break
        batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v
               for k,v in batch.items()}
        with torch.set_grad_enabled(train),torch.autocast(device_type=device.type,enabled=(amp and device.type=='cuda')):
            pred=model(batch['image'])
            loss,stats=center_loss(pred,batch,film=model.adapter if reg_lambda>0 else None,
                                   reg_lambda=reg_lambda)
        if not math.isfinite(stats['total']):
            raise FloatingPointError(f'Non-finite loss at iteration {iteration}: {stats}')
        if train:
            scaler.scale(loss/grad_accum).backward()
            boundary=((iteration+1)%grad_accum==0 or iteration+1==steps)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        totals.append(stats)
    return {k:float(np.mean([d[k] for d in totals])) for k in totals[0]}


def fit_source(model,annotation_json,image_dir,selected_ids,save_dir,
               epochs=5,batch_size=2,lr=1e-4,workers=2,grad_accum=2,
               val_annotation=None,val_image_dir=None,val_ids=None,
               max_train_batches=None,max_val_batches=32):
    """Trains only fusion and eta using COCO-person subset.
    Source val loss chooses best ckpt; report mAP separately with evaluate_person.
    """
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); n=model.freeze_phase('source')
    assert n>0
    train_ds=PersonDetectionDataset(annotation_json,image_dir,selected_ids=selected_ids,train=True)
    train_loader=_loader(train_ds,batch_size,True,workers)
    val_loader=None
    if val_annotation:
        val_ds=PersonDetectionDataset(val_annotation,val_image_dir,selected_ids=val_ids,train=False)
        val_loader=_loader(val_ds,batch_size,False,workers)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    directory=Path(save_dir);directory.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[]
    for epoch in range(1,epochs+1):
        train_stats=run_epoch(model,train_loader,device,optimizer,max_batches=max_train_batches,
                              grad_accum=grad_accum)
        val_stats=run_epoch(model,val_loader,device,max_batches=max_val_batches) if val_loader else None
        score=val_stats['total'] if val_stats else train_stats['total']
        info={'epoch':epoch,'train':train_stats,'val':val_stats}
        history.append(info);print(json.dumps(info))
        save_phase_checkpoint(directory/'source_last.pth',model,'source',epoch)
        if score<best:
            best=score;save_phase_checkpoint(directory/'source_best.pth',model,'source',epoch)
        (directory/'source_history.json').write_text(json.dumps(history,indent=2))
    return history


def fit_target(model,cctv_train_json,cctv_train_images,save_dir,shots,seed=42,
               steps=100,lr=1e-3,batch_size=1,reg_lambda=1e-3,
               val_json=None,val_images=None,val_steps=20,
               validation_every=20):
    """FiLM-only training; 1/3/5 means fully annotated CCTV IMAGES, not instances.
    Always call with a fresh SOURCE checkpoint for each shot and seed (no carry-over).
    Val is separately labeled data and its label budget must be disclosed in thesis.
    """
    seed_all(seed)
    ids=sample_cctv_support_ids(cctv_train_json,shots,seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device);n=model.freeze_phase('target')
    assert n==128, f'Unexpected FiLM parameter count {n}; expected 128'
    ds=PersonDetectionDataset(cctv_train_json,cctv_train_images,selected_ids=ids,train=True)
    loader=_loader(ds,batch_size,True,workers=0)
    val_loader=None
    if val_json:
        vds=PersonDetectionDataset(val_json,val_images,train=False)
        val_loader=_loader(vds,batch_size,False,workers=0)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,
                            weight_decay=0.0)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    directory=Path(save_dir)/f'shot_{shots}'/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'support_ids.json').write_text(json.dumps({'shots':shots,'seed':seed,'image_ids':ids},indent=2))
    history=[];best=float('inf');iterator=iter(loader)
    for step in range(1,steps+1):
        try:batch=next(iterator)
        except StopIteration:
            iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        model.train();optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            out=model(batch['image'])
            loss,stats=center_loss(out,batch,model.adapter,reg_lambda)
        if not math.isfinite(stats['total']):raise FloatingPointError(stats)
        scaler.scale(loss).backward();scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_([model.adapter.g,model.adapter.b],5)
        scaler.step(optim);scaler.update()
        if step%validation_every==0 or step==steps:
            val=run_epoch(model,val_loader,device,reg_lambda=reg_lambda,max_batches=val_steps) if val_loader else None
            entry={'step':step,'train':stats,'val':val}
            print(entry);history.append(entry)
            score=val['total'] if val is not None else stats['total']
            save_phase_checkpoint(directory/'target_last.pth',model,'target',step,
                                  {'shot_ids':ids,'shot':shots,'seed':seed})
            if score<best:
                best=score;save_phase_checkpoint(directory/'target_best.pth',model,'target',step,
                                                {'shot_ids':ids,'shot':shots,'seed':seed})
            (directory/'history.json').write_text(json.dumps(history,indent=2))
    return history,ids
