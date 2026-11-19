%%writefile sape_da/evaluate.py
"""Common evaluation primitives for COCO-80 source and CCTV person target.

mAP50 uses official pycocotools COCOeval; NCAcc50=TP/(TP+FP+FN),
score threshold 0.50 and 1:1 IoU>=.5. Latency is batch1 on REAL CCTV frames.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from .data import load_coco_json,letterbox,invert_letterbox_xyxy
from .model import decode_detections,decode_coco80


def _coco_eval(coco,results,ids,category_ids,output_path):
    from pycocotools.cocoeval import COCOeval
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results))
    if not results:
        return None,{'AP50:95':0.,'AP50':0.,'AP75':0.}
    pred=coco.loadRes(str(out));ev=COCOeval(coco,pred,'bbox')
    ev.params.imgIds=list(map(int,ids));ev.params.catIds=list(map(int,category_ids))
    ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    return ev,{'AP50:95':float(ev.stats[0]),'AP50':float(ev.stats[1]),
               'AP75':float(ev.stats[2])}


def _iou_xywh(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1])
    x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=max(0,a[2])*max(0,a[3])+max(0,b[2])*max(0,b[3])-inter
    return inter/union if union>0 else 0.


def ncacc_iou50(coco,image_ids,person_cat,results,score_threshold=.5):
    """Fixed pre-declared policy, one-to-one greedy matching, TP/(TP+FP+FN)."""
    indexed=defaultdict(list)
    for p in results:
        if int(p['category_id'])==person_cat and p['score']>=score_threshold:
            indexed[int(p['image_id'])].append(p)
    tp=fp=fn=0
    for iid in image_ids:
        gt=[a['bbox'] for a in coco.loadAnns(coco.getAnnIds(imgIds=[int(iid)],catIds=[person_cat],iscrowd=False))]
        used=set()
        for pred in sorted(indexed[int(iid)],key=lambda x:-x['score']):
            pairs=[(_iou_xywh(pred['bbox'],b),j) for j,b in enumerate(gt) if j not in used]
            if pairs:
                iou,j=max(pairs)
            else:
                iou,j=0.,-1
            if iou>=.5: tp+=1;used.add(j)
            else:fp+=1
        fn+=len(gt)-len(used)
    return {'TP':tp,'FP':fp,'FN':fn,'NCAcc50':tp/max(1,tp+fp+fn),
            'precision50':tp/max(1,tp+fp),'recall50':tp/max(1,tp+fn),
            'ncacc_formula':'TP/(TP+FP+FN)', 'ncacc_confidence_threshold':float(score_threshold)}


@torch.inference_mode()
def evaluate_person(model,annotation_json,image_root,result_path,device=None,
                    score_threshold=.001, ncacc_threshold=.5,progress_every=25):
    from pycocotools.coco import COCO
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device);model.eval()  # never mutate requires_grad during mid-training validation
    coco=COCO(str(annotation_json))
    person=[int(c['id']) for c in coco.dataset['categories'] if c['name'].lower()=='person']
    if len(person)!=1:raise ValueError('CCTV must have exactly one person category')
    cid=person[0];ids=sorted(coco.getImgIds());results=[]
    for ix,iid in enumerate(ids,1):
        im=coco.loadImgs([iid])[0];bgr=cv2.imread(str(Path(image_root)/im['file_name']))
        if bgr is None:raise FileNotFoundError(f'Missing CCTV image: {im["file_name"]}')
        x,_,meta=letterbox(bgr,[],1280,736,720)
        raw=model(x.unsqueeze(0).to(device))
        decoded=decode_detections(raw,736,1280,score_threshold=score_threshold,k=100)[0].cpu().numpy()
        if len(decoded):
            orig=invert_letterbox_xyxy(decoded[:,:4],meta)
            for box,conf in zip(orig,decoded[:,4]):
                x1,y1,x2,y2=map(float,box)
                if x2<=x1 or y2<=y1:continue
                results.append({'image_id':int(iid),'category_id':cid,
                  'bbox':[x1,y1,x2-x1,y2-y1],'score':float(conf)})
        if progress_every and ix%progress_every==0:print(f'CCTV evaluation {ix}/{len(ids)}')
    ev,metrics=_coco_eval(coco,results,ids,[cid],result_path)
    metrics.update(ncacc_iou50(coco,ids,cid,results,ncacc_threshold))
    metrics.update({'images':len(ids),'person_predictions':len(results)})
    return metrics


@torch.inference_mode()
def evaluate_source80(model,annotation_json,image_root,result_path,device=None,
                      score_threshold=.001,progress_every=250):
    from pycocotools.coco import COCO
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device);model.eval()  # never mutate requires_grad during mid-training validation
    coco=COCO(str(annotation_json));ids=sorted(coco.getImgIds())
    cat_ids=sorted(coco.getCatIds())
    if len(cat_ids)!=80 or cat_ids[0]!=1:raise ValueError('Need official 80 COCO categories')
    results=[]
    for ix,iid in enumerate(ids,1):
        im=coco.loadImgs([iid])[0];bgr=cv2.imread(str(Path(image_root)/im['file_name']))
        if bgr is None:raise FileNotFoundError(f'Missing COCO validation image: {im["file_name"]}')
        x,_,meta=letterbox(bgr,[],1280,736,720)
        raw=model(x.unsqueeze(0).to(device))
        decoded=decode_coco80(raw,736,1280,score_threshold=score_threshold,k=100)[0].cpu().numpy()
        if len(decoded):
            orig=invert_letterbox_xyxy(decoded[:,:4],meta)
            for box,score,cls in zip(orig,decoded[:,4],decoded[:,5]):
                x1,y1,x2,y2=map(float,box)
                if x2<=x1 or y2<=y1:continue
                results.append({'image_id':int(iid),'category_id':cat_ids[int(cls)],
                    'bbox':[x1,y1,x2-x1,y2-y1],'score':float(score)})
        if progress_every and ix%progress_every==0:print(f'COCO-80 evaluation {ix}/{len(ids)}')
    ev,metrics=_coco_eval(coco,results,ids,cat_ids,result_path)
    if ev is not None:
        # COCO precision: IoU x recall x category x area x maxDets
        precision=ev.eval['precision']
        person_slice=precision[0,:,cat_ids.index(1),0,-1]
        valid=person_slice[person_slice>=0]
        metrics['AP50_person']=float(valid.mean()) if len(valid) else 0.
    else:metrics['AP50_person']=0.
    metrics.update({'images':len(ids),'categories':80,'predictions':len(results)})
    return metrics


@torch.inference_mode()
def benchmark_real_cctv(model,image_json,image_dir,device=None,warmup=10,limit=50):
    """CUDA synchronized forward+postprocess on PREPROCESSED real CCTV images.

    Disk I/O and preprocessing excluded, support encoding not applicable to SA-PE-DA.
    CPU timing is not a substitute for matched Meta-DETR A100 timing.
    """
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device);model.eval()  # inference_mode ensures no gradients, preserves trainable flags
    raw=load_coco_json(image_json);images=raw['images'][:limit]
    inputs=[]
    for im in images:
        bgr=cv2.imread(str(Path(image_dir)/im['file_name']))
        if bgr is None:raise FileNotFoundError(im['file_name'])
        x,_,_=letterbox(bgr,[],1280,736,720);inputs.append(x.unsqueeze(0).to(device))
    if not inputs:raise ValueError('No real CCTV images for benchmark')
    for i in range(warmup):
        outs=model(inputs[i%len(inputs)]);decode_detections(outs,736,1280,score_threshold=.001,k=100)
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=[]
    for x in inputs:
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        outs=model(x);decode_detections(outs,736,1280,score_threshold=.001,k=100)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed.append(1000*(time.perf_counter()-start))
    return {'ms_mean':float(np.mean(elapsed)),'ms_median':float(np.median(elapsed)),
            'ms_std':float(np.std(elapsed)),'images':len(elapsed),
            'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'batch_size':1,'resolution':'1280x736',
            'scope':'forward+person heatmap peak/decode; excludes preprocessing, disk I/O'}
