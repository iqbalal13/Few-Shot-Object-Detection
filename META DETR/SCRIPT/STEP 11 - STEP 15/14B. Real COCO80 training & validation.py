from pathlib import Path
import json, math
import matplotlib.pyplot as plt

COCO_STAGE1_DIR=WORKSPACE/'outputs'/'stage1_coco80'
COCO_STAGE1_CKPT=COCO_STAGE1_DIR/'checkpoint.pth'
TRAIN_LOG=COCO_STAGE1_DIR/'train_metrics.jsonl'
VAL_LOG=COCO_STAGE1_DIR/'results.txt'
GRAPH_DIR=COCO_STAGE1_DIR/'graphs'
GRAPH_DIR.mkdir(parents=True,exist_ok=True)

def read_jsonl(path):
    if not path.is_file():
        return []
    rows=[]
    for n,line in enumerate(path.read_text().splitlines(),1):
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: print('Skipping incomplete log row',path.name,n)
    # On restarts, last occurrence for each epoch wins.
    return [r for _,r in sorted({int(r['epoch']):r for r in rows if 'epoch' in r}.items())]

def usable_xy(rows, key):
    pairs=[]
    for r in rows:
        value=r.get(key)
        if isinstance(value,(int,float)) and math.isfinite(float(value)):
            pairs.append((int(r['epoch'])+1, float(value)))
    return [x for x,_ in pairs], [y for _,y in pairs]

train_rows=read_jsonl(TRAIN_LOG)
val_rows=read_jsonl(VAL_LOG)
# Fallback: original results.txt also logs train loss on evaluation epochs.
train_history=train_rows if train_rows else val_rows

print('Stage 1 checkpoint exists:',COCO_STAGE1_CKPT.is_file())
print('Training epochs logged:',len(train_rows),'| COCO-Val evaluations:',len(val_rows))
if COCO_STAGE1_CKPT.is_file():
    state=torch.load(COCO_STAGE1_CKPT,map_location='cpu',weights_only=False)
    completed=int(state.get('epoch',-1))+1
    del state
    print(f'Official checkpoint epoch field: {completed}; may represent a PARTIAL time-budgeted epoch.')
else:
    print('⚠️ Belum ada checkpoint baru COCO-80. Tidak ada hasil training untuk digrafikkan.')

xt,yt=usable_xy(train_history,'train_loss')
xv,yv=usable_xy(val_rows,'test_loss')
if xt or xv:
    fig,ax=plt.subplots(figsize=(9,4.5))
    if xt: ax.plot(xt,yt,marker='o',label='Train: total loss')
    if xv: ax.plot(xv,yv,marker='s',label='COCO-Val: total loss')
    ax.set(xlabel='Epoch',ylabel='Weighted total loss',title='Meta-DETR Stage 1 — COCO-80 train vs COCO-Val loss')
    ax.grid(True,alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR/'loss_train_vs_val.png',dpi=170)
    plt.show()
else:
    print('Loss chart unavailable: train_metrics.jsonl / results.txt have no real loss observations.')

ap_points=[]
for r in val_rows:
    arr=r.get('test_coco_eval_bbox')
    if isinstance(arr,(list,tuple)) and len(arr)>=3:
        ap_points.append((int(r['epoch'])+1, float(arr[0])*100,float(arr[1])*100,float(arr[2])*100))
if ap_points:
    epochs=[r[0] for r in ap_points]
    fig,ax=plt.subplots(figsize=(9,4.5))
    for col,name in ((1,'AP50:95'),(2,'AP50'),(3,'AP75')):
        ax.plot(epochs,[r[col] for r in ap_points],marker='o',label=name)
    ax.set(xlabel='Epoch',ylabel='Bounding-box AP (%)',title='Meta-DETR Stage 1 — COCO-Val AP (80 categories)')
    ax.set_ylim(0,100)
    ax.grid(True,alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR/'coco_val_ap50_ap75_ap5095.png',dpi=170)
    plt.show()
    last=ap_points[-1]
    print(f'Latest COCO-Val @ epoch {last[0]}: AP50:95={last[1]:.2f}%, AP50={last[2]:.2f}%, AP75={last[3]:.2f}%')
else:
    print('AP chart unavailable: COCO-Val has not produced official test_coco_eval_bbox metrics yet.')
print('Saved real-data figure directory:',GRAPH_DIR)
# Strictly qualify a time-budgeted checkpoint: existence + real COCO-Val + explicit review.
G3_READY=False
status_file=COCO_STAGE1_DIR/'stage1_timebudget_status.json'
if status_file.is_file():
    stage1_status=json.loads(status_file.read_text())
    print('Time-budget training steps:',stage1_status.get('steps_completed'),
          '| elapsed train seconds:',stage1_status.get('training_seconds'),
          '| full epoch:',stage1_status.get('epoch_complete'))
    if stage1_status.get('epoch_complete') is False:
        print('This is a PARTIAL epoch; accuracy may be far from converged.')
    G3_READY=bool(COCO_STAGE1_CKPT.is_file() and ap_points and
                  stage1_status.get('status')=='finished_with_coco_val' and
                  stage1_status.get('steps_completed',0)>0 and SOURCE_CHECKPOINT_APPROVED)
print('G3 APPROVED FOR CCTV:',G3_READY,
      '(requires real COCO-Val AP and explicit SOURCE_CHECKPOINT_APPROVED=True after review)')

# Real Stage1 per-chunk loss (100 optimizer steps per point; no synthetic curves).
STEP_TRACE=COCO_STAGE1_DIR/'train_steps_metrics.jsonl'
chunk_rows=read_jsonl(STEP_TRACE) if False else []
if STEP_TRACE.is_file():
    for line in STEP_TRACE.read_text().splitlines():
        try: chunk_rows.append(json.loads(line))
        except json.JSONDecodeError: pass
if chunk_rows:
    valid=[r for r in chunk_rows if 'steps_completed' in r and 'train_loss' in r]
    if valid:
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.plot([r['steps_completed'] for r in valid],
                [r['train_loss'] for r in valid],marker='.',label='Mean loss / 100 steps')
        ax.set(xlabel='Optimizer steps',ylabel='Weighted training loss',
               title='Meta-DETR Stage 1 - actual COCO-80 training steps (~60 min budget)')
        ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(GRAPH_DIR/'train_loss_per_steps.png',dpi=170);plt.show()
