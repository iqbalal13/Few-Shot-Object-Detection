#14 a
from pathlib import Path
import json, shutil, gc

from google.colab import drive
drive.mount('/content/drive')

DRIVE_STAGE1 = Path('/content/drive/MyDrive/MetaDETR_Thesis/Stage1_COCO80')
LOCAL_STAGE1 = WORKSPACE / 'outputs' / 'stage1_coco80'
LOCAL_STAGE1.mkdir(parents=True, exist_ok=True)
assert DRIVE_STAGE1.is_dir(), 'Folder backup Drive Stage1 tidak ditemukan.'

# Aman: tidak overwrite file existing, terutama checkpoint dan snapshot.
def restore_if_missing(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        print('OPTIONAL FILE NOT FOUND:', source)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f'Konflik ukuran file, tidak akan overwrite: {destination}')
        print('ALREADY PRESENT:', destination)
    else:
        tmp = destination.with_name(destination.name + '.copying')
        shutil.copy2(source, tmp)
        if tmp.stat().st_size != source.stat().st_size:
            tmp.unlink(missing_ok=True)
            raise IOError(f'Incomplete restore: {source.name}')
        tmp.replace(destination)
        print('RESTORED:', destination)
    return True

for name in ('checkpoint.pth', 'partial_latest.pth',
             'stage1_timebudget_status.json', 'train_steps_metrics.jsonl',
             'stage1_console.log', 'results.txt'):
    restore_if_missing(DRIVE_STAGE1/name, LOCAL_STAGE1/name)

restore_if_missing(DRIVE_STAGE1/'source_manifest.json', WORKSPACE/'source_manifest.json')
assert (LOCAL_STAGE1/'checkpoint.pth').is_file(), 'Checkpoint COCO-80 tidak ditemukan'
assert (LOCAL_STAGE1/'stage1_timebudget_status.json').is_file(), 'Metadata hasil training hilang'

status = json.loads((LOCAL_STAGE1/'stage1_timebudget_status.json').read_text())
assert status.get('status')=='finished_with_coco_val', status
assert int(status.get('steps_completed',0))>0, 'Tidak ditemukan optimizer steps yang selesai'

# Snapshot eval resmi bisa masih ada di runtime tetapi belum pernah dicopy ke Drive.
LOCAL_COCO_EVAL = LOCAL_STAGE1/'eval'/'latest.pth'
DRIVE_COCO_EVAL = DRIVE_STAGE1/'eval'/'latest.pth'
if LOCAL_COCO_EVAL.is_file() and not DRIVE_COCO_EVAL.is_file():
    DRIVE_COCO_EVAL.parent.mkdir(parents=True,exist_ok=True)
    temp = DRIVE_COCO_EVAL.with_name('latest.copying.pth')
    shutil.copy2(LOCAL_COCO_EVAL,temp)
    if temp.stat().st_size != LOCAL_COCO_EVAL.stat().st_size:
        temp.unlink(missing_ok=True)
        raise IOError('COCOeval snapshot Drive copy incomplete')
    temp.replace(DRIVE_COCO_EVAL)
    print('NEW BACKUP saved: eval/latest.pth (per-class COCO precision)')
elif DRIVE_COCO_EVAL.is_file():
    restore_if_missing(DRIVE_COCO_EVAL,LOCAL_COCO_EVAL)

# Verify stored model payload, not merely that a 606 MB file exists.
import torch
checkpoint = torch.load(LOCAL_STAGE1/'checkpoint.pth', map_location='cpu', weights_only=False)
assert isinstance(checkpoint,dict) and isinstance(checkpoint.get('model'),dict)
print('MODEL CHECKPOINT VERIFIED: model tensors =',len(checkpoint['model']))
print('Checkpoint epoch field:',checkpoint.get('epoch'), '(NOT proof of a full epoch)')
del checkpoint;gc.collect()
print('Optimizer steps:',status['steps_completed'],
      '| GPU train seconds:',status['training_seconds'],
      '| epoch complete:',status['epoch_complete'])
print('Original COCOeval snapshot available:', LOCAL_COCO_EVAL.is_file())
print('Safe: no source retraining has been initiated.')
