from pathlib import Path
# CONTOH setelah mount Drive atau extract ke /content:
COCO_DIR = Path('/content/datasets/coco')
project_coco = ROOT/'data'/'coco'
REQUIRED = ['train2017','val2017','annotations/instances_train2017.json','annotations/instances_val2017.json']
if COCO_DIR.is_dir() and all((COCO_DIR/p).exists() for p in REQUIRED):
    if project_coco.is_symlink(): project_coco.unlink()
    elif project_coco.exists() and any(project_coco.iterdir()):
        raise RuntimeError(f'{project_coco} sudah berisi data. Atur secara manual, tidak otomatis dihapus.')
    elif project_coco.exists(): project_coco.rmdir()
    project_coco.symlink_to(COCO_DIR, target_is_directory=True)
    print('✅ COCO directory linked:',project_coco,'->',COCO_DIR)
else:
    print('COCO dataset belum tersedia / belum lengkap:',COCO_DIR)
    print('File/folder yang hilang:',[p for p in REQUIRED if not (COCO_DIR/p).exists()])
print('Split few-shot ada:',(ROOT/'data'/'coco_fewshot').exists())
