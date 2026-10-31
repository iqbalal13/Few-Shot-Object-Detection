import os, sys, subprocess
RUN_FINETUNE = False   # <- ubah True untuk menjalankan penuh
SHOT = 1              # 1, 3, atau 5
SEED = 1              # cek subfolder yang tersedia (contoh seed1)
EPOCHS = 500          # protokol asli; sangat lama di Colab single GPU
BATCH_SIZE = 1        # modifikasi hardware praktis, BUKAN reproduksi exact paper
SAVE_EVERY_EPOCH = 50
assert SHOT in (1,3,5)
assert EPOCHS >= 1
FS_ROOT = ROOT / 'data' / 'coco_fewshot'
# Jangan berasumsi nama persis JSON; split disediakan repository dan dibaca dataset loader aslinya.
FS_ANN_CANDIDATES = sorted(FS_ROOT.rglob(f'*{SHOT}shot*.json')) if FS_ROOT.exists() else []
print('Contoh file split yang cocok:', [str(p.relative_to(ROOT)) for p in FS_ANN_CANDIDATES[:6]])
OUT_DIR = ROOT / 'exps' / 'coco' / f'seed{SEED}_{SHOT}shot_colab'
cmd = [sys.executable,'-u','main.py',
       '--dataset_file','coco_base','--backbone','resnet101',
       '--num_feature_levels','1','--enc_layers','6','--dec_layers','6',
       '--hidden_dim','256','--num_queries','300',
       '--batch_size',str(BATCH_SIZE),'--num_workers','2',
       '--category_codes_cls_loss',
       '--resume',str(CHECKPOINT_PATH),
       '--fewshot_finetune','--fewshot_seed',str(SEED),'--num_shots',str(SHOT),
       '--epochs',str(EPOCHS),'--lr_drop_milestones','300','450',
       '--warmup_epochs','50','--save_every_epoch',str(SAVE_EVERY_EPOCH),
       '--eval_every_epoch',str(EPOCHS),'--output_dir',str(OUT_DIR)]
print('Command:', ' '.join(cmd))
print('Folder split few-shot:', FS_ROOT, 'exists:', FS_ROOT.is_dir())
if RUN_FINETUNE:
    assert all((ROOT/'data'/'coco'/p).exists() for p in REQUIRED),'COCO dataset belum lengkap'
    assert FS_ROOT.is_dir(),'Folder split few-shot tidak ditemukan'
    print('Peringatan: validasi nama split dan seed dengan datasets/ resmi sebelum training.')
    assert Path(CHECKPOINT_PATH).is_file(),'Checkpoint base belum ditemukan'
    subprocess.run(cmd,cwd=str(ROOT),env=os.environ.copy(),check=True)
else:
    print('Fine-tuning belum dijalankan (RUN_FINETUNE=False).')
