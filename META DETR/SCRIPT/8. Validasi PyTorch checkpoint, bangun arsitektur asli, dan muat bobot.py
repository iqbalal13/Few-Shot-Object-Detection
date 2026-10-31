import argparse, torch, sys

if not checkpoint_file_plausible(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f'Checkpoint COCO base belum siap: {CHECKPOINT_PATH}\n'
        'Gunakan link resmi pada bagian 7A, lalu upload /content/checkpoint.pth atau mount MyDrive di bagian 7B.'
    )

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from main import get_args_parser
from models import build_model

print('Memverifikasi file checkpoint COCO base...')
try:
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
except Exception as e:
    raise RuntimeError('File tersedia tetapi tidak bisa dibaca oleh torch.load; cek file asli/unduh ulang.') from e
assert isinstance(checkpoint, dict) and isinstance(checkpoint.get('model'), dict), (
    'Format checkpoint tidak cocok: membutuhkan dict dengan key model.'
)

architecture_flags = [
    '--dataset_file','coco_base', '--backbone','resnet101',
    '--num_feature_levels','1', '--enc_layers','6', '--dec_layers','6',
    '--hidden_dim','256', '--num_queries','300',
    '--category_codes_cls_loss', '--device','cuda'
]
args = get_args_parser().parse_args(architecture_flags)

metadata = checkpoint.get('args')
for field in ('backbone','num_feature_levels','enc_layers','dec_layers','hidden_dim','num_queries'):
    actual = getattr(metadata, field, None) if metadata is not None else None
    requested = getattr(args, field)
    if actual is not None and actual != requested:
        raise ValueError(f'Perbedaan konfigurasi {field}: checkpoint={actual}, notebook={requested}')
print('Checkpoint epoch:', checkpoint.get('epoch', 'tidak diketahui'))
print('Checkpoint state_dict keys:', len(checkpoint['model']))

model, criterion, postprocessors = build_model(args)
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
unexpected = [key for key in unexpected
              if not (key.endswith('total_ops') or key.endswith('total_params'))]
print('Missing keys:', len(missing), list(missing)[:20])
print('Unexpected keys:', len(unexpected), list(unexpected)[:20])
if missing or unexpected:
    raise RuntimeError('Checkpoint tidak cocok penuh dengan model. Jangan fine-tune sebelum diaudit.')
model = model.cuda().eval()
num_params = sum(p.numel() for p in model.parameters())
print('✅ Model Meta-DETR original + checkpoint COCO base BERHASIL DIMUAT.')
print('Total parameters:', f'{num_params:,}')
print('Device:', next(model.parameters()).device)
print('Catatan: operator CUDA lulus smoke test; full detector inference belum diuji.')
