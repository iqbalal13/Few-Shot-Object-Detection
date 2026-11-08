
# ============================================================
# STEP 15B — FIX SUPPORT RESIZE (TRAINING + EVALUATION)
# Jalankan setelah Step 9 dan Step 15
# ============================================================

from pathlib import Path
import importlib
import inspect

import metadetr_pipeline.stage2 as stage2
import metadetr_pipeline.evaluation as evaluation

# 1. Ubah konfigurasi support pada dua modul
patches = [
    (
        stage2,
        "support_tf=T.Compose([T.RandomResize([448],max_size=672),normalize])",
        "support_tf=T.Compose([T.RandomResize([672],max_size=1152),normalize])"
    ),
    (
        evaluation,
        "transforms=T.Compose([T.RandomResize([448],max_size=672),normalize])",
        "transforms=T.Compose([T.RandomResize([672],max_size=1152),normalize])"
    )
]

for module, old_code, new_code in patches:
    module_path = Path(module.__file__).resolve()
    source = module_path.read_text(encoding="utf-8")

    if source.count(old_code) == 1:
        source = source.replace(old_code, new_code)
        module_path.write_text(source, encoding="utf-8")
        print("PATCH APPLIED:", module_path.name)

    elif source.count(new_code) == 1:
        print("PATCH ALREADY ACTIVE:", module_path.name)

    else:
        raise RuntimeError(
            f"Resize pattern tidak ditemukan: {module_path}"
        )

# 2. Kembalikan sampler resmi jika sebelumnya sudah
#    diganti menggunakan safe_sample_support
dataset_module = importlib.import_module("datasets.dataset")
dataset_module = importlib.reload(dataset_module)

# 3. Reload modul agar kode terbaru aktif
importlib.invalidate_caches()
stage2 = importlib.reload(stage2)
evaluation = importlib.reload(evaluation)

# 4. Pastikan resize yang benar sudah aktif
train_source = inspect.getsource(stage2.create_train_loader)
eval_source = inspect.getsource(
    evaluation._representative_support_codes
)

assert "T.RandomResize([672],max_size=1152)" in train_source
assert "T.RandomResize([672],max_size=1152)" in eval_source

# 5. Pastikan sampler original kembali
assert (
    dataset_module.DetectionDataset
    .sample_support_samples.__name__
    == "sample_support_samples"
)

print("=" * 60)
print("SUPPORT RESIZE FIX ACTIVE")
print("TRAIN SUPPORT : short=672, max=1152")
print("EVAL SUPPORT  : short=672, max=1152")
print("SAMPLER       : Original Meta-DETR")
print("STATUS        : PASS")
print("=" * 60)
