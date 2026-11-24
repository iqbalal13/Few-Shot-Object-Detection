from pathlib import Path
import os, sys, subprocess, re, shutil
from torch.utils.cpp_extension import CUDA_HOME

OPS = ROOT / 'models' / 'ops'
assert OPS.is_dir(), 'Folder ops tidak ditemukan. Jalankan clone repo terlebih dahulu.'
print('Python executable:', sys.executable)
print('PyTorch:', torch.__version__)
print('CUDA PyTorch:', torch.version.cuda)
print('CUDA_HOME:', CUDA_HOME)
print('nvcc:', shutil.which('nvcc'))
assert CUDA_HOME, 'CUDA_HOME tidak tersedia; CUDA toolkit development dibutuhkan.'
assert shutil.which('nvcc'), 'nvcc tidak ditemukan.'
nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
print('nvcc version:', (nvcc_result.stdout or nvcc_result.stderr)[-800:])

# Patch API operator CUDA, bukan arsitektur model. Simpan .bak sekali saja.
APPLY_LEGACY_CUDA_API_PATCH = True
if APPLY_LEGACY_CUDA_API_PATCH:
    files = list((OPS / 'src').rglob('*.cu')) + list((OPS / 'src').rglob('*.cpp'))
    count = 0
    for f in files:
        old = f.read_text()
        new = old
        # AT_DISPATCH membutuhkan ScalarType, bukan DeprecatedTypeProperties
        new = re.sub(r'AT_DISPATCH_FLOATING_TYPES(_AND_HALF)?\(\s*(\w+)\.type\(\)',
                     lambda m: f'AT_DISPATCH_FLOATING_TYPES{m.group(1) or ""}({m.group(2)}.scalar_type()', new)
        # Pemeriksaan CUDA modern (tensor.is_cuda())
        new = re.sub(r'(\w+)\.type\(\)\.is_cuda\(\)', r'\1.is_cuda()', new)
        # Akses pointer C++ modern
        new = re.sub(r'\.data<([^>]+)>\(\)', r'.data_ptr<\1>()', new)
        if new != old:
            backup = f.with_name(f.name + '.pre_colab_patch.bak')
            if not backup.exists(): backup.write_text(old)
            f.write_text(new)
            count += 1
            print('Patched legacy API in:', f.relative_to(ROOT))
    print('CUDA source files modified:', count)

os.environ['MAX_JOBS'] = '2'
major, minor = torch.cuda.get_device_capability(0)
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
os.environ['PYTHONPATH'] = f'{OPS}:{ROOT}:' + os.environ.get('PYTHONPATH', '')
for path in (str(OPS), str(ROOT)):
    if path not in sys.path: sys.path.insert(0, path)
cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
print('Compile command:', ' '.join(cmd))
print('TORCH_CUDA_ARCH_LIST:', os.environ['TORCH_CUDA_ARCH_LIST'])
logfile = Path('/content/meta_detr_build.log')
p = subprocess.run(cmd, cwd=str(OPS), env=os.environ.copy(),
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, errors='replace')
logfile.write_text(p.stdout or '')
print('Compiler exit code:', p.returncode)
print('Build log:', logfile)
print('\n--- 120 LAST BUILD LOG LINES ---')
print('\n'.join((p.stdout or '').splitlines()[-120:]))
if p.returncode:
    raise RuntimeError(f'Build gagal. Buka {logfile} untuk pesan compiler lengkap (di atas).')
extensions = list(OPS.glob('MultiScaleDeformableAttention*.so'))
assert extensions, 'Compiler sukses tetapi extension .so tidak ditemukan.'
print('Build selesai:', [x.name for x in extensions])
