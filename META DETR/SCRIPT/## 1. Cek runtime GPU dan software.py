import sys, os, subprocess, platform, shutil
import torch, torchvision
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('Torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Aktifkan GPU terlebih dahulu di Runtime > Change runtime type > GPU.'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA capability:', torch.cuda.get_device_capability(0))
print('NVCC:', shutil.which('nvcc'))
if shutil.which('nvcc'):
    subprocess.run(['nvcc','--version'], check=False)
else:
    raise RuntimeError('nvcc belum ada; operator Deformable Attention resmi membutuhkan CUDA toolkit untuk build.')
