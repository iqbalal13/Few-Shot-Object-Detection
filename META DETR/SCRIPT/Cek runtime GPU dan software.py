!pip -q install "gdown>=5.2,<6" "pycocotools>=2.0.7" "pyyaml>=6" requests opencv-python-headless
import torch, torchvision
print('torch:',torch.__version__,'vision:',torchvision.__version__)
print('CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > GPU (A100 if available).'
