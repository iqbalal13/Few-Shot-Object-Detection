!pip -q install "gdown>=5.2,<6" "pycocotools>=2.0.7" "pyyaml>=6" "pandas>=2" requests opencv-python-headless
import torch, torchvision, cv2, sys
print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
print('CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
if not torch.cuda.is_available():
    raise RuntimeError('GPU A100 wajib untuk menjalankan protokol training ini.')
