from pathlib import Path
main_file = ROOT/'main.py'
s = main_file.read_text()
old = "torch.load(args.resume, map_location='cpu')"
new = "torch.load(args.resume, map_location='cpu', weights_only=False)"
if old in s:
    main_file.write_text(s.replace(old, new))
    print('Patched main.py: torch.load(weights_only=False)')
else:
    print('main.py sudah di-patch / versi lain')

backbone_file = ROOT/'models'/'backbone.py'
s = backbone_file.read_text()
old = 'pretrained=is_main_process(), norm_layer=norm_layer'
new = 'pretrained=False, norm_layer=norm_layer'
if old in s:
    backbone_file.write_text(s.replace(old, new))
    print('Patched backbone.py: skip redundant ImageNet download before full-checkpoint load')
else:
    print('backbone.py sudah di-patch / versi lain')
print('Diff lokal:')
subprocess.run(['git','-C',str(ROOT),'diff','--stat'],check=True)


# FIX torchvision >= 0.10: parsing lama `float(torchvision.__version__[:3])`
# salah membaca versi 0.26.0 sebagai 0.2, sehingga mencoba import
# `_NewEmptyTensorOp`, yang sudah tidak ada di torchvision modern.
import torchvision
misc_file = ROOT / 'util' / 'misc.py'
misc_src = misc_file.read_text()
legacy_check_05 = 'float(torchvision.__version__[:3]) < 0.5'
legacy_check_07 = 'float(torchvision.__version__[:3]) < 0.7'
modern_check_05 = "tuple(map(int, torchvision.__version__.split('+')[0].split('.')[:2])) < (0, 5)"
modern_check_07 = "tuple(map(int, torchvision.__version__.split('+')[0].split('.')[:2])) < (0, 7)"
fixed_src = misc_src.replace(legacy_check_05, modern_check_05).replace(legacy_check_07, modern_check_07)
if fixed_src != misc_src:
    misc_file.write_text(fixed_src)
    print('Patched util/misc.py: torchvision semantic version checks')
else:
    print('util/misc.py sudah di-patch atau pengecekan versi berbeda')
assert 'float(torchvision.__version__[:3])' not in misc_file.read_text(), 'Masih ada pengecekan versi lama; periksa util/misc.py.'
print('TorchVision detected:', torchvision.__version__)
print('Parsed major/minor:', tuple(map(int, torchvision.__version__.split('+')[0].split('.')[:2])))
