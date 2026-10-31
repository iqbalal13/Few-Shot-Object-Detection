from pathlib import Path
ROOT = Path('/content/Meta-DETR')
if not ROOT.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/ZhangGongjie/Meta-DETR.git',str(ROOT)],check=True)
else:
    print('Repository sudah ada:', ROOT)
subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],check=True)
print('Main:', (ROOT/'main.py').exists())
print('Models:', (ROOT/'models'/'meta_detr.py').exists())
assert (ROOT/'main.py').is_file() and (ROOT/'models'/'meta_detr.py').is_file()
