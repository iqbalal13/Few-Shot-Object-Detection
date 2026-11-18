# Audit gambar/label + pembatasan klaim same-camera (bukan otomatis leakage).
from sape_da.data import load_coco_json
raw={s:load_coco_json(CCTV_LOCAL/s/'annotations.json') for s in ('train','val','test')}
for s,data in raw.items():
  print(s,len(data['images']),'images,',len(data['annotations']),'person bounding boxes')
file_sets={s:set(Path(x['file_name']).name.split('_',1)[-1] for x in raw[s]['images'])
           for s in raw}
assert not (file_sets['train'] & file_sets['val'])
assert not (file_sets['train'] & file_sets['test'])
assert not (file_sets['val'] & file_sets['test'])
print('PASS: file basenames disjoint. Kamera yang sama bukan bukti kebocoran, '
      'namun generalisasi antarkamera TIDAK diklaim.')
