# Pilih alur sebelum Runtime -> Run all.
#RUN_MODE = 'stage2_cctv'
RUN_MODE = 'full_pipeline'  # default: COCO80 -> CCTV -> final test (satu notebook)
# Jika Stage1 sudah selesai di Drive: RUN_MODE = 'stage2_cctv' (CCTV saja, sampai final test).
assert RUN_MODE in ('full_pipeline', 'stage1', 'stage2_cctv')
print('META-DETR RUN MODE:', RUN_MODE)
