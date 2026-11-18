from sape_da.cctv_io import prepare_all_cctv
CCTV_DRIVE_ROOT=Path('/content/drive/MyDrive/DATASET/GAMBAR CCTV')
CCTV_LOCAL=PROJECT/'data'/'cctv'
CCTV_SUMMARY=prepare_all_cctv(CCTV_DRIVE_ROOT,CCTV_LOCAL,yolo_person_class_id=0)
CCTV_TRAIN_JSON=CCTV_LOCAL/'train'/'annotations.json'
CCTV_VAL_JSON=CCTV_LOCAL/'val'/'annotations.json'
CCTV_TEST_JSON=CCTV_LOCAL/'test'/'annotations.json'
CCTV_TRAIN_IMAGES=CCTV_LOCAL/'train'/'images'
CCTV_VAL_IMAGES=CCTV_LOCAL/'val'/'images'
CCTV_TEST_IMAGES=CCTV_LOCAL/'test'/'images'
print(json.dumps(CCTV_SUMMARY,indent=2))
print('Available split sizes:', {k:v['images'] for k,v in CCTV_SUMMARY.items()})
assert CCTV_SUMMARY['train']['images']>=5
assert CCTV_SUMMARY['val']['images']>0 and CCTV_SUMMARY['test']['images']>0
