from metadetr_pipeline.stage2 import run_finetune
from metadetr_pipeline.data_protocol import save_json
STAGE1_CHECKPOINT=WORKSPACE/'outputs'/'stage1_coco80'/'checkpoint.pth'
if RUN_STAGE2_SMOKE:
    assert G3_READY, 'G3 incomplete: finish COCO-80 + validation before CCTV.'
    assert STAGE1_CHECKPOINT.is_file(),'COCO-80 Stage1 checkpoint required'
    for k in STAGE2_SHOTS:
        result=run_finetune(ROOT,WORKSPACE,CCTV_ROOT,STAGE1_CHECKPOINT,shot=k,
                            epochs=STAGE2_EPOCHS,seed=SEED,lr=STAGE2_LR,smoke_only=True)
        report_path=WORKSPACE/'gates'/f'g4_stage2_{k}shot_smoke.json'
        save_json(report_path,result)
        print(f'G4 shot={k} PASS:',report_path)
else:
    print('G4 Stage2 smoke disabled. Set RUN_STAGE2_SMOKE=True after CCTV prep.')
