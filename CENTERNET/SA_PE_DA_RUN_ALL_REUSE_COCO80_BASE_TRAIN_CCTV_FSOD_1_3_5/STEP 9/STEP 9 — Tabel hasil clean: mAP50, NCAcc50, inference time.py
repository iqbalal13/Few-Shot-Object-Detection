full_table=pd.DataFrame(rows).sort_values(['shot','seed'])
summary=full_table.groupby('shot',as_index=False).agg(
    mAP50_mean=('mAP50_percent','mean'),
    NCAcc50_mean=('NCAcc50_percent','mean'),
    inference_ms_mean=('inference_ms_per_image','mean'))

hasil=summary.rename(columns={
    'shot':'Shot',
    'mAP50_mean':'mAP50 (%)',
    'NCAcc50_mean':'NCAcc50 (%)',
    'inference_ms_mean':'Inference Time (ms)'})
hasil['Shot']=hasil['Shot'].astype(str)+'-Shot'
display(hasil.round(2).reset_index(drop=True))

full_table.to_csv(METRICS_ROOT/'CCTV_TEST_per_shot_seed.csv',index=False)
hasil.to_csv(METRICS_ROOT/'CCTV_TEST_summary_clean.csv',index=False)
print('CSV saved:', METRICS_ROOT)
