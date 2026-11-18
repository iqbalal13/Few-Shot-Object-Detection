if SOURCE_VAL_METRICS.is_file():
    source_val_metrics = json.loads(SOURCE_VAL_METRICS.read_text())
    assert int(source_val_metrics['images']) == 5000
    assert int(source_val_metrics['categories']) == 80
    display(pd.DataFrame([{
        'COCO-Val AP50:95 (%)':round(100*source_val_metrics['AP50:95'],2),
        'COCO-Val AP50 (%)':round(100*source_val_metrics['AP50'],2),
        'COCO-Val person AP50 (%)':round(100*source_val_metrics['AP50_person'],2)
    }]))
else:
    print('Hasil COCO-Val lama tidak ditemukan; adaptasi CCTV tetap dapat berjalan dari SOURCE checkpoint.')
