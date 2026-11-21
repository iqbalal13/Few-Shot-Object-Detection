RUN_INFERENCE_ONLY=False
if RUN_INFERENCE_ONLY:
    from google.colab import files
    import cv2,numpy as np
    from sape_da.data import letterbox,invert_letterbox_xyxy
    from sape_da.model import decode_detections
    shot,seed=5,42
    best=TARGET_ROOT/f'shot_{shot}'/f'seed_{seed}'/'target_best.pth'
    net=SAPEDACenterNet()
    load_locked_checkpoint(best,net)
    net=net.cuda();net.freeze_phase('eval')
    uploaded=files.upload()
    for name,raw in uploaded.items():
        bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Not an image:'+name)
        inp,_,meta=letterbox(bgr,[],1280,736,720)
        with torch.inference_mode():
            pred=net(inp.unsqueeze(0).cuda())
            det=decode_detections(pred,736,1280,score_threshold=.5,k=100)[0].cpu().numpy()
        if len(det):
            xyxy=invert_letterbox_xyxy(det[:,:4],meta)
            for xy,sc in zip(xyxy,det[:,4]):
                a,b,c,d=np.round(xy).astype(int)
                cv2.rectangle(bgr,(a,b),(c,d),(25,180,20),2)
                cv2.putText(bgr,f'person {sc:.2f}',(a,max(20,b-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(25,180,20),1)
        out=METRICS_ROOT/('inference_'+Path(name).stem+f'_{shot}shot.jpg')
        cv2.imwrite(str(out),bgr)
        print('Saved inference image:',out)
