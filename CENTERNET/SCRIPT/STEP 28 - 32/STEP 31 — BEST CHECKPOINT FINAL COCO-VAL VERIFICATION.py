
# STEP 31 — OFFICIAL COCOeval A2 VS A3
import os,math,json,io,contextlib,gc
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

assert len(val_dataset)==5000

source_val_loader,_=build_val_loader(
    SEED+3301,encoded=True
)
assert len(source_val_loader)==625

def eval31(path,label):
    ck=torch.load(
        path,map_location="cpu",weights_only=False
    )

    assert ck["arm"]==label
    assert ck["additional_exposures"]==16000
    assert ck["gaussian_radius_mode"]=="v1_2_original"

    model_eval=CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    model_eval.load_state_dict(
        ck["model_state_dict"],strict=True
    )
    model_eval.eval()

    detections=[]
    seen=set()

    with torch.inference_mode():
        for batch in source_val_loader:
            images,targets=batch[0],batch[1]

            with torch.amp.autocast(
                "cuda",dtype=torch.bfloat16
            ):
                raw=model_eval(
                    images.to(device,non_blocking=True)
                )

            predictions=decode_centernet(
                {k:v.float() for k,v in raw.items()},
                K=100,score_threshold=0.
            )

            for pred,t in zip(predictions,targets):
                image_id=int(t["image_id"])
                assert image_id not in seen
                seen.add(image_id)

                scale=float(t["scale"])
                px,py=map(float,t["pad"].tolist())
                H,W=map(int,t["original_size"].tolist())

                boxes=pred["boxes"].detach().cpu().float().clone()

                boxes[:,[0,2]]=(
                    (boxes[:,[0,2]]-px)/scale
                ).clamp(0,W)

                boxes[:,[1,3]]=(
                    (boxes[:,[1,3]]-py)/scale
                ).clamp(0,H)

                scores=pred["scores"].detach().cpu().tolist()
                labels=pred["labels"].detach().cpu().tolist()

                for bb,sc,cls in zip(
                    boxes.tolist(),scores,labels
                ):
                    x1,y1,x2,y2=map(float,bb)

                    if (
                        x2<=x1 or y2<=y1
                        or not math.isfinite(sc)
                    ):
                        continue

                    detections.append({
                        "image_id":image_id,
                        "category_id":int(
                            val_dataset.cat_ids[int(cls)]
                        ),
                        "bbox":[
                            x1,y1,x2-x1,y2-y1
                        ],
                        "score":float(sc)
                    })

    assert seen==set(map(int,val_dataset.image_ids))

    with contextlib.redirect_stdout(io.StringIO()):
        dt=val_dataset.coco.loadRes(detections)

        ce=COCOeval(
            val_dataset.coco,dt,iouType="bbox"
        )

        ce.params.imgIds=sorted(
            map(int,val_dataset.image_ids)
        )
        ce.params.catIds=sorted(
            map(int,val_dataset.cat_ids)
        )
        ce.params.maxDets=[1,10,100]
        ce.params.iouThrs=np.linspace(.5,.95,10)

        ce.evaluate()
        ce.accumulate()
        ce.summarize()

    names=(
        "mAP_50_95","AP50","AP75",
        "AP_small","AP_medium","AP_large",
        "AR1","AR10","AR100",
        "AR_small","AR_medium","AR_large"
    )

    metrics=dict(zip(names,map(float,ce.stats)))

    prec=ce.eval["precision"]
    person_idx=ce.params.catIds.index(1)

    pi=prec[
        :,:,person_idx,
        ce.params.areaRngLbl.index("all"),2
    ]

    def positive_mean(v):
        good=v[v>=0]
        return float(good.mean()) if len(good) else None

    person={
        "person_AP_50_95":positive_mean(pi),
        "person_AP50":positive_mean(pi[0]),
        "person_AP75":positive_mean(pi[5])
    }

    result={
        "label":label,
        "checkpoint":path,
        "images":len(seen),
        "detections":len(detections),
        "metrics":metrics,
        "person_metrics":person
    }

    del model_eval,ce,ck
    gc.collect()
    torch.cuda.empty_cache()
    return result

paths={
    "A2_control":os.path.join(
        PILOT_ROOT,"A2_control","epoch2_add16k.pth"
    ),
    "A3_neighbor_R1":os.path.join(
        PILOT_ROOT,"A3_neighbor_R1","epoch2_add16k.pth"
    )
}

results31={}

for name,path in paths.items():
    assert os.path.isfile(path),path
    print("Evaluating:",name)

    results31[name]=eval31(path,name)

    print(results31[name]["metrics"])
    print(results31[name]["person_metrics"])

print("\nA3 MINUS A2 CONTINUATION (percentage points)")

a=results31["A2_control"]
b=results31["A3_neighbor_R1"]

for k in (
    "mAP_50_95","AP50","AP75",
    "AP_small","AP_medium","AP_large","AR100"
):
    print(
        k,
        round(100*(b["metrics"][k]-a["metrics"][k]),4)
    )

for k in (
    "person_AP_50_95",
    "person_AP50",
    "person_AP75"
):
    print(
        k,
        round(
            100*(b["person_metrics"][k]
                 -a["person_metrics"][k]),4
        )
    )

output31=os.path.join(
    PILOT_ROOT,"pilot_A2_vs_A3_COCOeval.json"
)

with open(output31,"w",encoding="utf-8") as f:
    json.dump(results31,f,indent=2,allow_nan=False)

print("STEP 31 PASS — SAVED:",output31)
