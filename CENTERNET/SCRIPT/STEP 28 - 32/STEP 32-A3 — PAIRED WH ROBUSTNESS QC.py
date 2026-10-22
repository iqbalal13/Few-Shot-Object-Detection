
# STEP 32-A3 — PAIRED WH ROBUSTNESS QC
import os,json,gc
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

def wh_size_iou(pred,true):
    p=pred.float().clamp_min(0)
    t=true.float()
    inter=torch.minimum(p,t).prod(-1)

    return inter/(
        p.prod(-1)+t.prod(-1)-inter
    ).clamp_min(1e-9)

loader=DataLoader(
    Subset(val_dataset,range(1000)),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=centernet_collate_fn
)

qc32={}

for label,path in paths.items():
    ck=torch.load(
        path,map_location="cpu",weights_only=False
    )

    net=CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    net.load_state_dict(
        ck["model_state_dict"],strict=True
    )
    net.eval()

    stats={
        g:defaultdict(float)
        for g in (
            "all","small","medium","large","person"
        )
    }

    with torch.inference_mode():
        for images,raw_targets in loader:
            with torch.amp.autocast(
                "cuda",dtype=torch.bfloat16
            ):
                output=net(images.to(device))

            pred=output["wh"].float()

            for b,raw in enumerate(raw_targets):
                t=build_centernet_target(
                    raw,
                    gaussian_radius_mode="v1_2_original"
                )

                active=torch.nonzero(
                    t["mask"]
                ).flatten()

                scale=float(raw["scale"])

                for k in active.tolist():
                    nm=t["neighbor_mask"][k]

                    if not nm.any():
                        continue

                    center=int(t["indices"][k])
                    cy,cx=divmod(center,OUTPUT_WIDTH)
                    truth=t["wh"][k].to(device)

                    center_iou=float(wh_size_iou(
                        pred[b,:,cy,cx],truth
                    ))

                    ix=t["neighbor_indices"][k][nm].tolist()
                    ny=[v//OUTPUT_WIDTH for v in ix]
                    nx=[v%OUTPUT_WIDTH for v in ix]

                    near_wh=pred[b,:,ny,nx].T

                    neighbor_iou=float(
                        wh_size_iou(
                            near_wh,truth
                        ).mean()
                    )

                    area=(
                        float(t["wh"][k].prod())
                        *(OUTPUT_STRIDE/scale)**2
                    )

                    group=(
                        "small" if area<32**2
                        else "medium" if area<96**2
                        else "large"
                    )

                    categories=["all",group]

                    if int(t["labels"][k])==0:
                        categories.append("person")

                    for g in categories:
                        stats[g]["n"]+=1
                        stats[g]["center"]+=center_iou
                        stats[g]["neighbor"]+=neighbor_iou

    qc32[label]={
        g:{
            "eligible_GT":int(s["n"]),
            "center_iou":s["center"]/max(1,s["n"]),
            "neighbor_iou":s["neighbor"]/max(1,s["n"]),
            "center_minus_neighbor":(
                s["center"]-s["neighbor"]
            )/max(1,s["n"])
        }
        for g,s in stats.items()
    }

    del net,ck
    gc.collect()
    torch.cuda.empty_cache()

for g in (
    "all","small","medium","large","person"
):
    c=qc32["A2_control"][g]
    a=qc32["A3_neighbor_R1"][g]

    assert c["eligible_GT"]==a["eligible_GT"]

    print(
        g,
        "eligible=",a["eligible_GT"],
        "Delta center=",round(
            a["center_iou"]-c["center_iou"],5
        ),
        "Delta neighbor=",round(
            a["neighbor_iou"]-c["neighbor_iou"],5
        )
    )

qc_file=os.path.join(
    PILOT_ROOT,"pilot_A3_WH_neighborhood_QC.json"
)

with open(qc_file,"w") as f:
    json.dump(qc32,f,indent=2)

print("STEP 32-A3 PASS — SAVED:",qc_file)
