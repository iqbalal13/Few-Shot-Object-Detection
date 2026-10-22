
# STEP 30 — PAIRED A2 VS A3 PILOT
import os,json,hashlib,random,gc,time
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Sampler
from google.colab import drive

drive.mount("/content/drive",force_remount=False)

A2_ROOT = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8"
)

A2_CKPT = os.path.join(A2_ROOT,"epoch20_160k.pth")
A2_MANIFEST = os.path.join(
    A2_ROOT,"balanced160k_manifest.jsonl"
)

PILOT_ROOT = os.path.join(
    "/content/drive/MyDrive",
    CONFIG["experiment_name"]
)
os.makedirs(PILOT_ROOT,exist_ok=True)

LOCKED_SHA = (
    "9a42db63045d2008fbabc5f50b6b15129139976fe54f4894745436ded27dd43f"
)

assert os.path.isfile(A2_CKPT)
assert os.path.isfile(A2_MANIFEST)
assert CONFIG["gaussian_radius_mode"]=="v1_2_original"
assert CONFIG["a3_neighbor_lambda"]==0.25
assert not train_dataset.augment
assert not val_dataset.augment

with open(A2_MANIFEST,"rb") as f:
    blob=f.read()

assert hashlib.sha256(blob).hexdigest()==LOCKED_SHA

rows = [
    json.loads(line)
    for line in blob.splitlines()
    if line
]

assert len(rows)==160000
pilot_rows=rows[:16000]
all_image_ids=list(map(int,train_dataset.image_ids))

for epoch in (1,2):
    one=pilot_rows[(epoch-1)*8000:epoch*8000]
    assert len(one)==8000

    assert set(
        Counter(
            x["selected_category_id"] for x in one
        ).values()
    )=={100}

    assert all(
        int(x["epoch"])==epoch
        and int(x["position"])==j
        and all_image_ids[int(x["dataset_index"])]==int(x["image_id"])
        for j,x in enumerate(one)
    )

base_ckpt=torch.load(
    A2_CKPT,map_location="cpu",weights_only=False
)

assert int(base_ckpt["epoch"])==20
assert base_ckpt["config"]["gaussian_radius_mode"]=="v1_2_original"
assert base_ckpt["config"]["wh_loss_type"]=="hybrid_abs_relative"
assert base_ckpt["source_manifest_sha256"]==LOCKED_SHA

BASE_WEIGHTS=base_ckpt["model_state_dict"]

def save30(obj,path):
    tmp=path+".tmp"
    torch.save(obj,tmp)
    os.replace(tmp,path)

class PilotSampler(Sampler):
    def __init__(self):
        self.epoch=1
        self.batches={
            e:[
                int(x["dataset_index"])
                for x in pilot_rows[(e-1)*8000:e*8000]
            ]
            for e in (1,2)
        }

    def set_epoch(self,epoch):
        self.epoch=int(epoch)

    def __len__(self):
        return 8000

    def __iter__(self):
        return iter(self.batches[self.epoch])

def reset_pilot_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

for ARM,NEIGHBOR_LAMBDA in (
    ("A2_control",0.),
    ("A3_neighbor_R1",0.25)
):
    print("\nPILOT ARM:",ARM)

    arm_dir=os.path.join(PILOT_ROOT,ARM)
    os.makedirs(arm_dir,exist_ok=True)

    latest=os.path.join(arm_dir,"latest.pth")
    rng_seed=int(SEED)+3170

    reset_pilot_rng(rng_seed)

    sampler=PilotSampler()
    train_pilot,train_gen=build_train_loader(
        rng_seed,sampler=sampler,encoded=True
    )

    assert len(train_pilot)==1000

    _,raw_check,enc_check=next(iter(train_pilot))

    for j in range(2):
        direct=build_centernet_target(
            raw_check[j],
            gaussian_radius_mode="v1_2_original"
        )

        for key in (
            "heatmap","wh","offset",
            "indices","mask",
            "neighbor_indices","neighbor_mask"
        ):
            assert torch.equal(
                enc_check[key][j],direct[key]
            ),(ARM,key)

    del raw_check,enc_check
    reset_pilot_rng(rng_seed)

    model_pilot=CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    model_pilot.load_state_dict(
        BASE_WEIGHTS,strict=True
    )

    optimizer_pilot=build_source_optimizer(model_pilot)

    for group in optimizer_pilot.param_groups:
        group["lr"]=CONFIG["a3_pilot_lr"]

    scheduler_pilot=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pilot,
        T_max=CONFIG["a3_pilot_epochs"],
        eta_min=2e-6
    )

    criterion.neighbor_lambda=NEIGHBOR_LAMBDA

    history=[]
    start=0

    if os.path.isfile(latest):
        ck=torch.load(
            latest,map_location="cpu",weights_only=False
        )

        assert ck["arm"]==ARM
        assert ck["base_source_sha256"]==LOCKED_SHA
        assert ck["base_checkpoint"]==A2_CKPT
        assert ck["neighbor_lambda"]==NEIGHBOR_LAMBDA
        assert ck["gaussian_radius_mode"]=="v1_2_original"
        assert ck["pilot_lr"]==CONFIG["a3_pilot_lr"]

        start=int(ck["epoch"])
        assert 1<=start<=2

        model_pilot.load_state_dict(
            ck["model_state_dict"],strict=True
        )

        optimizer_pilot.load_state_dict(
            ck["optimizer_state_dict"]
        )

        scheduler_pilot.load_state_dict(
            ck["scheduler_state_dict"]
        )

        history=ck["history"]

        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch_cpu"].cpu())
        torch.cuda.set_rng_state_all(
            ck["rng"]["torch_cuda"]
        )

    for epoch in range(start+1,3):
        sampler.set_epoch(epoch)
        train_gen.manual_seed(rng_seed+epoch)

        t0=time.perf_counter()

        train_stats=train_centernet_epoch(
            model_pilot,train_pilot,optimizer_pilot
        )

        assert train_stats["batches"]==1000

        scheduler_pilot.step()

        history.append({
            "epoch":epoch,
            "additional_exposures":epoch*8000,
            "stats":train_stats,
            "train_seconds":time.perf_counter()-t0
        })

        payload={
            "model_state_dict":model_pilot.state_dict(),
            "optimizer_state_dict":optimizer_pilot.state_dict(),
            "scheduler_state_dict":scheduler_pilot.state_dict(),
            "experiment_name":CONFIG["experiment_name"],
            "arm":ARM,
            "neighbor_lambda":NEIGHBOR_LAMBDA,
            "epoch":epoch,
            "additional_exposures":epoch*8000,
            "base_checkpoint":A2_CKPT,
            "base_source_sha256":LOCKED_SHA,
            "gaussian_radius_mode":"v1_2_original",
            "pilot_lr":CONFIG["a3_pilot_lr"],
            "history":history,
            "config":dict(CONFIG),
            "rng":{
                "python":random.getstate(),
                "numpy":np.random.get_state(),
                "torch_cpu":torch.get_rng_state(),
                "torch_cuda":torch.cuda.get_rng_state_all()
            }
        }

        save30(payload,latest)

        save30(
            payload,
            os.path.join(
                arm_dir,
                f"epoch{epoch}_add{epoch*8}k.pth"
            )
        )

        print(
            ARM,"epoch",epoch,
            "exposures:",epoch*8000
        )

    del model_pilot,optimizer_pilot,scheduler_pilot,train_pilot
    gc.collect()
    torch.cuda.empty_cache()

    assert os.path.isfile(
        os.path.join(arm_dir,"epoch2_add16k.pth")
    )

criterion.neighbor_lambda=0.

print("STEP 30 PASS — BOTH PILOT ARMS COMPLETE")
