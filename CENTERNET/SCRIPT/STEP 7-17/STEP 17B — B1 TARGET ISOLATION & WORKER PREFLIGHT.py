
# STEP 17B — A3 PREFLIGHT
import torch
from torch.utils.data import DataLoader,Subset

assert CONFIG["gaussian_radius_mode"]=="v1_2_original"

synthetic = {
    "boxes":torch.tensor(
        [[160.,160.,208.,208.]],
        dtype=torch.float32
    ),
    "labels":torch.tensor([0])
}

t = build_centernet_target(synthetic)

assert int(t["mask"].sum())==1
assert int(t["neighbor_mask"].sum())==8
assert torch.all(t["wh"][0]==12)

center = int(t["indices"][0])
cy,cx = divmod(center,160)

assert t["heatmap"][0,cy,cx]==1.
assert t["heatmap"][0,cy,cx+1]>0

# Duplicate overlapping boxes: no unambiguous neighbor.
overlap = {
    "boxes":synthetic["boxes"].repeat(2,1),
    "labels":torch.tensor([0,0])
}
t2 = build_centernet_target(overlap)
assert int(t2["neighbor_mask"].sum())==0

# Edge case.
edge = {
    "boxes":torch.tensor([[0.,0.,12.,12.]]),
    "labels":torch.tensor([0])
}
te = build_centernet_target(edge)
sel = te["neighbor_indices"][te["neighbor_mask"]]
assert ((sel>=0)&(sel<160*160)).all()

# Compare target heatmap with direct original V1.2 drawing.
for i in range(min(32,len(train_dataset))):
    _,raw = train_dataset[i]
    enc = build_centernet_target(raw)
    old_hm = torch.zeros_like(enc["heatmap"])
    good_count = 0

    for box,cls in zip(raw["boxes"],raw["labels"]):
        x1,y1,x2,y2 = map(float,box.tolist())
        bw,bh = x2-x1,y2-y1
        cx,cy = (x1+x2)/8.,(y1+y2)/8.

        if bw<=1 or bh<=1:
            continue
        if not (0<=cx<160 and 0<=cy<160):
            continue
        if good_count>=CONFIG["max_objects"]:
            continue

        good_count+=1

        r = max(0,int(gaussian_radius(
            bh/4.,bw/4.,
            mode="v1_2_original"
        )))

        draw_gaussian(
            old_hm[int(cls)],
            (int(cx),int(cy)),r
        )

    assert torch.equal(enc["heatmap"],old_hm)

worker_loader = DataLoader(
    Subset(train_dataset,range(8)),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    persistent_workers=False,
    collate_fn=centernet_encoded_collate_fn
)

images,raw_batch,encoded = next(iter(worker_loader))

for i,raw in enumerate(raw_batch):
    direct = build_centernet_target(raw)

    for key in (
        "heatmap","wh","offset",
        "indices","mask",
        "neighbor_indices","neighbor_mask"
    ):
        assert torch.equal(
            encoded[key][i],direct[key]
        ),key

assert encoded["neighbor_mask"].shape==(8,128,8)

print("STEP 17B PASS — A3 TARGET AND WORKERS")
