
# STEP 16 — A3 COLLISION-AWARE NEIGHBOR TARGETS
import torch

CONFIG["max_objects"] = 128
MAX_OBJECTS = CONFIG["max_objects"]
NUM_CLASSES = CONFIG["num_classes"]

def build_centernet_target(
    target,
    num_classes=NUM_CLASSES,
    output_height=OUTPUT_HEIGHT,
    output_width=OUTPUT_WIDTH,
    output_stride=OUTPUT_STRIDE,
    max_objects=MAX_OBJECTS,
    gaussian_radius_mode=None
):
    mode = gaussian_radius_mode or CONFIG["gaussian_radius_mode"]

    hm = torch.zeros(
        num_classes,output_height,output_width,
        dtype=torch.float32
    )
    wh = torch.zeros(max_objects,2,dtype=torch.float32)
    offset = torch.zeros(max_objects,2,dtype=torch.float32)
    indices = torch.zeros(max_objects,dtype=torch.long)
    mask = torch.zeros(max_objects,dtype=torch.bool)
    labels = torch.full((max_objects,),-1,dtype=torch.long)
    neighbor_indices = torch.zeros(max_objects,8,dtype=torch.long)
    neighbor_mask = torch.zeros(max_objects,8,dtype=torch.bool)

    accepted = []
    blocking_boxes = []
    n_candidates = 0

    for box,class_id in zip(target["boxes"],target["labels"]):
        x1,y1,x2,y2 = map(float,box.tolist())
        bw,bh = x2-x1,y2-y1

        if bw<=1 or bh<=1:
            continue

        cx = (x1+x2)/(2*output_stride)
        cy = (y1+y2)/(2*output_stride)
        w,h = bw/output_stride,bh/output_stride

        if not (
            0<=cx<output_width
            and 0<=cy<output_height
        ):
            continue

        n_candidates += 1

        blocking_boxes.append((
            x1/output_stride,y1/output_stride,
            x2/output_stride,y2/output_stride
        ))

        if len(accepted)>=max_objects:
            continue

        ix,iy = int(cx),int(cy)
        k = len(accepted)

        radius = max(0,int(gaussian_radius(
            h,w,
            min_overlap=CONFIG["gaussian_min_overlap"],
            mode=mode
        )))

        draw_gaussian(hm[int(class_id)],(ix,iy),radius)

        wh[k] = torch.tensor((w,h),dtype=torch.float32)
        offset[k] = torch.tensor(
            (cx-ix,cy-iy),dtype=torch.float32
        )
        indices[k] = iy*output_width+ix
        labels[k] = int(class_id)
        mask[k] = True

        accepted.append((
            k,ix,iy,len(blocking_boxes)-1
        ))

    radius = CONFIG["a3_neighbor_radius"]
    assert radius == 1

    reserved_centers = {
        (cx,cy) for _,cx,cy,_ in accepted
    }

    directions = [
        (dx,dy)
        for dy in (-1,0,1)
        for dx in (-1,0,1)
        if dx or dy
    ]

    for k,cx,cy,box_id in accepted:
        x1,y1,x2,y2 = blocking_boxes[box_id]

        for j,(dx,dy) in enumerate(directions):
            px,py = cx+dx,cy+dy

            if not (
                0<=px<output_width
                and 0<=py<output_height
            ):
                continue

            if (px,py) in reserved_centers:
                continue

            # Center of the feature-map cell.
            qx,qy = px+0.5,py+0.5

            if not (
                x1<=qx<=x2 and y1<=qy<=y2
            ):
                continue

            owners = sum(
                xl<=qx<=xr and yt<=qy<=yb
                for xl,yt,xr,yb in blocking_boxes
            )

            if owners!=1:
                continue

            neighbor_indices[k,j] = py*output_width+px
            neighbor_mask[k,j] = True

    return {
        "heatmap":hm,
        "wh":wh,
        "offset":offset,
        "indices":indices,
        "mask":mask,
        "labels":labels,
        "neighbor_indices":neighbor_indices,
        "neighbor_mask":neighbor_mask,
        "num_objects":len(accepted),
        "num_candidates":n_candidates,
        "num_dropped":n_candidates-len(accepted)
    }

sample_image,sample_target = train_dataset[0]
centernet_target = build_centernet_target(sample_target)

assert centernet_target["heatmap"].shape==(80,160,160)
assert centernet_target["neighbor_mask"].shape==(128,8)

assert not (
    centernet_target["neighbor_mask"]
    & ~centernet_target["mask"][:,None]
).any()

print(
    "STEP 16 PASS — VALID NEIGHBORS:",
    int(centernet_target["neighbor_mask"].sum())
)
