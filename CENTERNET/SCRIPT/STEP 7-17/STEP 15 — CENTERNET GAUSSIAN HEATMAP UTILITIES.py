
# STEP 15 — GAUSSIAN V1.2 FOR A3
import math
import torch
from functools import lru_cache

def gaussian_radius(height, width, min_overlap=0.7, mode=None):
    if mode is None:
        mode = CONFIG["gaussian_radius_mode"]

    h, w, o = float(height), float(width), float(min_overlap)
    assert h > 0 and w > 0 and 0 < o < 1

    a1, b1 = 1., h+w
    c1 = h*w*(1-o)/(1+o)

    a2, b2 = 4., 2*(h+w)
    c2 = (1-o)*h*w

    a3, b3 = 4*o, -2*o*(h+w)
    c3 = (o-1)*h*w

    d1 = math.sqrt(max(0., b1*b1-4*a1*c1))
    d2 = math.sqrt(max(0., b2*b2-4*a2*c2))
    d3 = math.sqrt(max(0., b3*b3-4*a3*c3))

    if mode == "v1_2_original":
        r1 = (b1+d1)/(2*a1)
        r2 = (b2+d2)/(2*a2)
    elif mode == "v1_3_corrected":
        r1 = (b1-d1)/(2*a1)
        r2 = (b2-d2)/(2*a2)
    else:
        raise ValueError(mode)

    r3 = (b3+d3)/(2*a3)
    return max(0., min(r1,r2,r3))

def gaussian2d(shape, sigma=1.):
    h,w = shape
    y = torch.arange(h,dtype=torch.float32)-(h-1)/2
    x = torch.arange(w,dtype=torch.float32)-(w-1)/2
    yy,xx = torch.meshgrid(y,x,indexing="ij")
    return torch.exp(-(xx**2+yy**2)/(2*sigma**2))

@lru_cache(maxsize=256)
def _gaussian_for_radius(radius):
    diameter = 2*int(radius)+1
    return gaussian2d(
        (diameter,diameter),
        sigma=diameter/6
    )

def draw_gaussian(heatmap,center,radius):
    radius = int(radius)
    x,y = int(center[0]),int(center[1])
    height,width = heatmap.shape

    left = min(x,radius)
    right = min(width-x-1,radius)
    top = min(y,radius)
    bottom = min(height-y-1,radius)

    if min(left,right,top,bottom)<0:
        return heatmap

    gaussian = _gaussian_for_radius(radius)

    region = heatmap[
        y-top:y+bottom+1,
        x-left:x+right+1
    ]

    kernel = gaussian[
        radius-top:radius+bottom+1,
        radius-left:radius+right+1
    ]

    torch.maximum(region,kernel,out=region)
    return heatmap

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert int(gaussian_radius(12,12)) == 1
assert int(gaussian_radius(
    12,12,mode="v1_3_corrected"
)) == 0

print("STEP 15 PASS — A3 USES GAUSSIAN V1.2")
