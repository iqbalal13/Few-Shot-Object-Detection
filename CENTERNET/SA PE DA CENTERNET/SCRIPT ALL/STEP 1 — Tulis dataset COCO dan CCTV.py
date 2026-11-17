%%writefile sape_da/data.py
"""Official COCO downloader, LOCKED 8,000-image COCO-80 source, CCTV COCO JSON loader,
letterbox (1280x736), CenterNet Gaussian / sparse width-height / offset targets.
"""
from __future__ import annotations
import concurrent.futures
import json
import math
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import cv2
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

# COCO 2017 *original* annotations. The upstream CDN currently may serve a TLS
# certificate with a mismatched hostname. Never use requests(verify=False).
OFFICIAL_ANNOTATIONS = 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMAGE_DOMAIN = 'images.cocodataset.org'
ANNOTATIONS_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ANNOTATION_SOURCES = [
    OFFICIAL_ANNOTATIONS,
    'https://storage.googleapis.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
    # Independently hosted original COCO 2017 archive, NOT Sama-COCO relabeling.
    # The SHA-256 above is verified for this exact published file.
    'https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/'
    'fbbee70f52f540987cca6936c9f1d7b1ff969ad1/annotations_trainval2017.zip',
]
IMAGE_BASES = [
    'https://images.cocodataset.org',
    'https://storage.googleapis.com/images.cocodataset.org',
    # Official COCO documents use plain HTTP. This is an insecure-transport LAST resort,
    # not a workaround that disables TLS verification on an HTTPS connection.
    'http://images.cocodataset.org',
]
# Original CenterNet expects OpenCV BGR image order, normalized after /255.
MEAN = np.array([0.408, 0.447, 0.470], np.float32)
STD = np.array([0.289, 0.274, 0.278], np.float32)


def _hash(path, algorithm='sha256'):
    import hashlib
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _verified_existing(dest, min_bytes, sha256=None, image=False):
    if not dest.is_file() or dest.stat().st_size < min_bytes:
        return False
    if sha256 is not None and _hash(dest) != sha256:
        return False
    if image:
        # Catches HTTP 200 HTML/error pages and truncated/corrupted JPEGs.
        data = cv2.imread(str(dest), cv2.IMREAD_COLOR)
        return data is not None and data.size > 0
    return True


def _safe_download_url(url):
    p = urlparse(url)
    if p.scheme not in ('https','http'):
        raise ValueError(f'Invalid download scheme: {url}')
    if p.scheme == 'http' and p.hostname != COCO_IMAGE_DOMAIN:
        raise ValueError('Plain HTTP only allowed for official COCO image host')
    if p.hostname not in (COCO_IMAGE_DOMAIN, 'storage.googleapis.com', 'huggingface.co'):
        raise ValueError(f'Not an approved source: {url}')


def fetch_file(url, dest, min_bytes=1024, max_tries=2, sha256=None, image=False):
    """Resumable-by-file download: validated TLS, atomic rename, content checks.

    `sha256` is mandatory for annotations in this workflow. Image files are
    checked via actual JPEG decode, not merely file size.
    """
    _safe_download_url(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _verified_existing(dest,min_bytes,sha256,image):
        return dest
    if dest.exists(): dest.unlink()
    for attempt in range(max_tries):
        tmp = dest.with_name(dest.name + '.part')
        try:
            with requests.get(url, stream=True, timeout=(12,75), allow_redirects=True) as r:
                r.raise_for_status()
                # Guard redirected download URLs (HF may redirect to Xet CAS domain,
                # which must still stay HTTPS; never allow redirect to plain HTTP).
                redirect_url = urlparse(r.url)
                if redirect_url.scheme != 'https' and urlparse(url).scheme == 'https':
                    raise ValueError('HTTPS source unexpectedly redirected to HTTP')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk: f.write(chunk)
            if not _verified_existing(tmp,min_bytes,sha256,image):
                raise IOError(f'Download content invalid or incomplete: {url}')
            tmp.replace(dest)
            return dest
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt + 1 == max_tries: raise
            time.sleep(1 + attempt)


def download_official_annotations(root):
    """Download COCO's ORIGINAL train/val annotations, with SHA-256 verification.

    If official CDN HTTPS is misconfigured, tries a Google Storage endpoint and
    then a byte-identical COCO 2017 mirror. Does not disable TLS certificate checks.
    """
    root = Path(root)
    dst = root / 'annotations_trainval2017.zip'
    if not _verified_existing(dst,5_000_000,ANNOTATIONS_SHA256):
        errors=[]
        for url in ANNOTATION_SOURCES:
            try:
                print(f'Trying COCO 2017 annotations from: {urlparse(url).hostname}')
                fetch_file(url,dst,min_bytes=5_000_000,max_tries=1,sha256=ANNOTATIONS_SHA256)
                print(f'Original COCO archive verified: SHA256 {ANNOTATIONS_SHA256[:16]}...')
                break
            except Exception as exc:
                errors.append(f'{urlparse(url).hostname}: {type(exc).__name__}: {exc}')
                print(f'  Source unavailable ({type(exc).__name__}); trying next verified source')
        else:
            raise RuntimeError('All approved COCO annotation sources failed.\n'+'\n'.join(errors))
    for filename in ('instances_train2017.json','instances_val2017.json'):
        output = root/'annotations'/filename
        if not output.exists():
            with zipfile.ZipFile(dst) as z, z.open('annotations/'+filename) as src:
                output.parent.mkdir(parents=True,exist_ok=True)
                with output.open('wb') as f:
                    import shutil
                    shutil.copyfileobj(src,f)
    return root/'annotations'/'instances_train2017.json',root/'annotations'/'instances_val2017.json'

def load_coco_json(path):
    with open(path,encoding='utf-8') as f:
        return json.load(f)


def is_valid_person(a, person_category=1):
    return (a.get('category_id') == person_category and not a.get('iscrowd',0)
            and a.get('bbox') is not None and len(a['bbox']) == 4
            and a['bbox'][2]>1 and a['bbox'][3]>1 and a.get('area',a['bbox'][2]*a['bbox'][3])>1)


def person_image_subset(coco, fraction=0.25, seed=42):
    """Deterministic stratified 25% of IMAGE IDs containing valid person instances.
    Strata = (min person box size bucket, count bucket). Samples per stratum
    via deterministic global allocation to target total, never duplicate images.
    """
    images = {int(i['id']):i for i in coco['images']}
    by_img=defaultdict(list)
    for a in coco['annotations']:
        if is_valid_person(a) and a['image_id'] in images: by_img[a['image_id']].append(a)
    if not by_img: raise ValueError('No COCO person (category_id=1) with valid box')
    groups=defaultdict(list)
    for image_id,anns in by_img.items():
        min_area=min(a['bbox'][2]*a['bbox'][3] for a in anns)
        size = 'small' if min_area < 32**2 else ('medium' if min_area<96**2 else 'large')
        count = 'single' if len(anns)==1 else ('few' if len(anns)<=4 else 'crowd')
        groups[(size,count)].append(image_id)
    rng=random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    total=len(by_img)
    target=round(total*fraction)
    # largest remainder method, samples distributed close to stratified proportions
    allocations={key:int(len(v)*target/total) for key,v in groups.items()}
    rest=target-sum(allocations.values())
    order=sorted(groups,key=lambda k:(-(len(groups[k])*target/total-allocations[k]),str(k)))
    for k in order[:rest]: allocations[k]+=1
    chosen=sorted(image for key,ids in groups.items() for image in ids[:allocations[key]])
    assert len(chosen)==target and len(chosen)==len(set(chosen))
    return chosen, {'total_person_images':total,'selected':len(chosen),'fraction':fraction,
                    'seed':seed,'strata':{str(k):allocations[k] for k in groups}}


def make_person_manifest(annotation_json, manifest_path, fraction=0.25, seed=42):
    coco=load_coco_json(annotation_json)
    selected,details=person_image_subset(coco,fraction=fraction,seed=seed)
    images={int(i['id']):i for i in coco['images']}
    entries=[]
    for img_id in selected:
        im=images[img_id]
        url=im.get('coco_url',f'https://images.cocodataset.org/train2017/{im["file_name"]}')
        if urlparse(url).hostname != COCO_IMAGE_DOMAIN:
            raise ValueError('Non official coco_url')
        entries.append({'id':img_id,'file_name':im['file_name'],'coco_url':url})
    data={'metadata':details,'images':entries}
    dest=Path(manifest_path);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2),encoding='utf8')
    return data


def manifest_for_person_val(annotation_json):
    coco=load_coco_json(annotation_json)
    valid={a['image_id'] for a in coco['annotations'] if is_valid_person(a)}
    return {'images':[{'id':i['id'],'file_name':i['file_name'],
                        'coco_url':i.get('coco_url',f'https://images.cocodataset.org/val2017/{i["file_name"]}')}
                       for i in coco['images'] if i['id'] in valid]}


def _image_relative_path(coco_url):
    """Take only well-formed COCO image paths from original metadata."""
    from pathlib import PurePosixPath
    parsed=urlparse(coco_url)
    if parsed.hostname != COCO_IMAGE_DOMAIN:
        raise ValueError(f'Non-COCO URL in annotation metadata: {coco_url}')
    path=PurePosixPath(parsed.path.lstrip('/'))
    if len(path.parts)!=2 or path.parts[0] not in ('train2017','val2017') or not path.name.endswith('.jpg'):
        raise ValueError(f'Unexpected COCO image path: {parsed.path}')
    return str(path)


# --- Fast COCO image downloads: cached JPEG resume + pooled HTTPS/HTTP connections ---
# The code below replaces ONLY download_images_from_manifest. Model/loss/train unchanged.
import threading
from requests.adapters import HTTPAdapter

_COCO_HTTP_LOCAL = threading.local()

def _coco_thread_session():
    """Thread-local session reuses TCP connections (avoids new requests.get per JPG)."""
    session = getattr(_COCO_HTTP_LOCAL, 'session', None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        session.headers.update({'User-Agent': 'SA-PE-DA-CenterNet-Research/1.0'})
        _COCO_HTTP_LOCAL.session = session
    return session


def _coco_cached_jpeg_complete(path, min_bytes=1024):
    """Fast restart check of already atomically-saved images.

    Downloaded JPGs are fully decoded before rename. After restart, SOI/EOI checks
    avoid re-decoding 16k JPG files just to learn what has already been saved.
    """
    try:
        path = Path(path)
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
        with path.open('rb') as f:
            start = f.read(3)
            f.seek(-2, 2)
            end = f.read(2)
        return start == b'\xff\xd8\xff' and end == b'\xff\xd9'
    except (OSError, ValueError):
        return False


def _coco_fast_download_one(url, dest, retries=2):
    """Validate-and-rename one COCO JPG; never treat partial downloads as complete."""
    _safe_download_url(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _coco_cached_jpeg_complete(dest):
        return dest
    # A partial JPG from a cancelled earlier session must never be accepted.
    if dest.exists():
        dest.unlink()
    tmp = dest.with_name(dest.name + '.part')
    session = _coco_thread_session()
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            tmp.unlink(missing_ok=True)
            with session.get(url, stream=True, timeout=(8, 35), allow_redirects=True) as r:
                r.raise_for_status()
                if urlparse(url).scheme == 'https' and urlparse(r.url).scheme != 'https':
                    raise ValueError('Refusing HTTPS-to-HTTP redirect')
                content_type = getattr(r, 'headers', {}).get('Content-Type', '')
                if 'text/html' in content_type.lower():
                    raise IOError(f'Server returned HTML instead of JPEG: {url}')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(256 * 1024):
                        if chunk:
                            f.write(chunk)
            if not _coco_cached_jpeg_complete(tmp):
                raise IOError(f'Incomplete/non-JPEG response: {url}')
            decoded = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise IOError(f'JPG decode failed: {url}')
            tmp.replace(dest)  # atomic on the same disk
            return dest
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt + 1 < max(1, retries):
                time.sleep(min(4., .75 * (2 ** attempt)))
    raise RuntimeError(f'{dest.name}: {type(last_error).__name__}: {last_error}')


def download_images_from_manifest(manifest, image_root, workers=16, report_every=250):
    """Fast resumable parallel downloader for selected original COCO JPGs.

    - Cache checks before networking (works with the existing 13k JPGs in /content).
    - Probe an actually MISSING image, not a cached one (fixes false origin probe).
    - Probe candidate COCO origins once; do not retry broken SSL origin for each JPG.
    - One requests.Session per thread with connection pooling.
    - as_completed progress shows completed downloads, not input-order blocking.
    - Raises on any missing/corrupt JPG; incomplete datasets never silently pass.
    """
    root = Path(image_root)
    root.mkdir(parents=True, exist_ok=True)
    entries = manifest['images']
    if not entries:
        raise ValueError('Empty COCO image manifest')
    seen = set()
    pending = []
    for item in entries:
        rel = _image_relative_path(item['coco_url'])
        filename = item['file_name']
        if filename != Path(rel).name or filename in seen:
            raise ValueError(f'Invalid/duplicate COCO manifest filename: {filename}')
        seen.add(filename)
        dest = root / filename
        if not _coco_cached_jpeg_complete(dest):
            pending.append((rel, dest))
    total = len(entries)
    cached = total - len(pending)
    print(f'COCO cache: {cached:,}/{total:,} JPG siap; perlu download: {len(pending):,}')
    if not pending:
        print(f'Images downloaded/verified: {total}/{total} (all cached)')
        return {'total': total, 'cached': cached, 'new': 0, 'failed': 0}

    # Use an uncached file as a genuine origin connectivity test.
    probe_rel, probe_dest = pending[0]
    origin = None
    errors = []
    for candidate in IMAGE_BASES:
        try:
            print(f'COCO server test: {candidate}', flush=True)
            _coco_fast_download_one(candidate + '/' + probe_rel, probe_dest, retries=1)
            origin = candidate
            break
        except Exception as exc:
            errors.append(f'{candidate}: {type(exc).__name__}: {str(exc)[:180]}')
            print('  Unavailable:', str(exc)[:130], flush=True)
    if origin is None:
        raise RuntimeError('No usable official COCO image endpoint:\n' + '\n'.join(errors))
    if origin.startswith('http://'):
        print('SECURITY NOTE: COCO official HTTP fallback is not encrypted; '
              'JPG signature+decode checks detect corruption, not malicious substitution.')
    print(f'Using one working origin: {origin}', flush=True)

    workers = max(1, min(24, int(workers)))
    remaining = pending[1:]
    ok = 1  # probe already saved one image successfully
    failed = []
    start = time.monotonic()
    print(f'Workers: {workers}; completed at start: {cached + 1:,}/{total:,}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_coco_fast_download_one, origin + '/' + rel, dest, 2): (rel, dest)
            for rel, dest in remaining
        }
        processed = 0
        for future in concurrent.futures.as_completed(futures):
            rel, dest = futures[future]
            processed += 1
            try:
                future.result()
                ok += 1
            except Exception as exc:
                failed.append(f'{rel}: {exc}')
            if processed % report_every == 0 or processed == len(remaining):
                elapsed = max(time.monotonic() - start, 1e-3)
                speed = processed / elapsed
                print(f'Images downloaded/verified: {cached + ok}/{total} | '
                      f'failed={len(failed)} | {speed:.1f} files/s', flush=True)

    if failed:
        error_log = root / '_sape_da_coco_download_errors.txt'
        error_log.write_text('\n'.join(failed), encoding='utf-8')
        raise RuntimeError(f'{len(failed)} COCO JPG downloads failed; saved detailed errors: '
                           f'{error_log}. Re-run this cell to retry only missing JPGs. '
                           f'First issue: {failed[0]}')
    assert cached + ok == total, 'Downloader completeness check failed'
    print(f'DONE: {total:,}/{total:,} valid COCO JPGs in {root}', flush=True)
    return {'total': total, 'cached': cached, 'new': ok, 'failed': 0, 'origin': origin}


def letterbox(bgr, boxes, width=1280,height=736, active_h=720):
    """Rescale whole image into 1280×720 content canvas with same aspect ratio,
    then center pad to 1280×736. No object cropping or geometric distortion.
    Actual active region may have extra side/top bars for non-16:9 sources.
    BGR constant pad ~= CenterNet mean BGR.
    """
    orig_h,orig_w=bgr.shape[:2]
    assert width%32==0 and height%32==0 and active_h<=height
    scale=min(width/orig_w,active_h/orig_h)
    new_w=max(1,int(round(orig_w*scale)))
    new_h=max(1,int(round(orig_h*scale)))
    resized=cv2.resize(bgr,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    pad_x=(width-new_w)//2
    pad_y=(height-new_h)//2
    canvas=np.empty((height,width,3),dtype=np.uint8)
    canvas[:]=np.round(MEAN*255).astype(np.uint8)
    canvas[pad_y:pad_y+new_h,pad_x:pad_x+new_w]=resized
    result=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy()
    if len(result):
        result[:,[0,2]]=result[:,[0,2]]*scale+pad_x
        result[:,[1,3]]=result[:,[1,3]]*scale+pad_y
        result[:,[0,2]]=result[:,[0,2]].clip(0,width)
        result[:,[1,3]]=result[:,[1,3]].clip(0,height)
    meta={'scale':scale,'pad_x':pad_x,'pad_y':pad_y,'orig_h':orig_h,'orig_w':orig_w}
    normalized=((canvas.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy()
    return torch.from_numpy(normalized),result,meta


def invert_letterbox_xyxy(boxes,meta):
    b=boxes.copy()
    b[:,[0,2]]=(b[:,[0,2]]-meta['pad_x'])/meta['scale']
    b[:,[1,3]]=(b[:,[1,3]]-meta['pad_y'])/meta['scale']
    b[:,[0,2]]=b[:,[0,2]].clip(0,meta['orig_w'])
    b[:,[1,3]]=b[:,[1,3]].clip(0,meta['orig_h'])
    return b


def gaussian_radius(hw,min_overlap=0.7):
    h,w=hw
    a1,b1,c1=1,h+w,w*h*(1-min_overlap)/(1+min_overlap)
    r1=(b1+math.sqrt(max(0,b1*b1-4*a1*c1)))/(2*a1)
    a2,b2,c2=4,2*(h+w),(1-min_overlap)*w*h
    r2=(b2+math.sqrt(max(0,b2*b2-4*a2*c2)))/(2*a2)
    a3,b3,c3=4*min_overlap,-2*min_overlap*(h+w),(min_overlap-1)*w*h
    r3=(b3+math.sqrt(max(0,b3*b3-4*a3*c3)))/(2*a3)
    return min(r1,r2,r3)


def draw_gaussian(heatmap,point,radius):
    x,y=int(point[0]),int(point[1]);diam=2*radius+1
    sigma=diam/6.
    axis=np.arange(-radius,radius+1)
    gauss=np.exp(-(axis[:,None]**2+axis[None,:]**2)/(2*sigma*sigma))
    h,w=heatmap.shape
    left,right=min(x,radius),min(w-x-1,radius)
    top,bottom=min(y,radius),min(h-y-1,radius)
    if min(left,right,top,bottom)<0: return
    roi=heatmap[y-top:y+bottom+1,x-left:x+right+1]
    np.maximum(roi,gauss[radius-top:radius+bottom+1,radius-left:radius+right+1],out=roi)


def build_targets(boxes,width=1280,height=736,stride=4,max_objs=256):
    out_h,out_w=height//stride,width//stride
    hm=np.zeros((1,out_h,out_w),np.float32)
    wh=np.zeros((max_objs,2),np.float32)
    reg=np.zeros((max_objs,2),np.float32)
    ind=np.zeros((max_objs,),np.int64)
    mask=np.zeros((max_objs,),np.float32)
    count=0
    for x1,y1,x2,y2 in boxes:
        if count>=max_objs: break
        x1,x2=np.clip([x1,x2],0,width)/stride
        y1,y2=np.clip([y1,y2],0,height)/stride
        w,h=x2-x1,y2-y1
        if w<=0 or h<=0: continue
        cx,cy=(x1+x2)/2,(y1+y2)/2
        ix,iy=int(cx),int(cy)
        ix=min(ix,out_w-1);iy=min(iy,out_h-1)
        radius=max(0,int(gaussian_radius((math.ceil(h),math.ceil(w)))))
        draw_gaussian(hm[0],(ix,iy),radius)
        wh[count]=[w,h];reg[count]=[cx-ix,cy-iy]
        ind[count]=iy*out_w+ix;mask[count]=1;count+=1
    return {'hm':torch.from_numpy(hm),'wh':torch.from_numpy(wh),
            'reg':torch.from_numpy(reg),'ind':torch.from_numpy(ind),
            'mask':torch.from_numpy(mask)}


class PersonDetectionDataset(Dataset):
    """Reads COCO format annotations for either COCO or CVAT-exported CCTV.
    CVAT categories may use a different category ID; resolve by category name.
    Coordinates ground truth: COCO [x,y,width,height] and category 'person'.
    """
    def __init__(self,annotation_json,image_dir,selected_ids=None,train=False,
                 width=1280,height=736,active_h=720):
        self.raw=load_coco_json(annotation_json)
        self.images={int(i['id']):i for i in self.raw['images']}
        person_ids={int(c['id']) for c in self.raw.get('categories',[]) if c.get('name','').lower()=='person'}
        if not person_ids:
            raise ValueError('Annotations must contain a category with name="person"')
        self.anns=defaultdict(list)
        for a in self.raw['annotations']:
            if a.get('category_id') in person_ids and not a.get('iscrowd',0):
                x,y,w,h=a['bbox']
                if w>1 and h>1:self.anns[int(a['image_id'])].append([x,y,x+w,y+h])
        ids=sorted(self.images) if selected_ids is None else sorted(int(i) for i in selected_ids)
        self.ids=[i for i in ids if i in self.images]
        if not self.ids: raise ValueError('Dataset has no selected image IDs')
        self.image_dir=Path(image_dir);self.train=train
        self.width,self.height,self.active_h=width,height,active_h
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        iid=self.ids[index]
        image=self.images[iid]
        path=self.image_dir/image['file_name']
        bgr=cv2.imread(str(path))
        if bgr is None: raise FileNotFoundError(f'Cannot read image {path}')
        boxes=np.asarray(self.anns.get(iid,[]),dtype=np.float32).reshape(-1,4)
        if self.train and random.random()<0.5:
            bgr=cv2.flip(bgr,1)
            if len(boxes):
                x1=self.images[iid].get('width',bgr.shape[1])-boxes[:,2]
                x2=self.images[iid].get('width',bgr.shape[1])-boxes[:,0]
                boxes[:,0],boxes[:,2]=x1.copy(),x2.copy()
        tensor,boxes_in,meta=letterbox(bgr,boxes,self.width,self.height,self.active_h)
        targets=build_targets(boxes_in,self.width,self.height)
        # Meta/GT retained for evaluation outside default batch collate
        return {'image':tensor,**targets,'image_id':iid}


def sample_cctv_support_ids(annotation_json,shots,seed=42):
    raw=load_coco_json(annotation_json)
    person_ids={int(c['id']) for c in raw['categories'] if c['name'].lower()=='person'}
    populated={int(a['image_id']) for a in raw['annotations'] if a['category_id'] in person_ids
               and a['bbox'][2]>1 and a['bbox'][3]>1}
    if len(populated)<shots:
        raise ValueError(f'CCTV train has only {len(populated)} annotated person images; need {shots}')
    return sorted(random.Random(seed).sample(sorted(populated),shots))

# ---------- LOCKED COCO-80 multi-class source targets ----------
OFFICIAL_COCO80_IDS = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
                      22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,
                      43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
                      62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,
                      85,86,87,88,89,90)


def build_multiclass_targets(boxes, labels, width=1280, height=736, stride=4,max_objs=1024):
    """Genuine 80-channel COCO heatmaps with class-agnostic wh/offset regression."""
    h,w=height//stride,width//stride
    hm=np.zeros((80,h,w),np.float32)
    wh=np.zeros((max_objs,2),np.float32)
    reg=np.zeros((max_objs,2),np.float32)
    ind=np.zeros(max_objs,np.int64)
    mask=np.zeros(max_objs,np.float32)
    count=0
    if len(boxes)!=len(labels):raise ValueError('bbox/label length mismatch')
    for bb,cls in zip(boxes,labels):
        if not (0<=int(cls)<80):raise ValueError(f'Invalid contiguous COCO index {cls}')
        if count>=max_objs:
            raise RuntimeError(f'More than {max_objs} target instances in image. Increase max_objs; do not silently truncate.')
        x1,y1,x2,y2=map(float,bb)
        x1,x2=np.clip([x1,x2],0,width)/stride
        y1,y2=np.clip([y1,y2],0,height)/stride
        bw,bh=x2-x1,y2-y1
        if bw<=0 or bh<=0:continue
        cx,cy=(x1+x2)/2,(y1+y2)/2
        ix=min(max(int(cx),0),w-1);iy=min(max(int(cy),0),h-1)
        radius=max(0,int(gaussian_radius((math.ceil(bh),math.ceil(bw)))))
        draw_gaussian(hm[int(cls)],(ix,iy),radius)
        wh[count]=[bw,bh];reg[count]=[cx-ix,cy-iy]
        ind[count]=iy*w+ix;mask[count]=1;count+=1
    return {k:torch.from_numpy(v) for k,v in
            {'hm':hm,'wh':wh,'reg':reg,'ind':ind,'mask':mask}.items()}


class COCO80Dataset(Dataset):
    """Official COCO category IDs -> 80 contiguous heatmap channels, preserves all boxes."""
    def __init__(self, annotation_json, image_dir, selected_ids=None,train=True,
                 width=1280,height=736,active_h=720):
        raw=load_coco_json(annotation_json)
        categories=sorted(int(c['id']) for c in raw['categories'])
        if tuple(categories)!=OFFICIAL_COCO80_IDS:
            raise ValueError('COCO source must have the official 80 category IDs')
        mapping={cat:i for i,cat in enumerate(categories)}
        self.images={int(i['id']):i for i in raw['images']}
        self.anns=defaultdict(list)
        for a in raw['annotations']:
            if int(a.get('iscrowd',0)) or int(a['category_id']) not in mapping:continue
            x,y,bw,bh=a['bbox']
            if bw<=1 or bh<=1 or a.get('area',bw*bh)<5:continue
            self.anns[int(a['image_id'])].append(([x,y,x+bw,y+bh],mapping[int(a['category_id'])]))
        self.ids=[int(i) for i in (sorted(self.images) if selected_ids is None else selected_ids)]
        if len(set(self.ids))!=len(self.ids) or any(i not in self.images for i in self.ids):
            raise ValueError('Invalid or duplicated source IDs')
        self.image_dir=Path(image_dir);self.train=train
        self.width,self.height,self.active_h=width,height,active_h
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        iid=self.ids[index]
        im=self.images[iid];path=self.image_dir/im['file_name']
        bgr=cv2.imread(str(path))
        if bgr is None:raise FileNotFoundError(f'Cannot read COCO JPG: {path}')
        anns=self.anns.get(iid,[])
        boxes=np.asarray([b for b,_ in anns],dtype=np.float32).reshape(-1,4)
        classes=np.asarray([c for _,c in anns],dtype=np.int64)
        if self.train and random.random()<0.5:
            bgr=cv2.flip(bgr,1)
            if len(boxes):
                ix1=bgr.shape[1]-boxes[:,2];ix2=bgr.shape[1]-boxes[:,0]
                boxes[:,0],boxes[:,2]=ix1.copy(),ix2.copy()
        tensor,transformed,_=letterbox(bgr,boxes,self.width,self.height,self.active_h)
        targets=build_multiclass_targets(transformed,classes,self.width,self.height)
        counts=np.bincount(classes,minlength=80).astype(np.int32)
        return {'image':tensor,**targets,'image_id':iid,
                'class_counts':torch.from_numpy(counts)}
