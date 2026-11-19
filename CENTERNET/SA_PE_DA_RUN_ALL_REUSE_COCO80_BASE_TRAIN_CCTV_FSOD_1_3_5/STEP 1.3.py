%%writefile sape_da/shared.py
"""Shared Meta-DETR v8.1 COCO-80 sampling semantics, independently vendored.

The category-wise quota/union/person-top-up algorithm below is intentionally
identical to MetaDETR_TwoStage_v8_1_LOCKED_G3_FIXED overlay's
metadetr_pipeline.data_protocol.source_classwise_quota, seed 42.
A final source_manifest.json is Meta-DETR compatible via image_ids.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import re


def source_classwise_quota(coco, seed=42, quota_per_class=100, total_images=8000,
                           external_manifest=None):
    if quota_per_class < 1 or total_images < 1:
        raise ValueError('quota_per_class and total_images must be positive')
    categories = sorted(int(c['id']) for c in coco['categories'])
    if len(categories) != 80 or 1 not in categories:
        raise ValueError('Expected all 80 COCO official classes, including person=1')
    images = {int(im['id']) for im in coco['images']}
    by_class = defaultdict(set)
    for ann in coco['annotations']:
        if (int(ann['category_id']) in categories and int(ann['image_id']) in images
                and not int(ann.get('iscrowd', 0)) and float(ann.get('area', 0)) >= 5):
            by_class[int(ann['category_id'])].add(int(ann['image_id']))
    if external_manifest:
        ext = json.loads(Path(external_manifest).read_text(encoding='utf-8'))
        ids = [int(x) for x in (ext['image_ids'] if isinstance(ext,dict) else ext)]
        if len(ids)!=total_images or len(set(ids))!=total_images or not set(ids).issubset(images):
            raise ValueError('Shared manifest must contain exactly 8,000 distinct COCO image IDs')
        source='external_shared_manifest'; first_stage_counts={str(c):None for c in categories};union_size=None
    else:
        rng=random.Random(seed);initial=set();first_stage_counts={}
        for cid in categories:
            candidates=sorted(by_class[cid]);n=min(quota_per_class,len(candidates))
            first_stage_counts[str(cid)]=n
            initial.update(rng.sample(candidates,n))
        union_size=len(initial)
        if union_size>total_images:raise ValueError('Quota union larger than 8,000')
        remaining=sorted(by_class[1]-initial);topup=total_images-union_size
        if len(remaining)<topup:raise ValueError('Not enough person images for Meta-DETR top-up')
        initial.update(rng.sample(remaining,topup));ids=sorted(initial)
        source='classwise_100_then_person_topup'
    selected=set(ids)
    image_counts={str(cid):len(by_class[cid]&selected) for cid in categories}
    missing=[cid for cid in categories if image_counts[str(cid)]==0]
    if missing:raise ValueError(f'Source manifest omits categories {missing}')
    return ids, {'selection_mode':source,'seed':seed,'quota_per_category':quota_per_class,
            'target_unique_images':total_images,'actual_unique_images':len(ids),
            'initial_category_draws':first_stage_counts,'initial_union_unique_images':union_size,
            'person_topup_count':None if union_size is None else total_images-union_size,
            'category_image_counts':image_counts,'person_image_count':image_counts['1']}


def create_shared_manifest(annotation_json, out_json, seed=42, external_manifest=None):
    coco=json.loads(Path(annotation_json).read_text())
    ids,stats=source_classwise_quota(coco,seed=seed,external_manifest=external_manifest)
    selected=set(ids)
    cats=sorted(int(c['id']) for c in coco['categories'])
    counts=Counter(int(a['category_id']) for a in coco['annotations']
        if int(a['image_id']) in selected and not a.get('iscrowd',0) and float(a.get('area',0))>=5)
    missing=[c for c in cats if counts[c]==0]
    if missing:raise ValueError(f'Manifest has zero eligible instances for {missing}')
    id_hash=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    result={'protocol':'SA-PE-DA-LOCKED-COCO80-v1; MetaDETR-LOCKED-v2.1 matching sampling',
        'image_ids':ids,'stats':stats,'counts_per_category':{str(c):counts[c] for c in cats},
        'images_per_category':stats['category_image_counts'],
        'rare_categories_lt_10_instances':[c for c in cats if counts[c]<10],
        'image_ids_sha256':id_hash}
    p=Path(out_json);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        previous=json.loads(p.read_text())
        if previous.get('image_ids')!=ids:raise RuntimeError('Existing source manifest does not match current locked sampling. Stop; do not overwrite.')
    p.write_text(json.dumps(result,indent=2))
    return result


def coverage_first_order(annotation_json, image_ids, seed=42):
    """Same 8k images, reorder to expose all 80 categories early; no added images.

    Meta-DETR v8.1 may use a different batch order. The shared manifest remains
    identical but this scheduling difference must be disclosed.
    """
    coco=json.loads(Path(annotation_json).read_text())
    selected=set(map(int,image_ids));eligible=defaultdict(set)
    for a in coco['annotations']:
        if (int(a['image_id']) in selected and not a.get('iscrowd',0)
                and float(a.get('area',0))>=5
                and a.get('bbox') is not None and len(a['bbox'])==4
                and float(a['bbox'][2])>1 and float(a['bbox'][3])>1):
            eligible[int(a['image_id'])].add(int(a['category_id']))
    allclasses={int(c['id']) for c in coco['categories']};uncovered=set(allclasses)
    first=[];remaining=set(selected)
    while uncovered:
        # Deterministic greedy set cover, prefer more uncovered categories.
        candidates=sorted(remaining,key=lambda i:(-len(eligible[i]&uncovered),i))
        if not candidates or not (eligible[candidates[0]]&uncovered):
            raise ValueError(f'Cannot expose all 80 classes: missing {uncovered}')
        chosen=candidates[0];first.append(chosen);remaining.remove(chosen)
        uncovered-=eligible[chosen]
    rest=sorted(remaining);random.Random(seed).shuffle(rest)
    return first+rest,{'coverage_prefix_images':len(first),'category_count':len(allclasses)}


def source_download_manifest(annotation_json, image_ids):
    coco=json.loads(Path(annotation_json).read_text())
    im={int(x['id']):x for x in coco['images']}
    entries=[]
    for iid in image_ids:
        i=im[int(iid)]
        entries.append({'id':int(iid),'file_name':i['file_name'],
            'coco_url':i.get('coco_url',f'https://images.cocodataset.org/train2017/{i["file_name"]}')})
    return {'images':entries}


def val_download_manifest(annotation_json):
    coco=json.loads(Path(annotation_json).read_text())
    return {'images':[{'id':int(i['id']),'file_name':i['file_name'],
              'coco_url':i.get('coco_url',f'https://images.cocodataset.org/val2017/{i["file_name"]}')}
                     for i in coco['images']]}


def canonical_cctv_filename(name):
    """Undo only the deterministic six-digit prefix added by this CCTV importer."""
    return re.sub(r'^\d{6}_', '', Path(name).name)


def shared_cctv_support(train_json,save_path,seed=42):
    """Filename-first nested K-shot selections, portable between model pipelines."""
    coco=json.loads(Path(train_json).read_text())
    person={int(c['id']) for c in coco['categories'] if c['name'].strip().lower()=='person'}
    valid={int(a['image_id']) for a in coco['annotations'] if int(a['category_id']) in person
           and not a.get('iscrowd',0) and a['bbox'][2]>=1 and a['bbox'][3]>=1}
    images={int(i['id']):canonical_cctv_filename(i['file_name']) for i in coco['images']}
    # CCTV import names can have an NN_ prefix. Prefer standardized original filenames
    # if present; write BOTH IDs and filenames for audit and inter-model conversion.
    candidates=sorted(valid,key=lambda i:images[i].casefold())
    if len(candidates)<5:raise ValueError('Need 5 person-annotated CCTV train images')
    rng=random.Random(seed);rng.shuffle(candidates)
    chosen=candidates[:5]
    manifest={'seed':seed,'shot_unit':'fully_annotated_image','nested':True,
        'shots':{str(k):[images[i] for i in chosen[:k]] for k in (1,3,5)}}
    path=Path(save_path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=json.loads(path.read_text())
        if old['shots']!=manifest['shots']:raise RuntimeError('Existing CCTV support manifest conflict; refuse overwrite')
    path.write_text(json.dumps(manifest,indent=2))
    return manifest


def resolve_support_ids(train_json,shared_manifest,shots):
    images=json.loads(Path(train_json).read_text())['images']
    byname={canonical_cctv_filename(i['file_name']):int(i['id']) for i in images}
    entry=json.loads(Path(shared_manifest).read_text())
    names=entry['shots'][str(shots)]
    if len(names)!=shots:raise ValueError('shot manifest count mismatch')
    missing=[n for n in names if n not in byname]
    if missing:raise ValueError(f'Support filenames absent from dataset: {missing}')
    return [byname[n] for n in names]
