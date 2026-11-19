%%writefile sape_da/cctv_io.py
"""Prepare CCTV splits directly from user's Google Drive folder structure.

Input:
  MyDrive/DATASET/GAMBAR CCTV/{TRAIN,VALIDATION,TESTING}/{IMAGE,LABEL}

Supported LABEL formats:
  - COCO object-detection JSON (categories/images/annotations)
  - YOLO detection TXT: class_id x_center y_center width height, normalized 0..1.

The source dataset remains on Drive; working images and generated COCO JSON are
placed on /content to avoid slow training reads from mounted Google Drive.
No labels are guessed: unsupported/ambiguous annotation formats fail loudly.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = {'train': 'TRAIN', 'val': 'VALIDATION', 'test': 'TESTING'}
PERSON_NAMES = {'person'}


def _image_entries(images_folder):
    if not images_folder.is_dir():
        raise FileNotFoundError(f'CCTV IMAGE folder not found: {images_folder}')
    images = sorted((p for p in images_folder.rglob('*')
                     if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda p: str(p.relative_to(images_folder)).casefold())
    if not images:
        raise RuntimeError(f'No image files found in {images_folder}')
    # Prevent ambiguous pairing between IMAGE and LABEL, even if nested paths exist.
    stems = {}
    for p in images:
        key = p.stem.casefold()
        if key in stems:
            raise ValueError(f'Duplicate image stem {p.stem!r}: {stems[key]} and {p}; '
                             'rename image files and matching labels uniquely')
        stems[key] = p
    return images


def _coalesce_formats(label_folder):
    if not label_folder.is_dir():
        raise FileNotFoundError(f'CCTV LABEL folder not found: {label_folder}')
    candidates = []
    for p in sorted(label_folder.rglob('*.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8-sig'))
        except Exception:
            continue
        if isinstance(d, dict) and all(k in d for k in ('images','annotations','categories')):
            candidates.append((p,d))
    if len(candidates) > 1:
        raise ValueError('Multiple valid COCO annotation JSON files found: '
                         + ', '.join(str(p) for p,_ in candidates))
    if candidates:
        return 'coco', candidates[0]
    txt = [p for p in label_folder.rglob('*.txt')
           if p.name.casefold() not in {'classes.txt', 'obj.names', 'train.txt', 'val.txt', 'test.txt', 'notes.txt'}]
    if txt:
        return 'yolo', txt
    raise ValueError(f'No supported annotations in {label_folder}. '
                     'Use COCO detection JSON or normalized YOLO bounding-box TXT. '
                     'CVAT XML, segmentation polygons and label-only classification '
                     'cannot be used for CenterNet object detection without conversion.')


def _copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Always refresh source image; this avoids silent stale copies on repeat runs.
    shutil.copy2(src, dst)
    try:
        with Image.open(dst) as im:
            w,h = im.size
            im.verify()
    except Exception as exc:
        dst.unlink(missing_ok=True)
        raise ValueError(f'Unreadable image: {src}') from exc
    if w < 16 or h < 16:
        raise ValueError(f'Image too small: {src} ({w}x{h})')
    return w,h


def _person_from_yolo(label_path, w, h, person_class_id):
    annotations=[]
    for line_no,line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(),1):
        line=line.strip()
        if not line:continue
        pieces=line.split()
        if len(pieces)!=5:
            raise ValueError(f'{label_path}:{line_no} expected YOLO detection 5 fields '
                             '(class xc yc w h), got {len(pieces)}. '
                             'Do not use classification or segmentation labels.')
        try:
            cls=int(pieces[0]); xc,yc,bw,bh=(float(v) for v in pieces[1:])
        except Exception as exc:
            raise ValueError(f'Invalid YOLO box: {label_path}:{line_no}: {line}') from exc
        if not all(0 <= v <= 1 for v in (xc,yc,bw,bh)) or bw<=0 or bh<=0:
            raise ValueError(f'YOLO coordinates must be normalized 0..1 with positive w/h: '
                             f'{label_path}:{line_no}: {line}')
        if cls != person_class_id:
            continue
        x1=max(0.0,(xc-bw/2)*w);y1=max(0.0,(yc-bh/2)*h)
        x2=min(float(w),(xc+bw/2)*w);y2=min(float(h),(yc+bh/2)*h)
        if x2-x1 < 1 or y2-y1 < 1:
            continue
        annotations.append([x1,y1,x2-x1,y2-y1])
    return annotations


def _person_from_coco(raw, image_paths, image_sizes, expected_names=PERSON_NAMES):
    categories={int(c['id']) for c in raw['categories']
                if str(c.get('name','')).strip().casefold() in expected_names}
    if not categories:
        found=[str(c.get('name')) for c in raw['categories']]
        raise ValueError(f'No category named person in COCO annotations. Categories: {found}')
    # Reference both stem and full file name, validating duplicates.
    indexed_by_name = {p.name.casefold():p.stem.casefold() for p in image_paths}
    by_source_id={}
    for im in raw['images']:
        name=Path(str(im['file_name'])).name.casefold()
        if name in indexed_by_name:
            key=indexed_by_name[name]
            if key in by_source_id.values():
                raise ValueError(f'Duplicate image entry in COCO JSON: {im["file_name"]}')
            by_source_id[int(im['id'])]=key
    missing_names=[im.get('file_name','') for im in raw['images']
                   if Path(str(im.get('file_name',''))).name.casefold() not in indexed_by_name]
    if missing_names:
        raise ValueError(f'{len(missing_names)} JSON images missing from IMAGE folder; '
                         f'first: {missing_names[:3]}')
    missing_metadata=set(image_sizes)-set(by_source_id.values())
    if missing_metadata:
        raise ValueError(f'COCO JSON omits {len(missing_metadata)} images present in IMAGE folder, '
                         f'e.g. {sorted(missing_metadata)[:3]}. Include every evaluated image in JSON.')
    result={p.stem.casefold():[] for p in image_paths}
    for a in raw['annotations']:
        if int(a['category_id']) not in categories or a.get('iscrowd',0):
            continue
        src_id=int(a['image_id'])
        if src_id not in by_source_id:
            raise ValueError(f'Annotation references missing image_id={src_id}')
        key=by_source_id[src_id]
        x,y,bw,bh = (float(v) for v in a['bbox'])
        w,h=image_sizes[key]
        x1=max(0.0,x); y1=max(0.0,y)
        x2=min(float(w),x+bw); y2=min(float(h),y+bh)
        if x2-x1>=1 and y2-y1>=1:
            result[key].append([x1,y1,x2-x1,y2-y1])
    return result


def prepare_cctv_split(source_root, destination_root, split, yolo_person_class_id=0,
                       allow_missing_yolo_labels=False):
    """Materialize Drive split as COCO JSON + images, with explicit validation.

    yolo_person_class_id=0 is the usual YOLO person class, but must be confirmed
    against the user's exported label convention. It is NOT auto-inferred.
    """
    if split not in SPLITS:raise ValueError(f'Unknown split: {split}')
    source_root=Path(source_root)
    image_dir=source_root/SPLITS[split]/'IMAGE'
    labels_dir=source_root/SPLITS[split]/'LABEL'
    images=_image_entries(image_dir)
    fmt,payload=_coalesce_formats(labels_dir)
    dest=Path(destination_root)/split
    dest_images=dest/'images'
    dest_images.mkdir(parents=True,exist_ok=True)
    stem_index={}
    coco_images=[]
    for j,source in enumerate(images,1):
        # Unique names in the local working folder regardless of source nesting.
        out_name=f'{j:06d}_{source.name}'
        w,h=_copy_image(source,dest_images/out_name)
        stem_index[source.stem.casefold()]={'id':j,'file_name':out_name,'width':w,'height':h}
        coco_images.append(stem_index[source.stem.casefold()])
    boxes_by_stem={key:[] for key in stem_index}
    if fmt=='yolo':
        class_names_file=labels_dir/'classes.txt'
        if class_names_file.is_file():
            class_names=[v.strip().casefold() for v in class_names_file.read_text(encoding='utf-8-sig').splitlines()]
            if yolo_person_class_id >= len(class_names) or class_names[yolo_person_class_id]!='person':
                raise ValueError(f'YOLO person_class_id={yolo_person_class_id} conflicts with '
                                 f'{class_names_file}: {class_names}. Set the correct class ID.')
        txt_files=payload
        all_label_stems={}
        for p in txt_files:
            key=p.stem.casefold()
            if key in all_label_stems:
                raise ValueError(f'Duplicate YOLO label stem: {p} and {all_label_stems[key]}')
            all_label_stems[key]=p
        unknown=set(all_label_stems)-set(stem_index)
        if unknown:raise ValueError(f'Labels without matching images: {sorted(unknown)[:5]}')
        missing=set(stem_index)-set(all_label_stems)
        if missing and not allow_missing_yolo_labels:
            raise ValueError(f'{len(missing)} images have no corresponding .txt label '
                             f'(e.g. {sorted(missing)[:3]}). Add empty TXT for confirmed '
                             'negative frames or set allow_missing_yolo_labels=True explicitly.')
        for stem, entry in stem_index.items():
            if stem in all_label_stems:
                boxes_by_stem[stem]=_person_from_yolo(
                    all_label_stems[stem],entry['width'],entry['height'],yolo_person_class_id)
        # Fail closed if there are no annotations of person at all.
        if not any(boxes_by_stem.values()):
            raise ValueError(f'No person bounding boxes from YOLO class '
                             f'{yolo_person_class_id}. Check the class_id in LABEL files.')
    else:
        _,raw=payload
        sizes={stem:(d['width'],d['height']) for stem,d in stem_index.items()}
        boxes_by_stem=_person_from_coco(raw,images,sizes)
    annotations=[]
    for stem,entry in stem_index.items():
        for box in boxes_by_stem[stem]:
            annotations.append({'id':len(annotations)+1,'image_id':entry['id'],
                                'category_id':1,'bbox':box,'area':box[2]*box[3],
                                'iscrowd':0,'segmentation':[]})
    if not annotations:
        raise ValueError(f'No valid person boxes in {source_root/SPLITS[split]/"LABEL"}')
    coco={'info':{'description':f'SA-PE-DA CCTV {split} sourced from Drive'},
          'licenses':[], 'images':coco_images,'annotations':annotations,
          'categories':[{'id':1,'name':'person','supercategory':'person'}]}
    dest.mkdir(parents=True,exist_ok=True)
    ann_path=dest/'annotations.json'
    tmp=ann_path.with_suffix('.json.part')
    tmp.write_text(json.dumps(coco,ensure_ascii=False),encoding='utf-8')
    tmp.replace(ann_path)
    summary={'split':split,'format':fmt,'images':len(coco_images),
             'person_instances':len(annotations),'annotation_json':str(ann_path),
             'images_folder':str(dest_images)}
    return summary


def prepare_all_cctv(source_root,destination_root,yolo_person_class_id=0,
                     allow_missing_yolo_labels=False):
    source_root=Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f'CCTV folder not available: {source_root}; '
                                'mount Drive and check Google account / folder shortcut.')
    summaries={}
    for split in SPLITS:
        summaries[split]=prepare_cctv_split(
            source_root,destination_root,split,
            yolo_person_class_id=yolo_person_class_id,
            allow_missing_yolo_labels=allow_missing_yolo_labels)
        print(f"CCTV {split}: {summaries[split]['images']} images, "
              f"{summaries[split]['person_instances']} person instances, "
              f"LABEL format={summaries[split]['format']}")
    return summaries
