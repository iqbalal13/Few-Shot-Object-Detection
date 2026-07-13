# ==========================================================
# STEP 17 : Load COCO Annotation
# ==========================================================

from pycocotools.coco import COCO

annotation_file = "/content/datasets/coco/annotations/instances_train2017.json"

coco = COCO(annotation_file)

print("=" * 60)
print("COCO Annotation Loaded Successfully")
print("=" * 60)

print("Number of Images     :", len(coco.imgs))
print("Number of Categories :", len(coco.cats))
print("Number of Annotations:", len(coco.anns))
