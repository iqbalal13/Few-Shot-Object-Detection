from sape_da.model import SAPEDACenterNet
from sape_da.locked_train import load_locked_checkpoint

source_model = SAPEDACenterNet()
source_ck = load_locked_checkpoint(SOURCE_FINAL,source_model)
assert source_ck.get('phase') == 'source', 'Checkpoint yang dipilih bukan source/base model.'
assert int(source_ck['step']) == 8000, 'Source checkpoint bukan eksperimen COCO-80 tepat 1 epoch.'
metadata = source_ck.get('metadata',{})
assert len(metadata.get('unique_image_ids_seen',[])) == 8000
assert int(metadata.get('source_classes_seen',0)) == 80
assert int(metadata.get('max_optimizer_steps',0)) == 8000
assert source_model.base.hm[-1].out_channels == 80
print('BASE CHECKPOINT READY: 80 COCO classes, 8,000 unique images, 8,000 updates.')
print('SOURCE TRAINING SKIPPED — NO optimizer step on COCO.')
del source_model,source_ck
