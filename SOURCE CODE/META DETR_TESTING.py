# CONTOH KODE SELANJUTNYA

from PIL import Image
import requests
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# 1. Siapkan gambar (contoh: dari URL)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# 2. Muat model dan prosesor yang sudah di-download
# (Karena Anda sudah login, ini akan berjalan lancar)
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# 3. Proses gambar dan lakukan deteksi
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 4. Konversi hasil deteksi ke format yang bisa dibaca
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# 5. Tampilkan hasilnya
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
