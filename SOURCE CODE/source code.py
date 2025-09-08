# ==============================================================================
# 
#  SOURCE CODE FINAL - OTOMATIS VIA HUGGING FACE (PALING STABIL)
# 
# ==============================================================================

# LANGKAH A: SETUP ENVIRONMENT DAN PROYEK
# ==============================================================================
print("🚀 [1/5] Menyiapkan environment dan instalasi...")
from google.colab import drive
import os
import json
import cv2
import shutil

# Hubungkan Google Drive
if not os.path.exists('/content/drive/My Drive'):
    drive.mount('/content/drive')

# Definisikan folder kerja proyek yang BERSIH dan BARU (tanpa spasi)
project_dir = '/content/drive/MyDrive/Proyek_MetaDETR'
os.makedirs(project_dir, exist_ok=True)
%cd {project_dir}

# Clone repositori Meta-DETR ke dalam folder proyek jika belum ada
repo_dir = os.path.join(project_dir, 'Meta-DETR')
if not os.path.exists(repo_dir):
    print("Meng-clone repositori Meta-DETR...")
    !git clone https://github.com/ZhangGongjie/Meta-DETR.git
# Pindah ke dalam direktori repo untuk instalasi
%cd {repo_dir}

# Instalasi semua library yang dibutuhkan, termasuk huggingface_hub
print("Menginstal library...")
!pip install -q -r requirements.txt
!pip install -q pycocotools submitit gdown opencv-python huggingface_hub

print("✅ [1/5] Setup environment selesai.")

# ---

# LANGKAH B: UNDUH MODEL DARI HUGGING FACE HUB (OTOMATIS)
# ==============================================================================
print("\n🚀 [2/5] Mengunduh model dari Hugging Face Hub (Cara Stabil)...")
from huggingface_hub import hf_hub_download

# Path tujuan untuk menyimpan model
model_path = os.path.join(repo_dir, 'metadetr_coco.pth')

if not os.path.exists(model_path):
    print("Model tidak ditemukan, mengunduh dari Hugging Face...")
    # Unduh file dari repo "tsmatz/meta-detr"
    downloaded_path = hf_hub_download(
        repo_id="tsmatz/meta-detr",
        filename="metadetr_coco.pth"
    )
    # Pindahkan file dari cache Hugging Face ke folder proyek kita
    shutil.move(downloaded_path, model_path)
    print("✅ Model berhasil diunduh dan dipindahkan.")
else:
    print("✅ Model pre-trained sudah ada.")


# ---

# LANGKAH C: FUNGSI KONVERSI (TIDAK PERLU DIUBAH)
# ==============================================================================
def convert_yolo_to_coco(img_folder, label_folder, output_json_path, class_names):
    coco_format = {"images": [], "annotations": [], "categories": []}
    for i, class_name in enumerate(class_names):
        coco_format["categories"].append({"id": i, "name": class_name, "supercategory": "object"})
    img_id, ann_id = 0, 0
    print(f"Memproses gambar dari: {img_folder}")
    for img_filename in sorted(os.listdir(img_folder)):
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        img_path = os.path.join(img_folder, img_filename)
        try:
            img = cv2.imread(img_path)
            height, width, _ = img.shape
        except Exception as e: print(f"Peringatan: Gagal membaca {img_path}. Error: {e}"); continue
        coco_format["images"].append({"id": img_id, "file_name": img_filename, "width": width, "height": height})
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(label_folder, label_filename)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                    abs_width, abs_height = bbox_width * width, bbox_height * height
                    x_min, y_min = (x_center * width) - (abs_width / 2), (y_center * height) - (abs_height / 2)
                    coco_format["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": int(class_id), "bbox": [x_min, y_min, abs_width, abs_height], "area": abs_width * abs_height, "iscrowd": 0, "segmentation": []})
                    ann_id += 1
        img_id += 1
    with open(output_json_path, 'w') as f: json.dump(coco_format, f, indent=4)
    print(f"✅ Konversi berhasil! File JSON disimpan di: {output_json_path}")

# ---

# LANGKAH D: JALANKAN KONVERSI
# ==============================================================================
print("\n🚀 [3/5] Memulai konversi dataset kustom Anda...")

# 🎯 ===================================================================================
# 🎯 UBAH DAFTAR NAMA KELAS DI BAWAH INI SESUAI DATASET ANDA
my_class_names = ["person", "box", "bag"]
# 🎯 ===================================================================================

# Path ke dataset ASLI Anda
base_train_folder = "/content/drive/MyDrive/DATASET/ORIGINAL DATASET/TRAIN"
yolo_train_img_folder = os.path.join(base_train_folder, "IMAGES")
yolo_train_label_folder = os.path.join(base_train_folder, "LABELS")

# Path untuk menyimpan dataset format COCO
coco_output_dir = os.path.join(project_dir, "COCO_DATASET")
coco_train_dir = os.path.join(coco_output_dir, "train2017")
coco_annotations_dir = os.path.join(coco_output_dir, "annotations")

# Buat direktori dan salin semua gambar
os.makedirs(coco_train_dir, exist_ok=True)
os.makedirs(coco_annotations_dir, exist_ok=True)
print(f"Menyalin gambar dari {yolo_train_img_folder} ke {coco_train_dir}...")
!cp -f "{yolo_train_img_folder}"/* "{coco_train_dir}/"

# Jalankan fungsi konversi
output_json_file = os.path.join(coco_annotations_dir, "instances_train2017.json")
convert_yolo_to_coco(yolo_train_img_folder, yolo_train_label_folder, output_json_file, my_class_names)

# ---

# LANGKAH E: SIAPKAN PERINTAH TRAINING
# ==============================================================================
print("\n🚀 [4/5] Menyiapkan perintah training final...")

final_training_command = f"""
!python main.py \\
  --batch_size 2 \\
  --fewshot_finetune \\
  --epochs 50 \\
  --lr_drop 40 \\
  --output_dir "outputs/custom_dataset_training" \\
  --coco_path "{coco_output_dir}" \\
  --resume "{model_path}" \\
  --num_workers 2
"""

print("\n\n✅ [5/5] SEMUA PERSIAPAN SELESAI!")
print("Salin perintah di bawah ini, tempel di SEL KODE BARU, lalu jalankan untuk memulai training.")
print("-------------------------------------------------------------------------------------------")
print(final_training_command)
print("-------------------------------------------------------------------------------------------")
