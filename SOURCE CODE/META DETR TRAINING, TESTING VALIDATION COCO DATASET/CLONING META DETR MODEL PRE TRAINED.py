# ==============================================================================
# SKRIP FINAL UNTUK DIJALANKAN SETELAH FACTORY RESET RUNTIME
# ==============================================================================

import os
from google.colab import drive

# --- Langkah 1: Hubungkan Google Drive ---
print("🔌 Menghubungkan Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("✔️ Google Drive berhasil terhubung.")

# --- Langkah 2: Masukkan Token GitHub Anda ---
# ❗️ GANTI STRING DI BAWAH INI DENGAN TOKEN GITHUB ANDA.
GITHUB_TOKEN = "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"

# --- Langkah 3: Menentukan Path Proyek ---
project_path = "/content/drive/MyDrive/Proyek_MetaDETR"
print(f"\n📁 Path proyek Anda: {project_path}")

# --- Langkah 4: Membersihkan dan Menyiapkan Folder Proyek ---
print("\n🧹 Membersihkan folder tujuan untuk memulai dari awal...")
!rm -rf "{project_path}"
!mkdir -p "{project_path}"
print("✔️ Folder tujuan sudah bersih dan siap.")

# --- Langkah 5: Clone Repositori dengan URL yang BENAR ---
print(f"\n🚀 Memulai proses download dari GitHub...")
!git clone https://{GITHUB_TOKEN}@github.com/facebookresearch/detr.git "{project_path}"

# --- Langkah 6: Verifikasi dan Pindah Direktori ---
print("\n🔍 Memeriksa hasil download...")
if not os.listdir(project_path):
    print("\n❌ GAGAL: Download sepertinya tidak berhasil. Folder masih kosong.")
    print("   Pastikan token GitHub yang Anda masukkan sudah benar.")
    raise SystemExit("Proses dihentikan.")

print("✔️ Download berhasil! Semua file proyek sudah ada di folder Anda.")
os.chdir(project_path)
print(f"✔️ Berhasil pindah ke direktori: {os.getcwd()}")

# --- Langkah 7: Instalasi Library yang Dibutuhkan ---
print("\n⚙️ Menginstal semua library yang dibutuhkan...")
!pip install -r requirements.txt
print("✔️ Instalasi library selesai.")

# --- Langkah 8: Login ke Hugging Face ---
from huggingface_hub import notebook_login
print("\n🔑 Silakan login ke Hugging Face.")
print("   Siapkan Access Token Anda dari: https://huggingface.co/settings/tokens")
notebook_login()

print("\n🎉 SELAMAT! SEMUA PROSES SETUP SELESAI DENGAN SUKSES.")
