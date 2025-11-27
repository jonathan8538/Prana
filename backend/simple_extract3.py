"""
Simple Extract - Ekstrak 3 gerakan: Agem_Kanan, Ngeed, Ngegol
"""

import subprocess
import os

# ⚠️ EDIT INI dengan path LENGKAP kamu
DATASET_PATH = "vid/raw/Video Dataset of Woman Basic Balinese Dance Movement for Action Recognition"

# 3 gerakan yang mau dilatih
movements = ['Agem_Kanan', 'Ngeed', 'Ngegol']

print("🎭 Ekstrak 3 Gerakan Tari Bali")
print("="*60)

for movement in movements:
    # Path ke folder video
    folder_path = os.path.join(DATASET_PATH, "train", movement)
    
    # Cek apakah folder ada
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ada: {folder_path}")
        print(f"   Lewati gerakan: {movement}\n")
        continue
    
    # Hitung jumlah video
    videos = [f for f in os.listdir(folder_path) 
              if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
    
    print(f"\n📹 Gerakan: {movement}")
    print(f"   Folder: {folder_path}")
    print(f"   Videos: {len(videos)} file")
    print(f"   Memproses...")
    
    # Run ekstraksi
    cmd = [
        'python', '01_training_data.py',
        '--video-folder', folder_path,
        '--class-name', movement,
        '--output-file', 'tari_bali_training.csv'
    ]
    
    subprocess.run(cmd)
    print(f"   ✅ Selesai!\n")

print("="*60)
print("✅ SELESAI!")
print("\nNext step:")
print("   python 02_pose_model_training.py")