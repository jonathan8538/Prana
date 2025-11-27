import mediapipe as mp
import cv2
import argparse
import csv
import numpy as np
from pathlib import Path
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_pose_data(video_folder, class_name, output_file):
    """
    Ekstrak pose data dari semua video dalam folder
    
    Args:
        video_folder: Path ke folder berisi video (contoh: vid/raw/train/Agem_Kanan)
        class_name: Nama gerakan (contoh: 'Agem_Kanan')
        output_file: Nama file CSV output (contoh: 'tari_bali_training.csv')
    """
    
    # Buat folder data jika belum ada
    Path('data').mkdir(parents=True, exist_ok=True)
    
    # Dapatkan semua video dalam folder
    video_files = [f for f in os.listdir(video_folder) 
                   if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
    
    print(f"\n{'='*50}")
    print(f"Memproses gerakan: {class_name}")
    print(f"Jumlah video: {len(video_files)}")
    print(f"{'='*50}\n")
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Gagal membuka: {video_file}")
            continue
        
        print(f"📹 Memproses: {video_file}")
        frame_count = 0
        saved_count = 0
        
        with mp_pose.Pose(min_detection_confidence=0.5, 
                         min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Proses setiap 3 frame untuk mengurangi data redundan
                if frame_count % 3 != 0:
                    continue
                
                # Convert ke RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Deteksi pose
                results = pose.process(image_rgb)
                
                # Kembali ke BGR untuk display
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Ekstrak coordinates
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        body_landmarks = []
                        
                        # 12 key points (seperti di SuperShy)
                        keypoints = [
                            mp_pose.PoseLandmark.LEFT_SHOULDER,   # 11
                            mp_pose.PoseLandmark.RIGHT_SHOULDER,  # 12
                            mp_pose.PoseLandmark.LEFT_ELBOW,      # 13
                            mp_pose.PoseLandmark.RIGHT_ELBOW,     # 14
                            mp_pose.PoseLandmark.LEFT_WRIST,      # 15
                            mp_pose.PoseLandmark.RIGHT_WRIST,     # 16
                            mp_pose.PoseLandmark.LEFT_HIP,        # 23
                            mp_pose.PoseLandmark.RIGHT_HIP,       # 24
                            mp_pose.PoseLandmark.LEFT_KNEE,       # 25
                            mp_pose.PoseLandmark.RIGHT_KNEE,      # 26
                            mp_pose.PoseLandmark.LEFT_ANKLE,      # 27
                            mp_pose.PoseLandmark.RIGHT_ANKLE      # 28
                        ]
                        
                        for point in keypoints:
                            landmark = landmarks[point.value]
                            body_landmarks.extend([landmark.x, landmark.y])
                        
                        # Siapkan data untuk CSV
                        row = np.around(body_landmarks, decimals=9).tolist()
                        row.insert(0, class_name)
                        
                        # Simpan ke CSV
                        with open(f'data/{output_file}', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', 
                                                   quotechar='"', 
                                                   quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)
                        
                        saved_count += 1
                
                except Exception as e:
                    pass
                
                # Display (optional - bisa di-comment jika tidak perlu)
                cv2.imshow('Pose Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        print(f"   ✅ Tersimpan: {saved_count} poses dari {frame_count} frames\n")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-folder", required=True, 
                    help="Folder berisi video (contoh: vid/raw/train/Agem_Kanan)")
    ap.add_argument("--class-name", required=True, 
                    help="Nama gerakan (contoh: Agem_Kanan)")
    ap.add_argument("--output-file", type=str, 
                    default='tari_bali_training.csv',
                    help="Nama file CSV output")
    
    args = vars(ap.parse_args())
    
    extract_pose_data(
        args['video_folder'],
        args['class_name'],
        args['output_file']
    )