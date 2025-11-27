import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class TariBaliPractice:
    def __init__(self, model_name='tari_bali_model'):
        # Load model
        model_path = f'model/{model_name}.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load classes
        with open(f'model/{model_name}_classes.txt', 'r') as f:
            self.classes = f.read().split(',')
        
        print(f"✅ Model loaded: {model_name}")
        print(f"✅ Gerakan yang tersedia: {self.classes}")
        
        # Tracking
        self.current_pose = None
        self.confidence = 0.0
        self.target_pose = self.classes[0] if self.classes else None
        
        # Buffer untuk smoothing
        self.pose_buffer = deque(maxlen=5)
        
        # Counter
        self.correct_frames = 0
        self.total_frames = 0
        
    def extract_landmarks(self, results):
        """Ekstrak 12 pose landmarks"""
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        body_landmarks = []
        
        keypoints = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        for point in keypoints:
            landmark = landmarks[point.value]
            body_landmarks.extend([landmark.x, landmark.y])
        
        return body_landmarks
    
    def predict_pose(self, landmarks):
        """Prediksi pose dari landmarks"""
        if landmarks is None:
            return None, 0.0
        
        X = pd.DataFrame([landmarks])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities.max()
        
        return prediction, confidence
    
    def draw_ui(self, frame):
        """Gambar UI (mirip gambar 3)"""
        h, w = frame.shape[:2]
        
        # Header bar - warna kuning untuk tutorial
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 200, 255), -1)
        cv2.putText(frame, "LATIHAN TARI BALI", (w//2 - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(frame, "TUTORIAL", (w//2 - 80, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Target pose
        target_text = f"Level 1: {self.target_pose}"
        cv2.putText(frame, target_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Info box kiri atas
        cv2.rectangle(frame, (10, 140), (300, 300), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 140), (300, 300), (255, 255, 255), 2)
        
        # Current detection
        if self.current_pose:
            is_correct = self.current_pose == self.target_pose
            color = (0, 255, 0) if is_correct else (0, 165, 255)
            
            # Status text
            if is_correct and self.confidence > 0.7:
                status_text = "BENAR!"
                status_color = (0, 255, 0)
            elif is_correct:
                status_text = "HAMPIR BENAR"
                status_color = (0, 200, 200)
            else:
                status_text = "SALAH POSE"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(frame, f"Terdeteksi:", (20, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{self.current_pose}", (20, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, f"Conf: {self.confidence*100:.0f}%", (20, 285),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Mencari pose...", (20, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Accuracy box di kiri bawah
        accuracy = (self.correct_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        cv2.rectangle(frame, (10, h-120), (300, h-20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h-120), (300, h-20), (255, 255, 255), 2)
        cv2.putText(frame, "AKURASI:", (20, h-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{accuracy:.0f}%", (20, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Progress bar
        bar_width = int(270 * accuracy / 100)
        cv2.rectangle(frame, (20, h-35), (280, h-25), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h-35), (20 + bar_width, h-25), (0, 255, 0), -1)
        
        # Footer - Controls
        cv2.rectangle(frame, (0, h-60), (w, h), (40, 40, 40), -1)
        controls = "R: Restart | Q: Keluar | SPACE: Lanjut | 1-3: Pilih Gerakan"
        cv2.putText(frame, controls, (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Jalankan aplikasi latihan"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("🎭 SISTEM LATIHAN TARI BALI")
        print("="*60)
        print("\nKontrol:")
        print("  1, 2, 3 - Pilih gerakan target")
        print("  SPACE   - Ganti ke gerakan berikutnya")
        print("  R       - Reset counter")
        print("  Q       - Keluar")
        print("\nGerakan tersedia:")
        for i, pose in enumerate(self.classes, 1):
            print(f"  {i}. {pose}")
        print("\n" + "="*60 + "\n")
        
        with mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Detect pose
                results = pose.process(image_rgb)
                
                # Back to BGR
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Extract dan predict
                    landmarks = self.extract_landmarks(results)
                    if landmarks:
                        prediction, confidence = self.predict_pose(landmarks)
                        
                        # Smoothing dengan buffer
                        self.pose_buffer.append((prediction, confidence))
                        
                        # Ambil pose paling sering muncul di buffer
                        if len(self.pose_buffer) >= 3:
                            poses = [p[0] for p in self.pose_buffer]
                            confs = [p[1] for p in self.pose_buffer]
                            
                            # Most common pose
                            self.current_pose = max(set(poses), key=poses.count)
                            self.confidence = np.mean([c for p, c in self.pose_buffer if p == self.current_pose])
                        
                        # Update statistics
                        self.total_frames += 1
                        if self.current_pose == self.target_pose and self.confidence > 0.6:
                            self.correct_frames += 1
                
                # Draw UI
                image = self.draw_ui(image)
                
                # Show
                cv2.imshow('Latihan Tari Bali', image)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset
                    self.correct_frames = 0
                    self.total_frames = 0
                    self.pose_buffer.clear()
                    print("🔄 Counter di-reset")
                elif key == ord(' '):
                    # Next pose
                    current_idx = self.classes.index(self.target_pose)
                    next_idx = (current_idx + 1) % len(self.classes)
                    self.target_pose = self.classes[next_idx]
                    print(f"➡️  Target baru: {self.target_pose}")
                elif key == ord('1') and len(self.classes) > 0:
                    self.target_pose = self.classes[0]
                    print(f"🎯 Target: {self.target_pose}")
                elif key == ord('2') and len(self.classes) > 1:
                    self.target_pose = self.classes[1]
                    print(f"🎯 Target: {self.target_pose}")
                elif key == ord('3') and len(self.classes) > 2:
                    self.target_pose = self.classes[2]
                    print(f"🎯 Target: {self.target_pose}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "="*60)
        print("📊 STATISTIK LATIHAN")
        print("="*60)
        accuracy = (self.correct_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        print(f"Total frames: {self.total_frames}")
        print(f"Correct frames: {self.correct_frames}")
        print(f"Akurasi: {accuracy:.1f}%")
        print("="*60)


if __name__ == '__main__':
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default='tari_bali_model',
                    help="Nama model (tanpa .pkl)")
    args = vars(ap.parse_args())
    
    try:
        app = TariBaliPractice(model_name=args['model_name'])
        app.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Pastikan model sudah di-training terlebih dahulu!")
    except Exception as e:
        print(f"\n❌ Error: {e}")