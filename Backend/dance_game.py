import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class TariBaliGame:
    def __init__(self, model_name='tari_bali_xgb_model'):
        # >>> Pakai artefak bersih dari Colab (tanpa wrapper custom)
        self.model = joblib.load('model/xgb_pipeline.pkl')            # Pipeline(StandardScaler, XGBClassifier)
        self.classes = np.load('model/xgb_classes.npy', allow_pickle=True).astype(str).tolist()
     # ['Agem_Kanan','Ngeed','Ngegol']

        
        print(f"✅ Model loaded!")
        print(f"✅ Classes: {self.classes}")
        
        # Game levels
        self.levels = [
            {'name': 'Level 1: Agem Kanan', 'target': 'Agem_Kanan', 'required': 70},
            {'name': 'Level 2: Ngeed', 'target': 'Ngeed', 'required': 75},
            {'name': 'Level 3: Ngegol', 'target': 'Ngegol', 'required': 80}
        ]
        
        self.current_level = 0
        self.level_completed = [False, False, False]
        
        # Tracking
        self.current_pose = None
        self.confidence = 0.0
        self.pose_buffer = deque(maxlen=5)
        
        # Scoring
        self.correct_frames = 0
        self.total_frames = 0
        
        # Combo
        self.combo = 0
        self.max_combo = 0
        self.last_was_correct = False
        
        # ✅ Tutorial video paths
        self.dataset_path = r"C:\Users\yabes\OneDrive\Documents\DanceAI\DanceVision-AI-Driven-Dance-Proficiency-Assessment\vid\raw\Video Dataset of Woman Basic Balinese Dance Movement for Action Recognition\train"
        self.tutorial_cap = None
        self.load_tutorial_video()
        
    def load_tutorial_video(self):
        """Load video tutorial dari dataset"""
        if self.tutorial_cap is not None:
            self.tutorial_cap.release()
        
        level = self.levels[self.current_level]
        target_folder = os.path.join(self.dataset_path, level['target'])
        
        if not os.path.exists(target_folder):
            print(f"⚠️  Folder tidak ditemukan: {target_folder}")
            self.tutorial_cap = None
            return
        
        # Ambil video pertama dari folder
        video_files = [f for f in os.listdir(target_folder) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
        
        if not video_files:
            print(f"⚠️  Tidak ada video di: {target_folder}")
            self.tutorial_cap = None
            return
        
        # Load video pertama
        video_path = os.path.join(target_folder, video_files[0])
        self.tutorial_cap = cv2.VideoCapture(video_path)
        
        if self.tutorial_cap.isOpened():
            print(f"✅ Tutorial video loaded: {video_files[0]}")
        else:
            print(f"❌ Gagal load video: {video_path}")
            self.tutorial_cap = None
    
    def get_tutorial_frame(self):
        """Get frame dari tutorial video (loop terus)"""
        if self.tutorial_cap is None or not self.tutorial_cap.isOpened():
            # Return placeholder kalau video tidak ada
            return self.create_tutorial_placeholder(
                self.levels[self.current_level]['target']
            )
        
        ret, frame = self.tutorial_cap.read()
        
        if not ret:
            # Loop video ke awal
            self.tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.tutorial_cap.read()
        
        if ret:
            # Resize
            frame = cv2.resize(frame, (400, 530))
            
            # Tambah label di atas video
            label_bar = np.zeros((70, 400, 3), dtype=np.uint8)
            label_bar[:] = (0, 200, 255)
            cv2.putText(label_bar, "VIDEO TUTORIAL", (80, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            frame_with_label = np.vstack([label_bar, frame])
            
            return frame_with_label
        
        return self.create_tutorial_placeholder(
            self.levels[self.current_level]['target']
        )
    
    def create_tutorial_placeholder(self, target_name):
        """Buat placeholder kalau video tidak ada"""
        tutorial = np.ones((600, 400, 3), dtype=np.uint8) * 220
        
        # Header
        cv2.rectangle(tutorial, (0, 0), (400, 70), (0, 200, 255), -1)
        cv2.putText(tutorial, "VIDEO TUTORIAL", (70, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Target name
        cv2.putText(tutorial, target_name, (70, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 200), 3)
        
        # Note
        cv2.putText(tutorial, "Video tidak ditemukan", (80, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        cv2.putText(tutorial, "Cek path dataset", (100, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return tutorial
        
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
        if landmarks is None:
            return None, 0.0

        X = pd.DataFrame([landmarks])
        y_int = self.model.predict(X)[0]           # 0/1/2
        proba = self.model.predict_proba(X)[0]     # urutan kelas = self.classes
        label = self.classes[y_int]                # 'Agem_Kanan' / 'Ngeed' / 'Ngegol'
        conf = proba[y_int]
        return label, conf

    
    def update_score(self, is_correct, confidence):
        """Update score dengan combo system"""
        self.total_frames += 1
        
        # HANYA COUNT CORRECT jika confidence > 75%
        if is_correct and confidence > 0.75:
            self.correct_frames += 1
            
            # Combo naik HANYA jika frame sebelumnya juga correct
            if self.last_was_correct:
                self.combo += 1
            else:
                self.combo = 1
            
            self.max_combo = max(self.max_combo, self.combo)
            self.last_was_correct = True
        else:
            self.combo = 0
            self.last_was_correct = False
    
    def get_accuracy(self):
        """Hitung akurasi"""
        if self.total_frames == 0:
            return 0
        return (self.correct_frames / self.total_frames) * 100
    
    def check_level_complete(self):
        """Cek level selesai"""
        accuracy = self.get_accuracy()
        required = self.levels[self.current_level]['required']
        return accuracy >= required and self.total_frames >= 100
    
    def next_level(self):
        """Pindah ke level berikutnya"""
        self.level_completed[self.current_level] = True
        
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.reset_score()
            self.load_tutorial_video()  # Load video baru
            return True
        return False
    
    def reset_score(self):
        """Reset score untuk level baru"""
        self.correct_frames = 0
        self.total_frames = 0
        self.combo = 0
        self.max_combo = 0
        self.last_was_correct = False
        self.pose_buffer.clear()
    
    def draw_split_screen(self, webcam_frame):
        """Buat split screen: kiri webcam, kanan video tutorial LIVE"""
        # Resize webcam
        webcam_resized = cv2.resize(webcam_frame, (800, 600))
        
        # Get tutorial frame (LIVE VIDEO!)
        tutorial_frame = self.get_tutorial_frame()
        tutorial_resized = cv2.resize(tutorial_frame, (400, 600))
        
        # Combine side by side
        combined = np.hstack([webcam_resized, tutorial_resized])
        
        return combined
    
    def draw_ui(self, frame):
        """Draw game UI"""
        # Buat split screen dulu
        split_frame = self.draw_split_screen(frame)
        
        h, w = split_frame.shape[:2]
        
        # Get current level info
        level = self.levels[self.current_level]
        target_pose = level['target']
        required_score = level['required']
        accuracy = self.get_accuracy()
        
        # Header
        header_color = (0, 200, 255)
        cv2.rectangle(split_frame, (0, 0), (w, 100), header_color, -1)
        
        # Level name
        cv2.putText(split_frame, level['name'], (w//2 - 250, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(split_frame, "TUTORIAL", (w//2 - 100, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Level indicators
        for i, completed in enumerate(self.level_completed):
            x = w - 180 + (i * 50)
            y = 40
            if completed:
                color = (0, 255, 0)
            elif i == self.current_level:
                color = (255, 200, 0)
            else:
                color = (100, 100, 100)
            
            cv2.circle(split_frame, (x, y), 22, color, -1)
            cv2.circle(split_frame, (x, y), 22, (255, 255, 255), 3)
            cv2.putText(split_frame, str(i+1), (x-11, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Left panel - Detection
        panel_x = 20
        panel_y = 120
        cv2.rectangle(split_frame, (panel_x, panel_y), (panel_x + 300, panel_y + 180), 
                     (0, 0, 0), -1)
        cv2.rectangle(split_frame, (panel_x, panel_y), (panel_x + 300, panel_y + 180), 
                     (255, 255, 255), 2)
        
        # Target
        cv2.putText(split_frame, "TARGET:", (panel_x + 15, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(split_frame, target_pose, (panel_x + 15, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Status
        if self.current_pose:
            is_correct = self.current_pose == target_pose
            
            if is_correct and self.confidence > 0.75:
                status = "SEMPURNA!"
                status_color = (0, 255, 0)
            elif is_correct and self.confidence > 0.65:
                status = "BENAR"
                status_color = (0, 255, 150)
            else:
                status = "SALAH"
                status_color = (0, 0, 255)
            
            cv2.putText(split_frame, status, (panel_x + 15, panel_y + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.putText(split_frame, f"{self.current_pose} ({self.confidence*100:.0f}%)", 
                       (panel_x + 15, panel_y + 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Score panel (bottom)
        score_y = h - 120
        cv2.rectangle(split_frame, (0, score_y), (w, h), (30, 30, 30), -1)
        
        # Accuracy
        acc_color = (0, 255, 0) if accuracy >= required_score else (0, 165, 255)
        cv2.putText(split_frame, f"AKURASI: {accuracy:.0f}% / {required_score}%", 
                   (30, score_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, acc_color, 2)
        
        # Progress bar
        bar_w = 400
        bar_x = 30
        bar_y = score_y + 55
        
        cv2.rectangle(split_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 25), 
                     (80, 80, 80), -1)
        filled_w = int(bar_w * accuracy / 100)
        cv2.rectangle(split_frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + 25), 
                     acc_color, -1)
        
        # Combo
        if self.combo > 3:
            combo_color = (0, 255, 255) if self.combo < 15 else (0, 255, 0)
            cv2.putText(split_frame, f"COMBO: {self.combo}x", 
                       (500, score_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, combo_color, 3)
        
        # Score
        cv2.putText(split_frame, f"Score: {self.correct_frames}/{self.total_frames}", 
                   (750, score_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Level complete
        if self.check_level_complete():
            overlay = split_frame.copy()
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, h//2 + 50), 
                         (0, 255, 0), -1)
            split_frame = cv2.addWeighted(split_frame, 0.6, overlay, 0.4, 0)
            
            cv2.putText(split_frame, "LEVEL COMPLETE!", 
                       (w//2 - 200, h//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            cv2.putText(split_frame, "Tekan SPACE untuk lanjut", 
                       (w//2 - 190, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Footer
        footer = "R: Restart | Q: Quit | SPACE: Next Level | 1-3: Jump Level"
        cv2.putText(split_frame, footer, (30, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return split_frame
    
    def run(self):
        """Jalankan game"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("🎮 TARI BALI - GAME MODE WITH VIDEO TUTORIAL")
        print("="*70)
        print("\n✨ Features:")
        print("  ✅ Live Webcam (Kiri)")
        print("  ✅ Video Tutorial (Kanan)")
        print("  ✅ Real-time Pose Detection")
        print("  ✅ Combo System (need confidence > 75%)")
        print("\nLevels:")
        for i, level in enumerate(self.levels, 1):
            print(f"  {i}. {level['name']} - Target: {level['required']}%")
        
        print("\nControls:")
        print("  SPACE - Next level")
        print("  R     - Restart")
        print("  1-3   - Jump level")
        print("  Q     - Quit")
        print("="*70 + "\n")
        
        self.reset_score()
        
        with mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Detect pose
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
                    
                    # Predict
                    landmarks = self.extract_landmarks(results)
                    if landmarks:
                        prediction, confidence = self.predict_pose(landmarks)
                        self.pose_buffer.append((prediction, confidence))
                        
                        if len(self.pose_buffer) >= 3:
                            poses = [p[0] for p in self.pose_buffer]
                            self.current_pose = max(set(poses), key=poses.count)
                            self.confidence = np.mean([c for p, c in self.pose_buffer 
                                                      if p == self.current_pose])
                        
                        # Update score
                        target = self.levels[self.current_level]['target']
                        is_correct = self.current_pose == target
                        self.update_score(is_correct, self.confidence)
                
                # Draw UI
                image = self.draw_ui(image)
                
                cv2.imshow('Tari Bali - Game Mode', image)
                
                # Controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_score()
                    print(f"🔄 Level {self.current_level + 1} restarted")
                elif key == ord(' '):
                    if self.check_level_complete():
                        if self.next_level():
                            print(f"🎉 Level {self.current_level} unlocked!")
                        else:
                            print("🏆 ALL LEVELS COMPLETED!")
                            break
                elif key == ord('1'):
                    self.current_level = 0
                    self.reset_score()
                    self.load_tutorial_video()
                    print("📍 Jump to Level 1")
                elif key == ord('2'):
                    self.current_level = 1
                    self.reset_score()
                    self.load_tutorial_video()
                    print("📍 Jump to Level 2")
                elif key == ord('3'):
                    self.current_level = 2
                    self.reset_score()
                    self.load_tutorial_video()
                    print("📍 Jump to Level 3")
        
        cap.release()
        
        # Release tutorial video
        if self.tutorial_cap is not None:
            self.tutorial_cap.release()
        
        cv2.destroyAllWindows()
        
        # Stats
        print("\n" + "="*70)
        print("📊 GAME SUMMARY")
        print("="*70)
        for i, (level, completed) in enumerate(zip(self.levels, self.level_completed), 1):
            status = "✅" if completed else "❌"
            print(f"{status} Level {i}: {level['name']}")
        print("="*70)


if __name__ == '__main__':
    try:
        game = TariBaliGame()
        game.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()