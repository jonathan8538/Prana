import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os
import json
import random

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

GAME_CONFIG = {
    'target_accuracy': 70,          # Turunkan dari 75 -> 70
    'min_frames': 12,               # Turunkan dari 15 -> 12
    'perfect_threshold': 0.90,      # Turunkan dari 0.95 -> 0.90
    'good_threshold': 0.80,         # Turunkan dari 0.90 -> 0.80
    'almost_threshold': 0.70,       # Turunkan dari 0.85 -> 0.70
    'min_confidence': 0.60,         # Turunkan dari 0.80 -> 0.60 (SANGAT RENDAH!)
    'cooldown_time': 0.7,           # Turunkan dari 0.8 -> 0.7
    'difficulty_factor': 0.0,       # Matikan difficulty! (dari 0.3 -> 0.0)
    'difficulty_reduction': (0.95, 0.99)  # Almost no reduction
}

POSE_LEVELS = [
    {'level': 1, 'pose': 'Jump-1Leg', 'model': 'sajojo_model1', 'difficulty': '⭐ Mudah', 
     'indonesian': 'Loncat Kaki Satu', 'time_range': (15, 23)},
    {'level': 2, 'pose': 'Arms-UpDown', 'model': 'sajojo_model2', 'difficulty': '⭐ Mudah', 
     'indonesian': 'Tangan Atas-Bawah', 'time_range': (24, 29)},
    {'level': 3, 'pose': 'Arms-LeftRight', 'model': 'sajojo_model3', 'difficulty': '⭐⭐ Sedang', 
     'indonesian': 'Tangan Kanan-Kiri', 'time_range': (29, 34)},
    {'level': 4, 'pose': 'Arms-BothUp', 'model': 'sajojo_model4', 'difficulty': '⭐⭐ Sedang', 
     'indonesian': 'Kedua Tangan Atas', 'time_range': (35, 40)},
    {'level': 5, 'pose': 'Water-Splash', 'model': 'sajojo_model5', 'difficulty': '⭐⭐⭐ Sulit', 
     'indonesian': 'Percik Air', 'time_range': (57, 72)},
    {'level': 6, 'pose': 'Fly', 'model': 'sajojo_model6', 'difficulty': '⭐⭐⭐ Sulit', 
     'indonesian': 'Terbang', 'time_range': (77, 80)},
]

class SajojoGame:
    def __init__(self):
        self.current_level = 0
        self.unlocked_levels = [0]
        self.scores = {}
        self.load_progress()
    
    def load_progress(self):
        if os.path.exists('sajojo_progress.json'):
            try:
                with open('sajojo_progress.json', 'r') as f:
                    data = json.load(f)
                    self.current_level = data.get('current_level', 0)
                    self.unlocked_levels = data.get('unlocked_levels', [0])
                    self.scores = data.get('scores', {})
            except:
                pass
    
    def save_progress(self):
        with open('sajojo_progress.json', 'w') as f:
            json.dump({
                'current_level': self.current_level,
                'unlocked_levels': self.unlocked_levels,
                'scores': self.scores
            }, f)
    
    def unlock_next(self):
        next_level = self.current_level + 1
        if next_level < len(POSE_LEVELS) and next_level not in self.unlocked_levels:
            self.unlocked_levels.append(next_level)
            self.save_progress()
            return True
        return False

def extract_landmarks(landmarks):
    try:
        body = []
        indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in indices:
            body.append(landmarks[idx].x)
            body.append(landmarks[idx].y)
        return np.around(body, decimals=9).tolist()
    except:
        return None

def predict(landmarks, model):
    try:
        features = extract_landmarks(landmarks)
        if not features:
            print("DEBUG: extract_landmarks returned None")
            return None, 0
        X = pd.DataFrame([features])
        pose = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        conf = np.max(proba)
        
        # DEBUG: Print setiap 30 frame
        if hasattr(predict, 'frame_count'):
            predict.frame_count += 1
        else:
            predict.frame_count = 0
            
        if predict.frame_count % 30 == 0:
            print(f"DEBUG: Pose={pose}, Conf={conf:.3f}, Proba={proba}")
        
        return pose, conf
    except Exception as e:
        print(f"ERROR in predict: {e}")
        return None, 0

def apply_difficulty(conf):
    if random.random() < GAME_CONFIG['difficulty_factor']:
        reduction_factor = random.uniform(
            GAME_CONFIG['difficulty_reduction'][0], 
            GAME_CONFIG['difficulty_reduction'][1]
        )
        conf = conf * reduction_factor
    return conf

def draw_ui_split(combined_frame, state, level_info, tutorial_time):
    """Draw UI on the combined split-screen frame"""
    h, w = combined_frame.shape[:2]
    
    # Header bar
    cv2.rectangle(combined_frame, (0, 0), (w, 80), (40, 40, 40), -1)
    
    # Left side - Your Camera label
    cv2.putText(combined_frame, "KAMU", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Right side - Tutorial label
    cv2.putText(combined_frame, "TUTORIAL", 
                (w//2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Center - Level info
    level_text = f"Level {level_info['level']}: {level_info['indonesian']}"
    text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(combined_frame, level_text, 
                (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Score in top right
    cv2.putText(combined_frame, f"Skor: {state['score']}", 
                (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
    
    # Streak
    if state['streak'] > 0:
        color = (0, 255, 0) if state['streak'] >= 5 else (255, 255, 255)
        cv2.putText(combined_frame, f"Kombo: {state['streak']}x", 
                    (w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Progress bar
    progress = min(state['good_frames'] / GAME_CONFIG['min_frames'], 1.0)
    bar_y = 85
    cv2.rectangle(combined_frame, (10, bar_y), (w-10, bar_y + 15), (80, 80, 80), -1)
    bar_w = int((w - 20) * progress)
    color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
    cv2.rectangle(combined_frame, (10, bar_y), (10 + bar_w, bar_y + 15), color, -1)
    cv2.putText(combined_frame, f"{state['good_frames']}/{GAME_CONFIG['min_frames']}", 
                (w//2 - 30, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Tutorial time indicator
    cv2.putText(combined_frame, f"Tutorial: {tutorial_time:.1f}s", 
                (w//2 + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Feedback overlay (center)
    if state['feedback'] and state['feedback_alpha'] > 0:
        alpha = state['feedback_alpha']
        overlay = combined_frame.copy()
        cv2.rectangle(overlay, (w//2 - 200, h//2 - 60), (w//2 + 200, h//2 + 60), 
                     state['feedback_color'], -1)
        cv2.addWeighted(overlay, alpha, combined_frame, 1 - alpha, 0, combined_frame)
        cv2.putText(combined_frame, state['feedback'], 
                    (w//2 - 150, h//2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    
    # Bottom status bar
    acc = (state['correct'] / state['total'] * 100) if state['total'] > 0 else 0
    can_pass = acc >= GAME_CONFIG['target_accuracy'] and state['good_frames'] >= GAME_CONFIG['min_frames']
    
    if can_pass:
        bottom_color = (0, 255, 0)
        text = "LULUS! Tekan SPACE untuk lanjut"
    else:
        bottom_color = (60, 60, 100)
        text = f"Akurasi: {acc:.1f}% | Perlu: {GAME_CONFIG['target_accuracy']}%"
    
    cv2.rectangle(combined_frame, (0, h-60), (w, h), bottom_color, -1)
    cv2.putText(combined_frame, text, 
                (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined_frame, "R: Ulangi | Q: Keluar | SPACE: Lanjut", 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    game = SajojoGame()
    
    print("\n" + "="*60)
    print("TARI SAJOJO - Split Screen Mode")
    print("="*60)
    print("\n🎬 FITUR BARU: Split Screen dengan Tutorial!")
    print("  - Kiri: Kamera Anda")
    print("  - Kanan: Video Tutorial")
    print("  - Ikuti gerakan tutorial secara real-time!")
    
    print("\nLevel Terbuka:")
    for i in game.unlocked_levels:
        level = POSE_LEVELS[i]
        best = game.scores.get(f"level_{i}", 0)
        print(f"  Level {level['level']}: {level['indonesian']} {level['difficulty']} - Terbaik: {best} poin")
    
    input("\nTekan ENTER untuk mulai...")
    
    level_info = POSE_LEVELS[game.current_level]
    model_path = f"model/{level_info['model']}.pkl"
    
    print(f"\nMemuat model {level_info['model']}...")
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    print("Model siap")
    
    # Load video tutorial
    tutorial_video_path = r"C:\Users\yabes\OneDrive\Documents\DanceAI\DanceVision-AI-Driven-Dance-Proficiency-Assessment\vid\raw\sajojo_tutorial.webm"
    
    if not os.path.exists(tutorial_video_path):
        print(f"⚠️ Video tutorial tidak ditemukan: {tutorial_video_path}")
        print("Game akan berjalan tanpa tutorial video.")
        tutorial_cap = None
    else:
        tutorial_cap = cv2.VideoCapture(tutorial_video_path)
        if not tutorial_cap.isOpened():
            print("⚠️ Tidak bisa membuka video tutorial")
            tutorial_cap = None
        else:
            # Set video ke waktu yang tepat untuk level ini
            start_time = level_info['time_range'][0]
            fps = tutorial_cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"Tutorial video siap (mulai dari detik {start_time})")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam tidak bisa dibuka")
        if tutorial_cap:
            tutorial_cap.release()
        return
    print("Webcam siap\n")
    
    state = {
        'score': 0, 'streak': 0, 'correct': 0, 'total': 0,
        'good_frames': 0, 'feedback': '', 
        'feedback_color': (255, 255, 255), 'feedback_alpha': 0
    }
    
    last_good_time = 0
    frame_count = 0
    tutorial_start_time = time.time()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process webcam frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            user_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks on user frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    user_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )
                
                # Predict pose
                user_pose, conf = predict(results.pose_landmarks.landmark, model)
                
                # DEBUG INFO di layar - TAMPILKAN SELALU
                debug_y = 90
                cv2.putText(user_frame, f"Deteksi: {user_pose if user_pose else 'None'}", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(user_frame, f"Conf: {conf:.3f}", 
                           (10, debug_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(user_frame, f"Expected: {level_info['pose']}", 
                           (10, debug_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if user_pose:
                    user_main = user_pose.split(' ')[0]
                    expected = level_info['pose']
                    
                    # Apply difficulty modifier
                    original_conf = conf
                    conf = apply_difficulty(conf)
                    
                    # Show if difficulty was applied
                    if conf != original_conf:
                        cv2.putText(user_frame, f"(Difficulty: {conf:.3f})", 
                                   (10, debug_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # DIPERBAIKI: Cooldown check pertama, baru evaluasi
                    current_time = time.time()
                    if current_time - last_good_time >= GAME_CONFIG['cooldown_time']:
                        
                        # Cek apakah confidence cukup tinggi
                        if conf >= GAME_CONFIG['min_confidence']:
                            state['total'] += 1  # HITUNG SEMUA DETEKSI dengan confidence tinggi
                            
                            if user_main == expected:
                                state['correct'] += 1
                                last_good_time = current_time
                                
                                if conf >= GAME_CONFIG['perfect_threshold']:
                                    state['feedback'] = "SEMPURNA!"
                                    state['feedback_color'] = (0, 255, 0)
                                    points = 10
                                    state['streak'] += 1
                                    state['good_frames'] += 1
                                elif conf >= GAME_CONFIG['good_threshold']:
                                    state['feedback'] = "BAGUS!"
                                    state['feedback_color'] = (0, 255, 255)
                                    points = 5
                                    state['streak'] += 1
                                    state['good_frames'] += 1
                                elif conf >= GAME_CONFIG['almost_threshold']:
                                    state['feedback'] = "HAMPIR"
                                    state['feedback_color'] = (255, 200, 100)
                                    points = 2
                                    state['good_frames'] += 1
                                else:
                                    state['feedback'] = "KURANG TEPAT"
                                    state['feedback_color'] = (255, 150, 0)
                                    points = 0
                                
                                if points > 0:
                                    if state['streak'] >= 10:
                                        points *= 3
                                    elif state['streak'] >= 5:
                                        points *= 2
                                    
                                    state['score'] += points
                                    state['feedback_alpha'] = 0.8
                            else:
                                # POSE SALAH (bukan expected)
                                state['feedback'] = "SALAH POSE"
                                state['feedback_color'] = (0, 0, 255)
                                state['feedback_alpha'] = 0.8
                                state['streak'] = 0
                                last_good_time = current_time
                        else:
                            # Confidence terlalu rendah - TAMPILKAN FEEDBACK
                            cv2.putText(user_frame, "Conf terlalu rendah!", 
                                       (10, debug_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # user_pose is None
                    cv2.putText(user_frame, "ERROR: Tidak bisa predict", 
                               (10, debug_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Get tutorial frame
            if tutorial_cap and tutorial_cap.isOpened():
                ret_tut, tutorial_frame = tutorial_cap.read()
                
                # Loop tutorial video dalam range yang ditentukan
                if not ret_tut:
                    # Reset ke awal range
                    start_frame = int(level_info['time_range'][0] * fps)
                    tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret_tut, tutorial_frame = tutorial_cap.read()
                
                # Check if we've passed the end time
                current_tutorial_frame = tutorial_cap.get(cv2.CAP_PROP_POS_FRAMES)
                current_tutorial_time = current_tutorial_frame / fps
                end_time = level_info['time_range'][1]
                
                if current_tutorial_time > end_time:
                    # Loop back to start
                    start_frame = int(level_info['time_range'][0] * fps)
                    tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret_tut, tutorial_frame = tutorial_cap.read()
                    current_tutorial_time = level_info['time_range'][0]
                
                if ret_tut:
                    tutorial_frame = cv2.resize(tutorial_frame, (640, 480))
                else:
                    # Create black frame if tutorial not available
                    tutorial_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(tutorial_frame, "Tutorial tidak tersedia", 
                               (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                tutorial_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(tutorial_frame, "Tutorial tidak tersedia", 
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                current_tutorial_time = 0
            
            # Combine frames side by side
            combined = np.hstack([user_frame, tutorial_frame])
            
            # Fade out feedback
            if state['feedback_alpha'] > 0:
                state['feedback_alpha'] -= 0.05
            
            # Draw UI on combined frame
            draw_ui_split(combined, state, level_info, current_tutorial_time)
            
            cv2.imshow('Tari Sajojo - Split Screen (Tekan Q untuk Keluar)', combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            acc = (state['correct'] / state['total'] * 100) if state['total'] > 0 else 0
            can_pass = acc >= GAME_CONFIG['target_accuracy'] and state['good_frames'] >= GAME_CONFIG['min_frames']
            
            if key == ord(' ') and can_pass:
                level_key = f"level_{game.current_level}"
                if level_key not in game.scores or state['score'] > game.scores[level_key]:
                    game.scores[level_key] = state['score']
                
                print(f"\nLevel {level_info['level']} LULUS!")
                print(f"Skor: {state['score']} | Akurasi: {acc:.1f}%")
                
                if game.unlock_next():
                    print(f"Level {game.current_level + 2} terbuka!")
                    game.current_level += 1
                    game.save_progress()
                    if game.current_level < len(POSE_LEVELS):
                        break
                    else:
                        print("\nSEMUA LEVEL SELESAI!")
                        break
                else:
                    print("\nKamu juara!")
                    break
            
            elif key == ord('r'):
                state = {
                    'score': 0, 'streak': 0, 'correct': 0, 'total': 0,
                    'good_frames': 0, 'feedback': '',
                    'feedback_color': (255, 255, 255), 'feedback_alpha': 0
                }
                last_good_time = 0
                
                # Reset tutorial video
                if tutorial_cap:
                    start_frame = int(level_info['time_range'][0] * fps)
                    tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            elif key == ord('q'):
                break
            
            frame_count += 1
    
    cap.release()
    if tutorial_cap:
        tutorial_cap.release()
    cv2.destroyAllWindows()
    print("\nTerima kasih!")

if __name__ == '__main__':
    main()