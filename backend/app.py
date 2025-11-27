from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import base64
import threading
import time
import os
from collections import deque

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class TariBaliDetector:
    def __init__(self):
        # Load model
        self.model = joblib.load('model/xgb_pipeline.pkl')
        self.classes = np.load('model/xgb_classes.npy', allow_pickle=True).astype(str).tolist()
        
        print(f"✅ Model loaded! Classes: {self.classes}")
        
        # State
        self.is_running = False
        self.current_level = 0
        self.pose_buffer = deque(maxlen=5)
        self.current_pose = None
        self.confidence = 0.0
        
        # Stats
        self.correct_frames = 0
        self.total_frames = 0
        self.combo = 0
        self.max_combo = 0
        self.last_was_correct = False
        
        # Levels
        self.levels = [
            {'name': 'Agem Kanan', 'target': 'Agem_Kanan', 'required': 70},
            {'name': 'Ngeed', 'target': 'Ngeed', 'required': 75},
            {'name': 'Ngegol', 'target': 'Ngegol', 'required': 80}
        ]
        
        # Camera
        self.cap = None
        self.pose_detector = None
        
        # Tutorial Video
        self.dataset_path = "Agem_Kanan"
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
            return None
        
        ret, frame = self.tutorial_cap.read()
        
        if not ret:
            # Loop video ke awal
            self.tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.tutorial_cap.read()
        
        if ret:
            # Resize to match webcam aspect
            frame = cv2.resize(frame, (640, 480))
            return frame
        
        return None
    
    def extract_landmarks(self, results):
        """Extract 12 pose landmarks"""
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
        """Predict pose from landmarks"""
        if landmarks is None:
            return None, 0.0
        
        X = pd.DataFrame([landmarks])
        y_int = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        label = self.classes[y_int]
        conf = proba[y_int]
        
        return label, conf
    
    def update_score(self, is_correct, confidence):
        """Update scoring with combo system"""
        self.total_frames += 1
        
        if is_correct and confidence > 0.75:
            self.correct_frames += 1
            
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
        """Calculate accuracy"""
        if self.total_frames == 0:
            return 0
        return (self.correct_frames / self.total_frames) * 100
    
    def get_status(self, is_correct, confidence):
        """Get status text"""
        if is_correct and confidence > 0.90:
            return "PERFECT!"
        elif is_correct and confidence > 0.80:
            return "GOOD"
        else:
            return "BAD"
    
    def start_detection(self, level_id):
        """Start pose detection loop"""
        self.current_level = level_id
        self.is_running = True
        self.reset_stats()
        self.load_tutorial_video()  # Load video untuk level yang dipilih
        
        self.cap = cv2.VideoCapture(0)
        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"🎮 Starting detection for level {level_id + 1}")
        
        # Run detection loop in thread
        thread = threading.Thread(target=self._detection_loop)
        thread.daemon = True
        thread.start()
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose_detector.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
                
                # Predict pose
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
                    
                    # Get status
                    status = self.get_status(is_correct, self.confidence)
                    
                    # Encode webcam frame to base64
                    _, buffer = cv2.imencode('.jpg', image)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Get tutorial frame
                    tutorial_frame = self.get_tutorial_frame()
                    tutorial_base64 = None
                    if tutorial_frame is not None:
                        _, tut_buffer = cv2.imencode('.jpg', tutorial_frame)
                        tutorial_base64 = base64.b64encode(tut_buffer).decode('utf-8')
                    
                    # Emit to frontend
                    socketio.emit('frame_update', {
                        'frame': frame_base64,
                        'tutorial_frame': tutorial_base64,
                        'current_pose': self.current_pose,
                        'confidence': float(self.confidence * 100),
                        'status': status,
                        'accuracy': float(self.get_accuracy()),
                        'combo': self.combo,
                        'score': {
                            'correct': self.correct_frames,
                            'total': self.total_frames
                        },
                        'target': target
                    })
            
            time.sleep(0.03)  # ~30 FPS
        
        self.cleanup()
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
    
    def reset_stats(self):
        """Reset statistics"""
        self.correct_frames = 0
        self.total_frames = 0
        self.combo = 0
        self.max_combo = 0
        self.last_was_correct = False
        self.pose_buffer.clear()
        self.current_pose = None
        self.confidence = 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        if self.pose_detector:
            self.pose_detector.close()
        if self.tutorial_cap:
            self.tutorial_cap.release()
        print("🛑 Detection stopped and resources cleaned")

# Global detector instance
detector = TariBaliDetector()

# Routes
@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'message': 'Tari Bali Backend Server',
        'levels': detector.levels
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    emit('connected', {'message': 'Connected to Tari Bali server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')
    detector.stop_detection()

@socketio.on('start_practice')
def handle_start_practice(data):
    level_id = data.get('level', 0)
    print(f'🎮 Starting practice for level {level_id + 1}')
    
    detector.start_detection(level_id)
    emit('practice_started', {
        'level': level_id,
        'target': detector.levels[level_id]['target']
    })

@socketio.on('stop_practice')
def handle_stop_practice():
    print('🛑 Stopping practice')
    detector.stop_detection()
    emit('practice_stopped', {
        'final_accuracy': detector.get_accuracy(),
        'max_combo': detector.max_combo,
        'total_frames': detector.total_frames
    })

@socketio.on('reset_practice')
def handle_reset_practice():
    print('🔄 Resetting practice')
    detector.reset_stats()
    emit('practice_reset', {'message': 'Stats reset successfully'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌸 TARI BALI - Backend Server")
    print("="*70)
    print("\n✨ Server starting...")
    print("📡 Backend URL: http://localhost:5000")
    print("🔌 WebSocket ready for connections")
    print("\n" + "="*70 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)