import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

# Cek apakah model ada
if not os.path.exists('model/tari_bali_model.pkl'):
    print("❌ Model tidak ditemukan!")
    print("Pastikan file ada di: model/tari_bali_model.pkl")
    exit()

if not os.path.exists('model/tari_bali_model_classes.txt'):
    print("❌ Classes file tidak ditemukan!")
    print("Pastikan file ada di: model/tari_bali_model_classes.txt")
    exit()

print("✅ Loading model...")

# Load model
model = joblib.load('model/tari_bali_model.pkl')

# Load classes
with open('model/tari_bali_model_classes.txt', 'r') as f:
    classes = f.read().strip().split(',')

print(f"✅ Model loaded!")
print(f"✅ Classes: {classes}")

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("\n🎥 Opening webcam...")
print("Press 'Q' to quit")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

print("✅ Webcam ready!\n")

with mp_pose.Pose(min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame (mirror)
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
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                body_landmarks = []
                
                # 12 keypoints
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
                
                # Predict
                X = pd.DataFrame([body_landmarks])
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                confidence = probabilities.max()
                
                # Display result
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                
                # Background box
                cv2.rectangle(image, (10, 10), (500, 120), (0, 0, 0), -1)
                
                # Text
                cv2.putText(image, f'Gerakan: {prediction}', 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, color, 2)
                cv2.putText(image, f'Confidence: {confidence*100:.1f}%', 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2)
            
            except Exception as e:
                pass
        
        # Show
        cv2.imshow('Test Model Tari Bali', image)
        
        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Test selesai!")