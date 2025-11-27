import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import argparse
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def predict_video(input_video, model_name, show_landmarks=True):
    """
    Prediksi gerakan tari dari video
    
    Args:
        input_video: Path ke video input
        model_name: Nama model (tanpa .pkl)
        show_landmarks: Tampilkan pose landmarks atau tidak
    """
    
    # Load model
    model_path = f'model/{model_name}.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return
    
    print(f"✅ Loading model: {model_path}")
    model = joblib.load(model_path)
    
    # Load classes
    with open(f'model/{model_name}_classes.txt', 'r') as f:
        classes = f.read().split(',')
    print(f"✅ Classes: {classes}")
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"❌ Gagal membuka video: {input_video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video
    os.makedirs('vid/annotated', exist_ok=True)
    output_path = f'vid/annotated/output_{os.path.basename(input_video)}'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🎬 Memproses video...")
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, 
                     min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Detect pose
            results = pose.process(image_rgb)
            
            # Back to BGR
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if show_landmarks and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Predict
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    body_landmarks = []
                    
                    # 12 key points
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
                    
                    # Tampilkan hasil
                    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                    
                    # Background box
                    cv2.rectangle(image, (10, 10), (500, 100), (0, 0, 0), -1)
                    
                    # Text
                    cv2.putText(image, f'Gerakan: {prediction}', 
                               (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, color, 2)
                    cv2.putText(image, f'Confidence: {confidence*100:.1f}%', 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    print(f"Frame {frame_count}: {prediction} ({confidence*100:.1f}%)")
            
            except Exception as e:
                print(f"Error di frame {frame_count}: {e}")
            
            # Write frame
            out.write(image)
            
            # Display
            cv2.imshow('Tari Bali - Pose Prediction', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Video output disimpan: {output_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-video", required=True,
                    help="Path ke video input")
    ap.add_argument("--model-name", type=str, default='tari_bali_model',
                    help="Nama model (tanpa .pkl)")
    ap.add_argument("--hide-landmarks", action='store_true',
                    help="Sembunyikan pose landmarks")
    
    args = vars(ap.parse_args())
    
    predict_video(
        args['input_video'],
        args['model_name'],
        show_landmarks=not args['hide_landmarks']
    )