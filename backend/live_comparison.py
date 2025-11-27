import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import argparse
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_pose_landmarks(landmarks):
    """Extract 24 features (12 landmarks x 2 coordinates) for model prediction"""
    try:
        body_landmarks = []
        
        pose_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        
        for idx in pose_indices:
            body_landmarks.append(landmarks[idx].x)
            body_landmarks.append(landmarks[idx].y)
        
        return np.around(body_landmarks, decimals=9).tolist()
    except:
        return None

def predict_pose(landmarks, model):
    """Predict pose using trained model"""
    try:
        features = extract_pose_landmarks(landmarks)
        if features is None:
            return None, None, None
        
        X = pd.DataFrame([features])
        pose_class = model.predict(X)[0]
        pose_prob = model.predict_proba(X)[0]
        confidence = np.max(pose_prob)
        
        return pose_class, confidence, pose_prob
    except Exception as e:
        return None, None, None

def get_feedback_color(tutorial_pose, user_pose, confidence):
    """Get color based on pose matching and confidence"""
    if tutorial_pose is None or user_pose is None:
        return (100, 100, 100), "NO DETECT"
    
    tutorial_main = tutorial_pose.split(' ')[0] if ' ' in tutorial_pose else tutorial_pose
    user_main = user_pose.split(' ')[0] if ' ' in user_pose else user_pose
    
    # Both are "Miss" - no pose detected properly
    if tutorial_main == "Miss" and user_main == "Miss":
        return (100, 100, 100), "TRANSITION"
    
    # User is "Miss" but tutorial has pose
    if user_main == "Miss" and tutorial_main != "Miss":
        return (130, 137, 245), "MISS"
    
    # Tutorial is "Miss" (transition) - don't penalize user
    if tutorial_main == "Miss":
        return (200, 200, 200), "TRANSITION"
    
    # Compare actual poses
    if tutorial_main == user_main:
        if confidence >= 0.8:
            return (134, 240, 125), "PERFECT!"
        elif confidence >= 0.6:
            return (130, 245, 231), "GOOD"
        else:
            return (255, 200, 100), "LOW CONF"
    else:
        return (130, 137, 245), "WRONG POSE"

def draw_pose_info(frame, pose_class, confidence, color, feedback_text, position='top'):
    """Draw pose information on frame"""
    height, width = frame.shape[:2]
    
    # Background bar
    cv2.rectangle(frame, (0, 0), (width, 80), color, -1)
    
    if pose_class:
        # Pose name
        pose_display = pose_class if pose_class != "Miss" else "---"
        cv2.putText(
            frame,
            f'Pose: {pose_display}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Confidence
        if pose_class != "Miss":
            cv2.putText(
                frame,
                f'{int(confidence * 100)}%',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        
        # Feedback text (for user only)
        if feedback_text:
            cv2.putText(
                frame,
                feedback_text,
                (width - 200, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
    else:
        cv2.putText(
            frame,
            'No pose detected',
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutorial-video", type=str, required=True,
                    help="Tutorial video file in vid/raw/")
    ap.add_argument("--output-name", type=str, default="comparison_challenge",
                    help="Output video name (without extension)")
    ap.add_argument("--countdown", type=int, default=5,
                    help="Countdown seconds before starting")
    ap.add_argument("--show-landmarks", action='store_true',
                    help="Show pose landmarks (skeleton)")
    args = vars(ap.parse_args())

    # Load key poses mapping
    key_poses_path = 'data/key_poses.xlsx'
    if not os.path.exists(key_poses_path):
        print(f"Error: {key_poses_path} not found!")
        print("This file is needed for Challenge Mode to know which model to use per frame.")
        exit()
    
    print(f"Loading key poses mapping: {key_poses_path}")
    key_poses_df = pd.read_excel(key_poses_path)
    print("✓ Key poses loaded successfully")
    print(f"  Total keypoint sections: {len(key_poses_df)}")

    # Load all models (model1 to model10)
    models = {}
    print("\nLoading models...")
    for i in range(1, 11):
        model_path = f'model/model{i}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[f'model{i}'] = joblib.load(f)
            print(f"  ✓ model{i} loaded")
        else:
            print(f"  ⚠ model{i} not found (skipping)")
    
    if not models:
        print("Error: No models loaded!")
        exit()
    
    print(f"\n✓ Total models loaded: {len(models)}")

    # Open tutorial video
    tutorial_cap = cv2.VideoCapture(f'vid/raw/{args["tutorial_video"]}')
    if not tutorial_cap.isOpened():
        print(f"Error: Cannot open tutorial video vid/raw/{args['tutorial_video']}")
        exit()

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Cannot open webcam")
        exit()

    # Get video properties
    fps = int(tutorial_cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    # Set target resolution
    target_width = 640
    target_height = 480
    
    # Combined frame size
    combined_width = target_width * 2
    combined_height = target_height
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = f'vid/annotated/{args["output_name"]}.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

    # Countdown
    print(f"\n{'='*60}")
    print(f"🎬 CHALLENGE MODE - Auto Multi-Model")
    print(f"{'='*60}")
    print(f"Tutorial: {args['tutorial_video']}")
    print(f"Models: Auto-switching based on frame timing")
    print(f"Output: {output_path}")
    print(f"\n⏰ Starting in {args['countdown']} seconds...")
    print("💃 Get ready to dance!")
    
    for i in range(args['countdown'], 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🚀 GO! Press 'Q' to stop\n")

    frame_count = 0
    total_frames = int(tutorial_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    match_count = 0
    total_compared = 0
    current_model_name = None
    current_pose_name = None
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while tutorial_cap.isOpened() and webcam.isOpened():
            # Read frames
            ret_tutorial, frame_tutorial = tutorial_cap.read()
            ret_webcam, frame_webcam = webcam.read()
            
            if not ret_tutorial or not ret_webcam:
                break
            
            # Flip webcam
            frame_webcam = cv2.flip(frame_webcam, 1)
            
            # Resize
            frame_tutorial = cv2.resize(frame_tutorial, (target_width, target_height))
            frame_webcam = cv2.resize(frame_webcam, (target_width, target_height))
            
            # Process tutorial
            image_tutorial = cv2.cvtColor(frame_tutorial, cv2.COLOR_BGR2RGB)
            image_tutorial.flags.writeable = False
            results_tutorial = pose.process(image_tutorial)
            image_tutorial.flags.writeable = True
            image_tutorial = cv2.cvtColor(image_tutorial, cv2.COLOR_RGB2BGR)
            
            # Process webcam
            image_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            image_webcam.flags.writeable = False
            results_webcam = pose.process(image_webcam)
            image_webcam.flags.writeable = True
            image_webcam = cv2.cvtColor(image_webcam, cv2.COLOR_RGB2BGR)
            
            # Get the correct model for current frame (Challenge Mode!)
            filtered_df = key_poses_df[
                (key_poses_df['start_frame'] <= frame_count) & 
                (key_poses_df['end_frame'] >= frame_count)
            ]
            
            tutorial_pose = None
            tutorial_conf = 0
            user_pose = None
            user_conf = 0
            
            if not filtered_df.empty:
                current_model_row = filtered_df.iloc[0]
                model_name = current_model_row['model_name']
                expected_pose = current_model_row['class_name']
                
                # Track model changes
                if model_name != current_model_name:
                    current_model_name = model_name
                    current_pose_name = expected_pose
                    print(f"[Frame {frame_count}] Switching to {model_name} (Expected: {expected_pose})")
                
                if model_name in models:
                    model = models[model_name]
                    
                    # Predict tutorial pose
                    tutorial_pose, tutorial_conf, _ = predict_pose(
                        results_tutorial.pose_landmarks.landmark if results_tutorial.pose_landmarks else None,
                        model
                    )
                    
                    # Predict user pose
                    user_pose, user_conf, _ = predict_pose(
                        results_webcam.pose_landmarks.landmark if results_webcam.pose_landmarks else None,
                        model
                    )
            
            # Calculate matching (only for valid poses, not "Miss" or transitions)
            if tutorial_pose and user_pose:
                tutorial_main = tutorial_pose.split(' ')[0]
                user_main = user_pose.split(' ')[0]
                
                # Only count if both are actual poses (not "Miss")
                if tutorial_main != "Miss" and user_main != "Miss":
                    total_compared += 1
                    if tutorial_main == user_main and user_conf >= 0.6:
                        match_count += 1
            
            # Get feedback
            feedback_color, feedback_text = get_feedback_color(tutorial_pose, user_pose, user_conf)
            
            # Draw landmarks if enabled
            if args['show_landmarks']:
                if results_tutorial.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image_tutorial,
                        results_tutorial.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )
                
                if results_webcam.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image_webcam,
                        results_webcam.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )
            
            # Draw pose info
            draw_pose_info(image_webcam, user_pose, user_conf, feedback_color, feedback_text)
            draw_pose_info(image_tutorial, tutorial_pose, tutorial_conf, (100, 100, 200), None)
            
            # Add labels
            cv2.putText(image_webcam, 'YOU', (target_width - 80, target_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image_tutorial, 'TUTORIAL', (10, target_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Progress bar
            progress = frame_count / total_frames if total_frames > 0 else 0
            bar_width = int(target_width * progress)
            cv2.rectangle(image_tutorial, (0, target_height - 8), 
                         (bar_width, target_height), (0, 255, 0), -1)
            
            # Time display
            current_time = frame_count / fps
            total_time = total_frames / fps
            time_text = f"{int(current_time)}s/{int(total_time)}s"
            cv2.putText(image_tutorial, time_text, (target_width - 120, target_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Model indicator on tutorial
            if current_model_name:
                cv2.putText(image_tutorial, current_model_name, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Combine frames
            combined_frame = np.hstack([image_webcam, image_tutorial])
            
            # Write and display
            out.write(combined_frame)
            cv2.imshow('Dance Comparison - Challenge Mode - Press Q to Stop', combined_frame)
            
            frame_count += 1
            
            # Progress every 30 frames
            if frame_count % 30 == 0:
                accuracy = (match_count / total_compared * 100) if total_compared > 0 else 0
                print(f"Frame {frame_count}/{total_frames} | Accuracy: {accuracy:.1f}% | Current: {current_pose_name}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Stopping...")
                break
    
    # Release
    tutorial_cap.release()
    webcam.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final stats
    accuracy = (match_count / total_compared * 100) if total_compared > 0 else 0
    print(f"\n{'='*60}")
    print(f"✅ CHALLENGE MODE COMPLETED!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {int(frame_count / fps)}s")
    print(f"   Poses compared: {total_compared}")
    print(f"   Poses matched: {match_count}")
    print(f"   Overall accuracy: {accuracy:.1f}%")
    print(f"\n💾 Video saved: {output_path}")
    print(f"{'='*60}\n")