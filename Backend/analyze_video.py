import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

"""
Tool untuk analisis video dan cari timestamp gerakan

Cara pakai:
1. Video akan play dengan pose detection
2. Tekan SPACE untuk pause dan catat timestamp
3. Catat detik mulai dan selesai setiap gerakan
"""

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Tidak bisa buka video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print("\n" + "="*70)
    print("VIDEO ANALYSIS TOOL")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    print("\n📋 CONTROLS:")
    print("  SPACE: Pause/Resume")
    print("  →: Forward 1 second")
    print("  ←: Backward 1 second")
    print("  ↑: Forward 5 seconds")
    print("  ↓: Backward 5 seconds")
    print("  M: Mark timestamp (catat ke konsol)")
    print("  Q: Quit")
    
    print("\n🎯 GERAKAN YANG DICARI:")
    print("  1. Jump-1Leg (Loncat Kaki Satu)")
    print("  2. Arms-UpDown (Tangan Atas-Bawah)")
    print("  3. Arms-LeftRight (Tangan Kiri-Kanan)")
    print("  4. Arms-BothUp (Tangan Keduanya Atas)")
    print("  5. Water-Splash (Percik Air)")
    print("  6. Fly (Terbang)")
    
    input("\nTekan ENTER untuk mulai...")
    
    paused = False
    frame_num = 0
    marked_timestamps = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⏹️  Video selesai")
                    break
                frame_num += 1
            else:
                # Stay on same frame when paused
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                ret, frame = cap.read()
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )
            
            # Resize for display
            image = cv2.resize(image, (1200, 675))
            h, w = image.shape[:2]
            
            # Calculate current time
            current_time = frame_num / fps
            
            # Draw info overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Time info
            time_text = f"Time: {current_time:.2f}s / {duration:.2f}s"
            cv2.putText(image, time_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Frame info
            frame_text = f"Frame: {frame_num}/{total_frames}"
            cv2.putText(image, frame_text, (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Status
            status = "PAUSED" if paused else "PLAYING"
            status_color = (0, 0, 255) if paused else (0, 255, 0)
            cv2.putText(image, status, (w - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Controls hint
            cv2.rectangle(image, (0, h-50), (w, h), (40, 40, 40), -1)
            cv2.putText(image, "SPACE: Pause | Arrow Keys: Navigate | M: Mark | Q: Quit", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Video Analysis - Cari Timestamp Gerakan', image)
            
            # Handle keyboard input
            wait_time = 1 if not paused else 0
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord(' '):
                paused = not paused
                print(f"\n{'⏸️  PAUSED' if paused else '▶️  PLAYING'} at {current_time:.2f}s")
            
            elif key == ord('m'):
                marked_timestamps.append(current_time)
                print(f"\n✓ MARKED: {current_time:.2f}s (Frame {frame_num})")
            
            elif key == 83:  # Right arrow
                # Forward 1 second
                new_frame = min(frame_num + int(fps), total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_num = new_frame
                print(f"→ Forward to {frame_num/fps:.2f}s")
            
            elif key == 81:  # Left arrow
                # Backward 1 second
                new_frame = max(frame_num - int(fps), 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_num = new_frame
                print(f"← Backward to {frame_num/fps:.2f}s")
            
            elif key == 82:  # Up arrow
                # Forward 5 seconds
                new_frame = min(frame_num + int(fps * 5), total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_num = new_frame
                print(f"⇧ Forward to {frame_num/fps:.2f}s")
            
            elif key == 84:  # Down arrow
                # Backward 5 seconds
                new_frame = max(frame_num - int(fps * 5), 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_num = new_frame
                print(f"⇩ Backward to {frame_num/fps:.2f}s")
            
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*70)
    print("MARKED TIMESTAMPS:")
    print("="*70)
    
    if marked_timestamps:
        for i, ts in enumerate(marked_timestamps, 1):
            print(f"{i}. {ts:.2f}s")
        
        print("\n📝 FORMAT UNTUK TRAINING:")
        print("\nMOVEMENTS = {")
        print("    'NamaGerakan': [")
        for i in range(0, len(marked_timestamps), 2):
            if i + 1 < len(marked_timestamps):
                start = int(marked_timestamps[i])
                end = int(marked_timestamps[i + 1])
                print(f"        {{'start': {start}, 'end': {end}, 'fps': 30}},")
        print("    ],")
        print("}")
    else:
        print("  (Tidak ada timestamp yang di-mark)")
    
    print("\n💡 TIPS:")
    print("  - Mark timestamp MULAI gerakan (tekan M)")
    print("  - Mark timestamp SELESAI gerakan (tekan M lagi)")
    print("  - Ulangi untuk setiap gerakan")

if __name__ == '__main__':
    import sys
    
    video_path = "vid/raw/sajojo_new.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    print(f"\nAnalyzing video: {video_path}")
    print("Make sure video exists in the path!")
    
    analyze_video(video_path)