import cv2

cap = cv2.VideoCapture('vid/raw/sajojo_tutorial.mp4.webm')

if cap.isOpened():
    print("✓ Video can be read!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps
    print(f"  FPS: {fps}")
    print(f"  Total frames: {int(frames)}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Read first frame to verify
    ret, frame = cap.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
        print("✓ All good! Ready to extract!")
    else:
        print("✗ Cannot read frames")
else:
    print("✗ Cannot open video")

cap.release()