import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get frame dimensions
    height, width = output_video_frames[0].shape[:2]
    
    # Use a more reliable codec (XVID is widely compatible)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    # Check if VideoWriter was successfully created
    if not out.isOpened():
        print(f"ERROR: Could not open video writer for {output_video_path}")
        # Try another codec as fallback
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
        
        if not out.isOpened():
            print(f"CRITICAL ERROR: Failed to create video writer with multiple codecs")
            return False
    
    # Write frames to video
    for frame in output_video_frames:
        out.write(frame)
    
    # Release resources
    out.release()
    
    # Verify the video was created
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        print(f"Successfully saved video to {output_video_path}")
        return True
    else:
        print(f"WARNING: Video file creation failed or file is empty")
        return False
