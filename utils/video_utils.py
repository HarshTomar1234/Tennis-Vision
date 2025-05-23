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

    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    height, width = output_video_frames[0].shape[:2]
    
  
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    
    if not out.isOpened():
        print(f"ERROR: Could not open video writer for {output_video_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
        
        if not out.isOpened():
            print(f"CRITICAL ERROR: Failed to create video writer with multiple codecs")
            return False
    
    
    for frame in output_video_frames:
        out.write(frame)
    
    
    out.release()
    
    
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        print(f"Successfully saved video to {output_video_path}")
        return True
    else:
        print(f"WARNING: Video file creation failed or file is empty")
        return False