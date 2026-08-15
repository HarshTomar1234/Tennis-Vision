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

def save_video(output_video_frames, output_video_path, fps=24.0):
    """
    Write annotated frames to disk.

    `fps` must be the SOURCE video's frame rate. It was hardcoded to 24, so a 30 fps
    clip was written 25% slow: the demo video played in mild slow motion, and its clock
    drifted against the 3-D viewer's timeline, which uses the real rate. Every rendered
    output shared that error.
    """
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height, width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video_path, fourcc, float(fps) if fps and fps > 0 else 24.0,
                          (width, height))
    
    
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