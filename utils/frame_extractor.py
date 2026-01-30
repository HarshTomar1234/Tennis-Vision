"""
Shot Frame Extractor
=====================
Extracts frames from video where shots occur (ball hit, serve, etc.)
Use this to inspect/debug the shot detection and ball tracking.

Usage:
    python utils/frame_extractor.py

Output:
    Creates images in output_videos/debug_frames/
"""

import cv2
import os
import pickle
import sys

def extract_shot_frames(
    video_path="input_videos/input_video_2.mp4",
    output_dir="output_videos/debug_frames",
    ball_stub_path="tracker_stubs/ball_detections.pkl",
    frames_before=5,
    frames_after=10
):
    """
    Extract frames around shot moments for debugging.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        ball_stub_path: Path to ball detection pickle
        frames_before: Frames to extract before shot
        frames_after: Frames to extract after shot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Load ball detections to find shot moments
    try:
        with open(ball_stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        print(f"Loaded ball detections for {len(ball_detections)} frames")
    except FileNotFoundError:
        print(f"Warning: Ball detections not found at {ball_stub_path}")
        ball_detections = None
    
    # Find shot frames (where ball detection suddenly appears/disappears or changes significantly)
    shot_frames = find_shot_frames(ball_detections) if ball_detections else []
    
    print(f"\nFound {len(shot_frames)} potential shot moments")
    print(f"Shot frames: {shot_frames[:20]}{'...' if len(shot_frames) > 20 else ''}")
    
    # Extract frames around each shot
    frames_extracted = 0
    for shot_idx, shot_frame in enumerate(shot_frames[:30]):  # Limit to first 30 shots
        start = max(0, shot_frame - frames_before)
        end = min(total_frames - 1, shot_frame + frames_after)
        
        # Create subdirectory for this shot
        shot_dir = os.path.join(output_dir, f"shot_{shot_idx:03d}_frame_{shot_frame}")
        os.makedirs(shot_dir, exist_ok=True)
        
        # Extract frames
        for frame_num in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Add frame info text
                rel_frame = frame_num - shot_frame
                label = "SHOT" if rel_frame == 0 else f"{rel_frame:+d}"
                cv2.putText(frame, f"Frame {frame_num} ({label})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw ball detection if available
                if ball_detections and frame_num < len(ball_detections):
                    ball_dict = ball_detections[frame_num]
                    if 1 in ball_dict:
                        x1, y1, x2, y2 = ball_dict[1]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                     (0, 255, 255), 2)
                        cv2.putText(frame, "Ball", (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Save frame
                filename = f"frame_{frame_num:04d}_{label.replace('+', 'plus').replace('-', 'minus')}.jpg"
                cv2.imwrite(os.path.join(shot_dir, filename), frame)
                frames_extracted += 1
    
    cap.release()
    print(f"\n✅ Extracted {frames_extracted} frames to {output_dir}")
    print(f"   Check subdirectories for each shot moment")


def find_shot_frames(ball_detections, threshold=50):
    """
    Find frames where shots likely occurred based on ball detection patterns.
    
    A shot is detected when:
    1. Ball appears after not being visible
    2. Ball position changes significantly (hit)
    3. Ball velocity changes direction
    """
    shot_frames = []
    prev_pos = None
    prev_detected = False
    
    for frame_num, ball_dict in enumerate(ball_detections):
        has_ball = 1 in ball_dict and ball_dict[1] is not None
        
        if has_ball:
            x1, y1, x2, y2 = ball_dict[1]
            curr_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Ball just appeared
            if not prev_detected and has_ball:
                shot_frames.append(frame_num)
            
            # Ball position changed significantly
            elif prev_pos is not None:
                dist = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
                if dist > threshold:
                    shot_frames.append(frame_num)
            
            prev_pos = curr_pos
            prev_detected = True
        else:
            prev_detected = False
    
    # Remove duplicates within 5 frames
    filtered = []
    for f in shot_frames:
        if not filtered or f - filtered[-1] > 5:
            filtered.append(f)
    
    return filtered


def extract_specific_frames(video_path, frame_numbers, output_dir="output_videos/debug_frames/specific"):
    """
    Extract specific frame numbers from video.
    
    Args:
        video_path: Path to video
        frame_numbers: List of frame numbers to extract
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_num:04d}.jpg"), frame)
            print(f"  Saved frame {frame_num}")
    
    cap.release()
    print(f"✅ Saved {len(frame_numbers)} frames to {output_dir}")


if __name__ == "__main__":
    print("="*60)
    print("       SHOT FRAME EXTRACTOR - Debug Tool")
    print("="*60)
    
    # Extract frames around shot moments
    extract_shot_frames()
    
    print("\n" + "="*60)
    print("Done! Check output_videos/debug_frames/ for extracted images")
    print("="*60)
