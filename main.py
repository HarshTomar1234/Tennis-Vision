from utils import (read_video, save_video)
import cv2

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt


def main():

    # Reading video frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    print(f"Loaded {len(video_frames)} frames from {input_video_path}")

    # Detecting players and ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/last.pt")

    print("Detecting players...")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    
    print("Detecting ball...")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    print("Interpolating ball positions...")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detection
    print("Detecting court lines...")
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players
    print("Filtering players...")
    player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)

    # MiniCourt
    print("Setting up mini court visualization...")
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    print("Detecting ball shots...")
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"Detected ball shots at frames: {ball_shot_frames}")

    # Convert positions to mini court positions
    print("Converting to mini court coordinates...")
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Create initial output frames
    print("Creating output video...")
    output_video_frames = video_frames.copy()

    # ENHANCEMENT: Implement additional validation and confidence threshold for more accurate detections
    # Higher confidence thresholds for both player and ball detections to reduce false positives
    player_confidence_threshold = 0.7  # Only consider high-confidence player detections
    ball_confidence_threshold = 0.6    # Slightly lower for ball as it's smaller and harder to detect
    
    # ENHANCEMENT: Apply additional filtering to player detections based on court position and size
    print("Enhancing player detection accuracy...")
    player_detections = player_tracker.filter_by_confidence(player_detections, player_confidence_threshold)
    
    # ENHANCEMENT: Apply additional filtering to ball detections
    print("Enhancing ball detection accuracy...")
    ball_detections = ball_tracker.filter_by_confidence(ball_detections, ball_confidence_threshold)
    
    # Draw Player Bounding Boxes - with darker, more prominent outlines
    print("Drawing player bounding boxes...")
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections, thickness=2)
    
    # Draw Ball Bounding Boxes - with enhanced visibility
    print("Drawing ball bounding boxes...")
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, color=(0, 255, 255), thickness=2)

    # Draw Court Keypoints - matching the keypoints that will be shown on mini court
    print("Drawing court keypoints...")
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints, point_color=(0, 140, 255), radius=5)

    # ENHANCEMENT: Draw Mini Court with improved visual styling without labels
    print("Drawing mini court with enhanced styling...")
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    
    # ENHANCEMENT: Draw ball trajectory first as background layer with improved visual style
    print("Visualizing ball trajectory with enhanced visualization...")
    output_video_frames = mini_court.draw_ball_trajectory(output_video_frames, ball_mini_court_detections)
    
    # ENHANCEMENT: Draw players with improved visibility - darker, more prominent circles
    # Note: removed labels per user request
    print("Drawing player positions with enhanced visualization...")
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections, color=(0, 255, 0), draw_trail=True, label=None)
    
    # ENHANCEMENT: Draw current ball position with improved visibility
    # Note: removed labels per user request
    print("Drawing ball positions with enhanced visualization...")
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255), label=None)

    # Draw frame number and additional info on top left corner
    print("Adding frame information...")
    for i, frame in enumerate(output_video_frames):
        # Draw frame number
        cv2.putText(frame, f"Frame: {i}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Indicate if this is a ball shot frame
        if i in ball_shot_frames:
            cv2.putText(frame, "BALL SHOT!",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save output video
    print("Saving output video...")
    save_video(output_video_frames, "output_videos/output_video.avi")
    
    print("Processing complete! Video saved to output_videos/output_video.avi")



if __name__ == "__main__":
    main()