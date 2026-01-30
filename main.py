from utils import (read_video, 
                   save_video,
                   measure_distance_between_points,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   ShotClassifier,
                   draw_shot_classifications,
                   UILayoutManager
                   )
import cv2
import constants
import os
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt
import pandas as pd
from copy import deepcopy

# Feature toggle flags
ENABLE_SHOT_CLASSIFICATION = True  # Set to False to disable shot classification
ENABLE_PER_FRAME_KEYPOINTS = True  # Set to True for camera-robust detection (slower but more accurate)
USE_BYTETRACK = True  # Set to True for smooth ball trajectory with Kalman filter
ENABLE_POSE_TRACKING = True  # Set to True for pose estimation with BBoxMaskPose (v2.0)
POSE_MODEL = 'vitpose-b'  # Options: 'vitpose-b', 'maskpose-b', 'pmpose'

def main():
    try:
        # Reading video frames
        input_video_path = "input_videos/input_video_2.mp4"
        video_frames = read_video(input_video_path)
        print(f"Loaded {len(video_frames)} frames from {input_video_path}")

        # Detecting players and ball
        player_tracker = PlayerTracker(model_path="yolov8x")
        ball_tracker = BallTracker(model_path="models/last.pt")

        print("Detecting players...")
        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
        
        print("Detecting ball...")
        if USE_BYTETRACK:
            print("[BYTETRACK MODE] Using Kalman filter for smooth ball trajectory...")
            ball_detections = ball_tracker.detect_frames_with_tracking(
                video_frames, 
                read_from_stub=False,  # Must be False to use tracking
                stub_path="tracker_stubs/ball_detections_tracked.pkl"
            )
        else:
            ball_detections = ball_tracker.detect_frames(
                video_frames, 
                read_from_stub=True, 
                stub_path="tracker_stubs/ball_detections.pkl"
            )

        print("Interpolating ball positions...")
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

        # Court Line Detection
        print("Detecting court lines...")
        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        
        # CAMERA-ROBUST: Detect keypoints per-frame or just first frame
        if ENABLE_PER_FRAME_KEYPOINTS:
            print("[CAMERA-ROBUST MODE] Detecting keypoints for ALL frames...")
            print("Note: This takes longer but handles camera motion correctly.")
            all_court_keypoints = court_line_detector.predict_all_frames(
                video_frames, smooth=True, window_size=5
            )
            # For backward compatibility, also keep first frame's keypoints
            court_keypoints = all_court_keypoints[0]
        else:
            print("[FAST MODE] Detecting keypoints for first frame only...")
            court_keypoints = court_line_detector.predict(video_frames[0])
            # Create list with same keypoints for all frames (old behavior)
            all_court_keypoints = [court_keypoints] * len(video_frames)

        # Choose players
        print("Filtering players...")
        player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)

        # Normalize player IDs to 1 and 2
        # Use a list to ensure consistent ordering if needed, or simple iteration
        first_frame_players = player_detections[0].keys()
        # Sort to ensure consistent mapping across runs if needed, or just map as they appear
        player_ids = sorted(list(first_frame_players)) 
        
        # We expect exactly 2 players after filtering, but let's handle cases safely
        player_id_map = {}
        for i, original_id in enumerate(player_ids):
             # Map first ID to 1, second to 2. If more, ignore or map sequentially.
             if i < 2:
                player_id_map[original_id] = i + 1
        
        print(f"Mapping player IDs: {player_id_map}")

        # Update player_detections with new IDs
        normalized_player_detections = []
        for frame_detections in player_detections:
            normalized_frame = {}
            for original_id, bbox in frame_detections.items():
                if original_id in player_id_map:
                    new_id = player_id_map[original_id]
                    normalized_frame[new_id] = bbox
            normalized_player_detections.append(normalized_frame)
        
        player_detections = normalized_player_detections

        # POSE TRACKING (v2.0): Detect player poses using BBoxMaskPose
        player_poses = None
        if ENABLE_POSE_TRACKING:
            try:
                from trackers import PoseTracker
                print(f"\n[POSE TRACKING] Initializing {POSE_MODEL}...")
                pose_tracker = PoseTracker(model=POSE_MODEL)
                
                print("[POSE TRACKING] Detecting poses for all players...")
                player_poses = pose_tracker.detect_poses(
                    video_frames,
                    player_detections,
                    read_from_stub=True,
                    stub_path="tracker_stubs/pose_detections.pkl"
                )
                print(f"[POSE TRACKING] Detected poses for {len(player_poses)} frames\n")
                
                # Export pose data for external analysis
                pose_tracker.export_pose_data(
                    player_poses,
                    "analysis/pose_data.csv",
                    format='csv'
                )
            except ImportError as e:
                print(f"[POSE TRACKING] Skipping - dependencies not installed: {e}")
                print("[POSE TRACKING] Install with: pip install mmpose mmdet mmengine")
                ENABLE_POSE_TRACKING_LOCAL = False
            except Exception as e:
                print(f"[POSE TRACKING] Skipping - error: {e}")
                player_poses = None

        # CAMERA-ROBUST: Setup UI layout manager for dynamic positioning
        print("Setting up dynamic UI layout...")
        layout_manager = UILayoutManager(video_frames[0].shape, court_keypoints)
        mini_court_params = layout_manager.get_mini_court_params()
        stats_params = layout_manager.get_stats_panel_params()
        
        # MiniCourt with dynamic positioning
        print("Setting up mini court visualization...")
        mini_court = MiniCourt(video_frames[0], layout_params=mini_court_params) 

        # Detect ball shots
        print("Detecting ball shots...")
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
        print(f"Detected ball shots at frames: {ball_shot_frames}")

        # Convert positions to mini court positions
        # CAMERA-ROBUST: Now uses per-frame keypoints for accurate mapping
        print("Converting to mini court coordinates...")
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, all_court_keypoints)

        # Shot Classification (if enabled)
        shot_classifications = {}
        if ENABLE_SHOT_CLASSIFICATION:
            print("Classifying shots...")
            shot_classifier = ShotClassifier()
            
            # Use pose-aware classification if pose data available
            if player_poses is not None:
                print("[POSE-AWARE] Using pose data for enhanced shot classification...")
                shot_classifications = shot_classifier.classify_shots_with_pose(
                    player_mini_court_detections, 
                    ball_mini_court_detections, 
                    ball_shot_frames,
                    mini_court.court_height,
                    player_poses
                )
            else:
                # Fall back to position-based classification
                shot_classifications = shot_classifier.classify_shots(
                    player_mini_court_detections, 
                    ball_mini_court_detections, 
                    ball_shot_frames,
                    mini_court.court_height
                )
            print(f"Classified {len(shot_classifications)} shots")

        player_stats_data  = [{
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,

            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }]   

        for ball_shot_ind in range(len(ball_shot_frames)-1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind + 1]
            ball_shot_time_in_seconds = (end_frame - start_frame)/ 24 # 24 fps

            # Get distance covered by the ball
            distance_covered_by_ball_pixels = measure_distance_between_points(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])

            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())

            # Speed of the ball shot in km/h
            speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

            # player who made the shot
            player_positions = player_mini_court_detections[start_frame]
            player_shot_ball = min(player_positions.keys(), key=lambda x: measure_distance_between_points(player_positions[x], ball_mini_court_detections[start_frame][1]))

            # Opponent player speed
            opponent_player_id = 1 if player_shot_ball == 2 else 2
            speed_of_opponent_player = 0

            if opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]: 
                distance_covered_by_opponent_player_pixels = measure_distance_between_points(
                                                                                                player_mini_court_detections[start_frame][opponent_player_id],
                                                                                                player_mini_court_detections[end_frame][opponent_player_id]
                                                                                            )

                distance_covered_by_opponent_player_meters = convert_pixel_distance_to_meters(
                    distance_covered_by_opponent_player_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )
                # Speed of the opponent player
                speed_of_opponent_player = distance_covered_by_opponent_player_meters / ball_shot_time_in_seconds * 3.6  

            current_player_stats = deepcopy(player_stats_data[-1])
            current_player_stats["frame_num"] = start_frame
            current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
            current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
            current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot   

            current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += speed_of_opponent_player
            current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = speed_of_opponent_player

            # NEW: Add shot type to player stats if enabled
            if ENABLE_SHOT_CLASSIFICATION and start_frame in shot_classifications:
                shot_type = shot_classifications[start_frame]['shot_type']
                current_player_stats[f"player_{player_shot_ball}_shot_type"] = shot_type

            player_stats_data.append(current_player_stats)

        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({"frame_num": range(len(video_frames))})

        player_state_data_df = pd.merge(frames_df, player_stats_data_df, on="frame_num", how="left")
        player_state_data_df = player_state_data_df.ffill()

        # Fixed column names and division logic
        player_state_data_df["player_1_average_shot_speed"] = player_state_data_df["player_1_total_shot_speed"] / player_state_data_df["player_1_number_of_shots"].replace(0, 1)
        player_state_data_df["player_2_average_shot_speed"] = player_state_data_df["player_2_total_shot_speed"] / player_state_data_df["player_2_number_of_shots"].replace(0, 1)

        player_state_data_df["player_1_average_player_speed"] = player_state_data_df["player_1_total_player_speed"] / player_state_data_df["player_1_number_of_shots"].replace(0, 1)
        player_state_data_df["player_2_average_player_speed"] = player_state_data_df["player_2_total_player_speed"] / player_state_data_df["player_2_number_of_shots"].replace(0, 1)

        
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
        
        # Draw Player Skeletons (v2.0): Overlay pose skeletons on players
        if ENABLE_POSE_TRACKING and player_poses is not None:
            try:
                print("Drawing player skeletons...")
                output_video_frames = pose_tracker.draw_skeletons(
                    output_video_frames, player_poses, thickness=2, radius=4
                )
            except Exception as e:
                print(f"[POSE TRACKING] Skeleton drawing failed: {e}")
        
        # Draw Ball Bounding Boxes - with enhanced visibility
        print("Drawing ball bounding boxes...")
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, color=(0, 255, 255), thickness=2)

        # Draw Player Stats with dynamic positioning
        print("Drawing player stats...")
        output_video_frames = draw_player_stats(output_video_frames, player_state_data_df, stats_params)

        # Draw Court Keypoints - CAMERA-ROBUST: Uses per-frame keypoints
        print("Drawing court keypoints...")
        if ENABLE_PER_FRAME_KEYPOINTS:
            output_video_frames = court_line_detector.draw_keypoints_on_video_dynamic(
                output_video_frames, all_court_keypoints, point_color=(0, 140, 255), radius=5)
        else:
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

        # NEW: Add shot classification overlays if enabled
        if ENABLE_SHOT_CLASSIFICATION:
            print("Adding shot classification overlays...")
            output_video_frames = draw_shot_classifications(output_video_frames, shot_classifications, ball_shot_frames)

        # Save output video
        print("Saving output video...")
        output_video_path = "output_videos/output_video.avi"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Save video with additional error handling
        success = save_video(output_video_frames, output_video_path)
        
        if success:
            print(f"Processing complete! Video saved to {output_video_path}")
        else:
            print(f"ERROR: Failed to save video to {output_video_path}")
            
            # Try alternative format as fallback
            print("Attempting to save as MP4 instead...")
            output_video_path_mp4 = "output_videos/output_video_2.mp4"
            success_mp4 = save_video(output_video_frames, output_video_path_mp4)
            
            if success_mp4:
                print(f"Successfully saved video as MP4 to {output_video_path_mp4}")
            else:
                print("CRITICAL ERROR: All video saving attempts failed.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()