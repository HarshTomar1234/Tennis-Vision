from utils import (read_video, 
                   save_video,
                   measure_distance_between_points,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   ShotClassifier,
                   draw_shot_classifications
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

def main():
    try:
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

        # NEW: Shot Classification (if enabled)
        shot_classifications = {}
        if ENABLE_SHOT_CLASSIFICATION:
            print("Classifying shots...")
            shot_classifier = ShotClassifier()
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
        
        # Draw Ball Bounding Boxes - with enhanced visibility
        print("Drawing ball bounding boxes...")
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, color=(0, 255, 255), thickness=2)

        # Draw Player Stats
        print("Drawing player stats...")
        output_video_frames = draw_player_stats(output_video_frames, player_state_data_df)

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
            output_video_path_mp4 = "output_videos/output_video.mp4"
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