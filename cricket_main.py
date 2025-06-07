from utils import (read_video, 
                   save_video,
                   measure_distance_between_points,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   CricketShotClassifier,
                   draw_cricket_shot_classifications
                   )
import cv2
import cricket_constants
import os
from cricket_trackers import PlayerTracker, BallTracker
from cricket_pitch_detector import CricketPitchDetector
from mini_cricket_pitch import MiniCricketPitch
import pandas as pd
from copy import deepcopy

# Feature toggle flags
ENABLE_SHOT_CLASSIFICATION = True
ENABLE_BOWLING_ANALYSIS = True
ENABLE_FIELDING_ANALYSIS = True

def main():
    try:
        # Reading video frames
        input_video_path = "input_videos/cricket_match.mp4"
        video_frames = read_video(input_video_path)
        print(f"Loaded {len(video_frames)} frames from {input_video_path}")

        # Detecting players and ball
        player_tracker = PlayerTracker(model_path="yolov8x")
        ball_tracker = BallTracker(model_path="models/cricket_ball_model.pt")

        print("Detecting players...")
        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/cricket_player_detections.pkl")
        
        print("Detecting ball...")
        ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/cricket_ball_detections.pkl")

        print("Interpolating ball positions...")
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

        # Cricket Pitch Detection
        print("Detecting cricket pitch...")
        pitch_model_path = "models/cricket_pitch_keypoints.pth"
        pitch_detector = CricketPitchDetector(pitch_model_path)
        pitch_keypoints = pitch_detector.predict(video_frames[0])

        # Filter players (batsman, bowler, wicket-keeper, fielders)
        print("Identifying cricket players...")
        player_detections = player_tracker.identify_cricket_players(player_detections, pitch_keypoints)

        # Mini Cricket Pitch
        print("Setting up mini cricket pitch visualization...")
        mini_pitch = MiniCricketPitch(video_frames[0])

        # Detect ball events (deliveries, shots, boundaries)
        print("Detecting cricket ball events...")
        ball_event_frames = ball_tracker.get_cricket_ball_events(ball_detections)
        print(f"Detected cricket events at frames: {ball_event_frames}")

        # Convert positions to mini pitch coordinates
        print("Converting to mini pitch coordinates...")
        player_mini_pitch_detections, ball_mini_pitch_detections = mini_pitch.convert_bounding_boxes_to_mini_pitch_coordinates(
            player_detections, ball_detections, pitch_keypoints)

        # Cricket Shot Classification
        shot_classifications = {}
        if ENABLE_SHOT_CLASSIFICATION:
            print("Classifying cricket shots...")
            shot_classifier = CricketShotClassifier()
            shot_classifications = shot_classifier.classify_cricket_shots(
                player_mini_pitch_detections, 
                ball_mini_pitch_detections, 
                ball_event_frames,
                mini_pitch.pitch_length
            )
            print(f"Classified {len(shot_classifications)} cricket shots")

        # Initialize cricket statistics
        cricket_stats_data = [{
            "frame_num": 0,
            "batsman_shots": 0,
            "batsman_total_runs": 0,
            "batsman_strike_rate": 0,
            "bowler_deliveries": 0,
            "bowler_economy_rate": 0,
            "ball_speed": 0,
            "shot_power": 0,
            "fielding_efficiency": 0,
        }]

        # Process cricket events
        for event_ind in range(len(ball_event_frames)-1):
            start_frame = ball_event_frames[event_ind]
            end_frame = ball_event_frames[event_ind + 1]
            event_time_seconds = (end_frame - start_frame) / 24  # 24 fps

            # Calculate ball speed and distance
            if start_frame in ball_mini_pitch_detections and end_frame in ball_mini_pitch_detections:
                distance_covered_pixels = measure_distance_between_points(
                    ball_mini_pitch_detections[start_frame][1], 
                    ball_mini_pitch_detections[end_frame][1]
                )
                
                distance_covered_meters = convert_pixel_distance_to_meters(
                    distance_covered_pixels, 
                    cricket_constants.PITCH_LENGTH, 
                    mini_pitch.get_pitch_length()
                )
                
                ball_speed_kmh = distance_covered_meters / event_time_seconds * 3.6

                # Identify batsman and bowler
                player_positions = player_mini_pitch_detections[start_frame]
                
                # Find batsman (closest to striker's end)
                batsman_id = min(player_positions.keys(), 
                                key=lambda x: measure_distance_between_points(
                                    player_positions[x], 
                                    mini_pitch.get_striker_end_position()
                                ))
                
                # Calculate shot power and direction
                shot_power = min(100, distance_covered_meters * 10)  # Normalize to 0-100
                
                current_stats = deepcopy(cricket_stats_data[-1])
                current_stats["frame_num"] = start_frame
                current_stats["ball_speed"] = ball_speed_kmh
                current_stats["shot_power"] = shot_power
                
                # Add shot classification data
                if ENABLE_SHOT_CLASSIFICATION and start_frame in shot_classifications:
                    shot_info = shot_classifications[start_frame]
                    current_stats["shot_type"] = shot_info['shot_type']
                    current_stats["shot_direction"] = shot_info.get('direction', 'Straight')
                
                cricket_stats_data.append(current_stats)

        # Convert to DataFrame for easier processing
        cricket_stats_df = pd.DataFrame(cricket_stats_data)
        frames_df = pd.DataFrame({"frame_num": range(len(video_frames))})
        cricket_stats_df = pd.merge(frames_df, cricket_stats_df, on="frame_num", how="left")
        cricket_stats_df = cricket_stats_df.ffill()

        # Create output video
        print("Creating cricket analysis video...")
        output_video_frames = video_frames.copy()

        # Draw player tracking with cricket-specific roles
        print("Drawing cricket player tracking...")
        output_video_frames = player_tracker.draw_cricket_bboxes(output_video_frames, player_detections)
        
        # Draw ball tracking
        print("Drawing cricket ball tracking...")
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, color=(255, 255, 0))

        # Draw cricket statistics
        print("Drawing cricket statistics...")
        output_video_frames = draw_cricket_stats(output_video_frames, cricket_stats_df)

        # Draw pitch keypoints
        print("Drawing pitch boundaries...")
        output_video_frames = pitch_detector.draw_keypoints_on_video(
            output_video_frames, pitch_keypoints, point_color=(0, 255, 0), radius=6)

        # Draw mini cricket pitch
        print("Drawing mini cricket pitch...")
        output_video_frames = mini_pitch.draw_mini_cricket_pitch(output_video_frames)
        
        # Draw ball trajectory on mini pitch
        print("Drawing ball trajectory...")
        output_video_frames = mini_pitch.draw_ball_trajectory(output_video_frames, ball_mini_pitch_detections)
        
        # Draw player positions on mini pitch
        print("Drawing player positions on mini pitch...")
        output_video_frames = mini_pitch.draw_cricket_players_on_mini_pitch(
            output_video_frames, player_mini_pitch_detections)
        
        # Draw ball position on mini pitch
        output_video_frames = mini_pitch.draw_points_on_mini_pitch(
            output_video_frames, ball_mini_pitch_detections, color=(255, 255, 0))

        # Add frame information and cricket-specific overlays
        print("Adding cricket match information...")
        for i, frame in enumerate(output_video_frames):
            # Frame number
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Cricket event indicators
            if i in ball_event_frames:
                cv2.putText(frame, "CRICKET EVENT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add shot classification overlays
        if ENABLE_SHOT_CLASSIFICATION:
            print("Adding cricket shot classification overlays...")
            output_video_frames = draw_cricket_shot_classifications(
                output_video_frames, shot_classifications, ball_event_frames)

        # Save output video
        print("Saving cricket analysis video...")
        output_video_path = "output_videos/cricket_analysis.avi"
        
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        success = save_video(output_video_frames, output_video_path)
        
        if success:
            print(f"Cricket analysis complete! Video saved to {output_video_path}")
        else:
            print(f"ERROR: Failed to save video to {output_video_path}")
            
            # Try MP4 format
            output_video_path_mp4 = "output_videos/cricket_analysis.mp4"
            success_mp4 = save_video(output_video_frames, output_video_path_mp4)
            
            if success_mp4:
                print(f"Successfully saved as MP4: {output_video_path_mp4}")
            else:
                print("CRITICAL ERROR: All video saving attempts failed.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()