import numpy as np
from utils import measure_distance_between_points, measure_xy_distance
import pandas as pd

class CricketShotClassifier:
    """
    Professional cricket shot classifier for match analysis.
    Categorizes shots as defensive, drive, cut, pull, sweep, hook, flick, loft, etc.
    """
    
    def __init__(self):
        # Define cricket shot types
        self.SHOT_TYPES = {
            'DEFENSIVE': 'Defensive',
            'DRIVE': 'Drive',
            'CUT': 'Cut', 
            'PULL': 'Pull',
            'SWEEP': 'Sweep',
            'HOOK': 'Hook',
            'FLICK': 'Flick',
            'LOFT': 'Loft',
            'SIX': 'Six',
            'FOUR': 'Four',
            'SINGLE': 'Single',
            'DOT': 'Dot Ball'
        }
        
        # Shot colors for visualization (BGR format)
        self.SHOT_COLORS = {
            self.SHOT_TYPES['DEFENSIVE']: (128, 128, 128),  # Gray
            self.SHOT_TYPES['DRIVE']: (0, 255, 0),          # Green
            self.SHOT_TYPES['CUT']: (255, 0, 0),            # Blue
            self.SHOT_TYPES['PULL']: (0, 165, 255),         # Orange
            self.SHOT_TYPES['SWEEP']: (255, 255, 0),        # Cyan
            self.SHOT_TYPES['HOOK']: (128, 0, 128),         # Purple
            self.SHOT_TYPES['FLICK']: (0, 255, 255),        # Yellow
            self.SHOT_TYPES['LOFT']: (255, 0, 255),         # Magenta
            self.SHOT_TYPES['SIX']: (0, 0, 255),            # Red
            self.SHOT_TYPES['FOUR']: (0, 255, 0),           # Green
            self.SHOT_TYPES['SINGLE']: (255, 255, 255),     # White
            self.SHOT_TYPES['DOT']: (64, 64, 64)            # Dark Gray
        }
        
        # Cricket-specific thresholds
        self.BOUNDARY_DISTANCE_THRESHOLD = 200  # Distance for boundary shots
        self.AGGRESSIVE_SHOT_THRESHOLD = 150    # Distance for aggressive shots
        self.DEFENSIVE_SHOT_THRESHOLD = 50      # Distance for defensive shots
        
        # Shot direction zones
        self.SHOT_DIRECTIONS = {
            'STRAIGHT': 'Straight',
            'OFF_SIDE': 'Off Side',
            'LEG_SIDE': 'Leg Side', 
            'BEHIND': 'Behind Wicket'
        }

    def classify_cricket_shots(self, player_mini_pitch_detections, ball_mini_pitch_detections, 
                              ball_event_frames, pitch_length):
        """
        Classify each cricket shot in the match.
        
        Args:
            player_mini_pitch_detections: Dictionary of player positions on mini pitch
            ball_mini_pitch_detections: Dictionary of ball positions on mini pitch
            ball_event_frames: List of frame numbers where cricket events occur
            pitch_length: Length of the mini pitch for relative positioning
            
        Returns:
            Dictionary mapping each event frame to its classification
        """
        shot_classifications = {}
        
        if len(ball_event_frames) <= 1:
            return shot_classifications
        
        # Classify each cricket event
        for i in range(len(ball_event_frames)-1):
            event_frame = ball_event_frames[i]
            next_event_frame = ball_event_frames[i+1]
            
            # Get batsman (player closest to striker's end)
            player_positions = player_mini_pitch_detections[event_frame]
            if not player_positions or not ball_mini_pitch_detections.get(event_frame, {}).get(1):
                continue
                
            ball_pos = ball_mini_pitch_detections[event_frame][1]
            
            # Find batsman (assuming player 1 is batsman for simplicity)
            batsman_id = 1 if 1 in player_positions else min(player_positions.keys())
            batsman_pos = player_positions[batsman_id]
            
            # Calculate ball trajectory and distance
            if event_frame in ball_mini_pitch_detections and next_event_frame in ball_mini_pitch_detections:
                ball_start = ball_mini_pitch_detections[event_frame][1]
                ball_end = ball_mini_pitch_detections[next_event_frame][1]
                
                ball_distance = measure_distance_between_points(ball_start, ball_end)
                ball_trajectory_x = ball_end[0] - ball_start[0]
                ball_trajectory_y = ball_end[1] - ball_start[1]
            else:
                ball_distance = 0
                ball_trajectory_x = 0
                ball_trajectory_y = 0
            
            # Determine shot type and direction
            shot_type, shot_direction = self._determine_cricket_shot_type(
                batsman_pos=batsman_pos,
                ball_start_pos=ball_pos,
                ball_distance=ball_distance,
                ball_trajectory_x=ball_trajectory_x,
                ball_trajectory_y=ball_trajectory_y,
                pitch_length=pitch_length,
                event_index=i
            )
            
            # Store classification
            shot_classifications[event_frame] = {
                'shot_type': shot_type,
                'direction': shot_direction,
                'batsman_id': batsman_id,
                'distance': ball_distance,
                'event_index': i
            }
            
        return shot_classifications

    def _determine_cricket_shot_type(self, batsman_pos, ball_start_pos, ball_distance, 
                                   ball_trajectory_x, ball_trajectory_y, pitch_length, event_index):
        """
        Determine cricket shot type based on ball movement and batsman position
        """
        # Determine shot direction first
        direction = self._determine_shot_direction(ball_trajectory_x, ball_trajectory_y)
        
        # Classify shot type based on distance and trajectory
        if ball_distance < self.DEFENSIVE_SHOT_THRESHOLD:
            return self.SHOT_TYPES['DEFENSIVE'], direction
        elif ball_distance > self.BOUNDARY_DISTANCE_THRESHOLD:
            # Check if it's a six or four based on trajectory
            if abs(ball_trajectory_y) > abs(ball_trajectory_x) and ball_trajectory_y > 0:
                return self.SHOT_TYPES['SIX'], direction
            else:
                return self.SHOT_TYPES['FOUR'], direction
        elif ball_distance > self.AGGRESSIVE_SHOT_THRESHOLD:
            # Determine aggressive shot type based on direction and trajectory
            if direction == self.SHOT_DIRECTIONS['STRAIGHT']:
                if ball_trajectory_y > 0:
                    return self.SHOT_TYPES['LOFT'], direction
                else:
                    return self.SHOT_TYPES['DRIVE'], direction
            elif direction == self.SHOT_DIRECTIONS['OFF_SIDE']:
                if ball_trajectory_y < 0:
                    return self.SHOT_TYPES['CUT'], direction
                else:
                    return self.SHOT_TYPES['DRIVE'], direction
            elif direction == self.SHOT_DIRECTIONS['LEG_SIDE']:
                if abs(ball_trajectory_x) > abs(ball_trajectory_y):
                    return self.SHOT_TYPES['PULL'], direction
                else:
                    return self.SHOT_TYPES['FLICK'], direction
            else:  # Behind wicket
                if ball_trajectory_y < 0:
                    return self.SHOT_TYPES['HOOK'], direction
                else:
                    return self.SHOT_TYPES['SWEEP'], direction
        else:
            # Medium distance shots
            if ball_distance > 75:
                return self.SHOT_TYPES['SINGLE'], direction
            else:
                return self.SHOT_TYPES['DOT'], direction

    def _determine_shot_direction(self, trajectory_x, trajectory_y):
        """Determine the direction of the cricket shot"""
        if abs(trajectory_x) < 20:  # Straight shot
            return self.SHOT_DIRECTIONS['STRAIGHT']
        elif trajectory_x > 0:  # Right side (off side for right-handed batsman)
            return self.SHOT_DIRECTIONS['OFF_SIDE']
        else:  # Left side (leg side for right-handed batsman)
            return self.SHOT_DIRECTIONS['LEG_SIDE']

    def get_shot_color(self, shot_type):
        """Get the color associated with a shot type for visualization"""
        return self.SHOT_COLORS.get(shot_type, (255, 255, 255))


def draw_cricket_shot_classifications(frames, shot_classifications, ball_event_frames):
    """
    Draw cricket shot classification information with detailed statistics
    """
    import cv2
    
    shot_classifier = CricketShotClassifier()
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]
        
        # Create shot history up to current frame
        shot_history = []
        runs_scored = 0
        balls_faced = 0
        
        for frame_num, shot_info in shot_classifications.items():
            if frame_num <= i:
                shot_history.append(shot_info)
                balls_faced += 1
                
                # Calculate runs based on shot type
                shot_type = shot_info['shot_type']
                if shot_type == 'Six':
                    runs_scored += 6
                elif shot_type == 'Four':
                    runs_scored += 4
                elif shot_type == 'Single':
                    runs_scored += 1
                # Defensive and dot balls = 0 runs
        
        # Calculate strike rate
        strike_rate = (runs_scored / max(balls_faced, 1)) * 100
        
        # Create cricket statistics board
        board_width = 550
        board_height = 200
        board_x = 20
        board_y = 400
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y), 
                     (board_x + board_width, board_y + board_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw board title
        cv2.rectangle(frame, (board_x, board_y), 
                     (board_x + board_width, board_y + 35), 
                     (0, 100, 0), -1)  # Dark green header
        cv2.putText(frame, "CRICKET SHOT ANALYSIS", (board_x + 150, board_y + 25), 
                   font, 0.8, (255, 255, 255), thickness)
        
        # Display current statistics
        stats_y = board_y + 60
        cv2.putText(frame, f"Runs: {runs_scored}", (board_x + 30, stats_y), 
                   font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"Balls: {balls_faced}", (board_x + 150, stats_y), 
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, f"Strike Rate: {strike_rate:.1f}", (board_x + 270, stats_y), 
                   font, font_scale, (255, 255, 0), thickness)
        
        # Show recent shots (last 6 balls)
        recent_shots = shot_history[-6:] if len(shot_history) > 6 else shot_history
        
        cv2.putText(frame, "Recent Shots:", (board_x + 30, board_y + 90), 
                   font, font_scale, (200, 200, 200), thickness)
        
        for idx, shot in enumerate(recent_shots):
            shot_type = shot['shot_type']
            shot_color = shot_classifier.get_shot_color(shot_type)
            
            # Shot indicator
            shot_x = board_x + 30 + (idx * 80)
            shot_y = board_y + 120
            
            # Draw shot circle
            cv2.circle(frame, (shot_x, shot_y), 20, shot_color, -1)
            cv2.circle(frame, (shot_x, shot_y), 20, (255, 255, 255), 2)
            
            # Shot abbreviation
            if shot_type == 'Six':
                abbr = '6'
            elif shot_type == 'Four':
                abbr = '4'
            elif shot_type == 'Single':
                abbr = '1'
            elif shot_type == 'Dot Ball':
                abbr = '•'
            else:
                abbr = shot_type[:2].upper()
            
            text_size = cv2.getTextSize(abbr, font, font_scale-0.1, thickness)[0]
            text_x = shot_x - text_size[0]//2
            text_y = shot_y + text_size[1]//2
            cv2.putText(frame, abbr, (text_x, text_y), 
                       font, font_scale-0.1, (0, 0, 0), thickness)
        
        # Shot type legend
        legend_x = width - 280
        legend_y = height - 220
        legend_width = 260
        legend_height = 200
        
        # Legend background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Legend title
        cv2.putText(frame, "SHOT TYPES", (legend_x + 80, legend_y + 25), 
                   font, 0.7, (255, 255, 255), thickness)
        
        # Legend items
        legend_items = [
            ("6", "Six", shot_classifier.get_shot_color("Six")),
            ("4", "Four", shot_classifier.get_shot_color("Four")),
            ("1", "Single", shot_classifier.get_shot_color("Single")),
            ("•", "Dot Ball", shot_classifier.get_shot_color("Dot Ball")),
            ("DR", "Drive", shot_classifier.get_shot_color("Drive")),
            ("CU", "Cut", shot_classifier.get_shot_color("Cut")),
            ("PU", "Pull", shot_classifier.get_shot_color("Pull")),
            ("DE", "Defensive", shot_classifier.get_shot_color("Defensive"))
        ]
        
        for idx, (abbr, name, color) in enumerate(legend_items):
            row = idx % 4
            col = idx // 4
            
            item_x = legend_x + 20 + (col * 120)
            item_y = legend_y + 50 + (row * 30)
            
            # Color indicator
            cv2.circle(frame, (item_x, item_y), 8, color, -1)
            cv2.circle(frame, (item_x, item_y), 8, (255, 255, 255), 1)
            
            # Abbreviation
            cv2.putText(frame, abbr, (item_x-5, item_y+3), 
                       font, 0.4, (0, 0, 0), 1)
            
            # Name
            cv2.putText(frame, name, (item_x + 15, item_y + 5), 
                       font, 0.5, (255, 255, 255), 1)
        
        # Show current shot notification
        if i in ball_event_frames and i in shot_classifications:
            shot_info = shot_classifications[i]
            shot_type = shot_info['shot_type']
            direction = shot_info['direction']
            
            # Notification at top
            notification_width = 400
            notification_x = (width - notification_width) // 2
            notification_y = 20
            
            shot_color = shot_classifier.get_shot_color(shot_type)
            
            # Background
            cv2.rectangle(frame, 
                         (notification_x, notification_y), 
                         (notification_x + notification_width, notification_y + 50), 
                         shot_color, -1)
            cv2.rectangle(frame, 
                         (notification_x, notification_y), 
                         (notification_x + notification_width, notification_y + 50), 
                         (255, 255, 255), 3)
            
            # Text
            shot_text = f"{shot_type.upper()} - {direction}"
            cv2.putText(frame, shot_text, 
                       (notification_x + 20, notification_y + 32), 
                       font, 0.9, (0, 0, 0), thickness+1)
    
    return frames


def draw_cricket_stats(output_video_frames, cricket_stats):
    """Draw cricket-specific statistics on video frames"""
    import cv2
    
    for index, row in cricket_stats.iterrows():
        ball_speed = row.get('ball_speed', 0)
        shot_power = row.get('shot_power', 0)
        shot_type = row.get('shot_type', 'N/A')
        shot_direction = row.get('shot_direction', 'N/A')

        frame = output_video_frames[index]
        
        # Cricket stats panel
        width = 400
        height = 180
        start_x = frame.shape[1]//2 - width//2
        start_y = 500
        end_x = start_x + width
        end_y = start_y + height

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Header
        cv2.rectangle(frame, (start_x, start_y), (end_x, start_y + 40), (0, 100, 0), -1)
        cv2.putText(frame, "CRICKET MATCH STATS", (start_x + 100, start_y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stats
        y_pos = start_y + 70
        cv2.putText(frame, f"Ball Speed: {ball_speed:.1f} km/h", (start_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(frame, f"Shot Power: {shot_power:.1f}%", (start_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(frame, f"Last Shot: {shot_type}", (start_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(frame, f"Direction: {shot_direction}", (start_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        output_video_frames[index] = frame
    
    return output_video_frames