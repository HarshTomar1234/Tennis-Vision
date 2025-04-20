import numpy as np
from utils import measure_distance_between_points, measure_xy_distance
import pandas as pd

class ShotClassifier:
    """
    A professional shot classifier for tennis match analysis.
    Categorizes shots as serve, forehand, backhand, volley, or smash.
    Based on player position, ball trajectory, and timing.
    """
    
    def __init__(self):
        # Define shot types
        self.SHOT_TYPES = {
            'SERVE': 'Serve',
            'FOREHAND': 'Forehand',
            'BACKHAND': 'Backhand',
            'VOLLEY': 'Volley',
            'SMASH': 'Smash'
        }
        
        # Shot colors for visualization (BGR format)
        self.SHOT_COLORS = {
            self.SHOT_TYPES['SERVE']: (0, 165, 255),     # Orange
            self.SHOT_TYPES['FOREHAND']: (0, 255, 0),    # Green
            self.SHOT_TYPES['BACKHAND']: (255, 0, 0),    # Blue
            self.SHOT_TYPES['VOLLEY']: (255, 255, 0),    # Cyan
            self.SHOT_TYPES['SMASH']: (0, 0, 255)        # Red
        }
        
        # Shot classification thresholds
        self.VOLLEY_DISTANCE_THRESHOLD = 150  # Distance from net for volley detection
        self.SMASH_HEIGHT_THRESHOLD = 0.7     # Relative height threshold for smash detection
        self.NET_Y_POSITION_RELATIVE = 0.5    # Relative position of the net (middle of court)
        
    def classify_shots(self, player_mini_court_detections, ball_mini_court_detections, 
                      ball_shot_frames, mini_court_height):
        """
        Classify each shot in the tennis match.
        
        Args:
            player_mini_court_detections: Dictionary of player positions on mini court
            ball_mini_court_detections: Dictionary of ball positions on mini court
            ball_shot_frames: List of frame numbers where shots occur
            mini_court_height: Height of the mini court for relative positioning
            
        Returns:
            Dictionary mapping each shot frame to its classification and the player who made it
        """
        shot_classifications = {}
        
        # Skip if not enough shots
        if len(ball_shot_frames) <= 1:
            return shot_classifications
        
        # Classify each shot
        for i in range(len(ball_shot_frames)-1):
            shot_frame = ball_shot_frames[i]
            next_shot_frame = ball_shot_frames[i+1]
            
            # Get player who made the shot (closest to ball at shot frame)
            player_positions = player_mini_court_detections[shot_frame]
            if not player_positions or not ball_mini_court_detections.get(shot_frame, {}).get(1):
                continue
                
            ball_pos = ball_mini_court_detections[shot_frame][1]
            player_shot_id = min(player_positions.keys(), 
                               key=lambda x: measure_distance_between_points(player_positions[x], ball_pos))
            
            # Extract player and ball positions
            player_pos = player_positions[player_shot_id]
            player_y = player_pos[1]
            
            # Get ball trajectory
            if shot_frame in ball_mini_court_detections and next_shot_frame in ball_mini_court_detections:
                ball_start = ball_mini_court_detections[shot_frame][1]
                ball_end = ball_mini_court_detections[next_shot_frame][1]
                ball_trajectory_y = ball_end[1] - ball_start[1]
            else:
                ball_trajectory_y = 0
            
            # Detect shot type
            shot_type = self._determine_shot_type(
                i=i,
                player_id=player_shot_id,
                player_y=player_y,
                ball_trajectory_y=ball_trajectory_y,
                mini_court_height=mini_court_height,
                is_first_shot=(i == 0)
            )
            
            # Store classification
            shot_classifications[shot_frame] = {
                'shot_type': shot_type,
                'player_id': player_shot_id,
                'frame_index': i  # Store frame index to track progression
            }
            
        return shot_classifications
    
    def _determine_shot_type(self, i, player_id, player_y, ball_trajectory_y, mini_court_height, is_first_shot):
        """
        Determine the type of shot based on player position and ball trajectory.
        
        Args:
            i: Shot index
            player_id: ID of player making the shot
            player_y: Y-coordinate of player on mini court
            ball_trajectory_y: Vertical component of ball trajectory
            mini_court_height: Height of mini court for relative positioning
            is_first_shot: Whether this is the first shot in a rally
            
        Returns:
            Shot type classification
        """
        # Default shot types based on court position (top/bottom half)
        net_y = mini_court_height * self.NET_Y_POSITION_RELATIVE
        default_shot = self.SHOT_TYPES['FOREHAND']
        
        # First shot in sequence is always a serve
        if is_first_shot:
            return self.SHOT_TYPES['SERVE']
        
        # Check for volley (player close to net)
        volley_threshold = self.VOLLEY_DISTANCE_THRESHOLD
        if abs(player_y - net_y) < volley_threshold:
            return self.SHOT_TYPES['VOLLEY']
        
        # Check for smash (ball high, player hitting downward)
        if ball_trajectory_y > 0 and ball_trajectory_y > mini_court_height * self.SMASH_HEIGHT_THRESHOLD:
            return self.SHOT_TYPES['SMASH']
        
        # Determine forehand/backhand based on player position and ball trajectory
        # For player 1 (usually bottom of court)
        if player_id == 1:
            if player_y > net_y and ball_trajectory_y < 0:
                return self.SHOT_TYPES['BACKHAND']
            else:
                return self.SHOT_TYPES['FOREHAND']
        # For player 2 (usually top of court)
        else:
            if player_y < net_y and ball_trajectory_y > 0:
                return self.SHOT_TYPES['BACKHAND']
            else:
                return self.SHOT_TYPES['FOREHAND']
                
    def get_shot_color(self, shot_type):
        """Get the color associated with a shot type for visualization"""
        return self.SHOT_COLORS.get(shot_type, (255, 255, 255))  # Default to white


def draw_shot_classifications(frames, shot_classifications, ball_shot_frames):
    """
    Draw shot classification information in a dedicated shot statistics board.
    
    Args:
        frames: List of video frames to draw on
        shot_classifications: Dictionary of shot classifications by frame
        ball_shot_frames: List of frame numbers where shots occur
        
    Returns:
        Frames with shot statistics board
    """
    import cv2
    
    # Initialize shot classifier for color mapping
    shot_classifier = ShotClassifier()
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Process each frame
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]
        
        # Create player shot histories up to the current frame
        player_shots = {1: [], 2: []}
        max_shots_to_display = 3
        
        # Only include shots that have happened up to this frame
        for frame_num, shot_info in shot_classifications.items():
            if frame_num <= i:  # Only include shots up to the current frame
                player_id = shot_info['player_id']
                shot_type = shot_info['shot_type']
                
                # Add to player's shot history (newest first)
                player_shots[player_id].insert(0, {'frame': frame_num, 'type': shot_type})
                
                # Keep only the most recent shots
                if len(player_shots[player_id]) > max_shots_to_display:
                    player_shots[player_id] = player_shots[player_id][:max_shots_to_display]
        
        # Create shot statistics board - positioned at Vienna logo area
        board_width = 500
        board_height = 120
        board_x = width // 2 - board_width // 2  # Center of the court, where Vienna text is
        board_y = 480  # Where the Vienna text is located
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y), 
                     (board_x + board_width, board_y + board_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw board title
        title = "SHOT ANALYSIS"
        cv2.putText(frame, title, (board_x + 180, board_y + 25), 
                   font, 0.9, (255, 255, 255), thickness)
        
        # Draw header for the table
        cv2.line(frame, (board_x, board_y + 35), 
                (board_x + board_width, board_y + 35), (200, 200, 200), 1)
        
        # Column headers
        cv2.putText(frame, "Player", (board_x + 30, board_y + 55), 
                   font, font_scale, (200, 200, 200), 1)
        cv2.putText(frame, "Recent Shots", (board_x + 250, board_y + 55), 
                   font, font_scale, (200, 200, 200), 1)
        
        # Dividing line below headers
        cv2.line(frame, (board_x, board_y + 65), 
                (board_x + board_width, board_y + 65), (200, 200, 200), 1)
        
        # Draw players and their shots
        for row, player_id in enumerate([1, 2]):
            y_pos = board_y + 90 + (row * 30)
            
            # Player name
            player_text = f"Player {player_id}"
            cv2.putText(frame, player_text, (board_x + 30, y_pos), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Recent shots with colors
            shots = player_shots.get(player_id, [])
            
            if not shots:
                # If no shots yet, display N/A
                cv2.putText(frame, "N/A", (board_x + 250, y_pos),
                           font, font_scale, (150, 150, 150), 1)
            else:
                # Display actual shots
                for col, shot in enumerate(shots):
                    shot_type = shot['type']
                    shot_color = shot_classifier.get_shot_color(shot_type)
                    
                    # Shot bubble
                    bubble_x = board_x + 200 + (col * 100)
                    bubble_y = y_pos - 15
                    
                    # Draw filled circle behind text
                    cv2.circle(frame, (bubble_x + 40, bubble_y), 20, shot_color, -1)
                    cv2.circle(frame, (bubble_x + 40, bubble_y), 20, (255, 255, 255), 1)  # White outline
                    
                    # Draw shot text
                    cv2.putText(frame, shot_type, (bubble_x + 10, bubble_y + 5), 
                              font, 0.6, (0, 0, 0), thickness)
        
        # Show "SHOT!" indicator when a shot is detected
        if i in ball_shot_frames:
            # Get the shot info if available
            if i in shot_classifications:
                shot_info = shot_classifications[i]
                player_id = shot_info['player_id']
                shot_type = shot_info['shot_type']
                
                # Message and color
                shot_message = f"Player {player_id}: {shot_type.upper()}"
                shot_color = shot_classifier.get_shot_color(shot_type)
                
                # Draw attention-grabbing notification at the top of the screen
                notification_width = 350
                notification_x = (width - notification_width) // 2
                notification_y = 20
                
                # Background with player color
                cv2.rectangle(frame, 
                             (notification_x, notification_y), 
                             (notification_x + notification_width, notification_y + 40), 
                             shot_color, -1)
                cv2.rectangle(frame, 
                             (notification_x, notification_y), 
                             (notification_x + notification_width, notification_y + 40), 
                             (255, 255, 255), 2)  # White border
                
                # Shot text
                cv2.putText(frame, shot_message, 
                           (notification_x + 20, notification_y + 28), 
                           font, 0.9, (0, 0, 0), thickness)
                
    return frames 