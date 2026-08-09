import numpy as np
from utils import measure_distance_between_points, measure_xy_distance
import pandas as pd

class ShotClassifier:
    """
    A professional shot classifier for tennis match analysis.
    Categorizes shots as serve, forehand, backhand, volley, or smash.
    Based on player position, ball trajectory, and timing.
    """
    
    def __init__(self, volley_threshold=40, smash_height_threshold=0.7, net_y_relative=0.5):
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

        self.VOLLEY_DISTANCE_THRESHOLD = volley_threshold        # px from net (default 40)
        self.SMASH_HEIGHT_THRESHOLD = smash_height_threshold
        self.NET_Y_POSITION_RELATIVE = net_y_relative
        
    def classify_shots(self, player_mini_court_detections, ball_mini_court_detections,
                      ball_shot_frames, mini_court_height, serve_frames=None):
        """
        Classify each shot in the tennis match.

        Args:
            player_mini_court_detections: Dictionary of player positions on mini court
            ball_mini_court_detections: Dictionary of ball positions on mini court
            ball_shot_frames: List of frame numbers where shots occur
            mini_court_height: Height of the mini court for relative positioning
            serve_frames: Frames carrying physical serve evidence (see
                utils/serve_detector.py). When omitted, no shot is labelled a serve —
                deliberately, because the alternative was labelling whichever shot
                came first, which is wrong on any clip that starts mid-point.

        Returns:
            Dictionary mapping each shot frame to its classification and the player who made it
        """
        serve_frames = set(serve_frames or ())
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
                is_first_shot=(shot_frame in serve_frames)
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

        # `is_first_shot` now means "carries serve evidence" (ball struck above the
        # player's head, from a baseline) rather than "happens to be first in the
        # clip". The old positional rule labelled a mid-rally groundstroke a serve
        # whenever a clip started mid-point, which is most of the time — see
        # utils/serve_detector.py.
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
        """Get color for a shot type. Case-insensitive lookup."""
        color = self.SHOT_COLORS.get(shot_type)
        if color is None:
            color = self.SHOT_COLORS.get(shot_type.title())
        return color or (255, 255, 255)


def draw_shot_classifications(frames, shot_classifications, ball_shot_frames,
                              display_delay: int = 3, banner_duration: int = 15):
    """
    Draw shot classification information in a dedicated shot statistics board.

    display_delay: frames after the y-reversal before the SHOT ANALYSIS panel updates.
                   Avoids showing the label before the racket contacts the ball.
    banner_duration: how many frames the top-center shot announcement banner stays
                     on screen (default 15 ≈ 0.5 s at 30 fps).
    """
    import cv2

    shot_classifier = ShotClassifier()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]

        # Build shot history for each player up to this frame,
        # applying display_delay so the type appears after actual contact.
        player_shots = {1: [], 2: []}
        max_shots_to_display = 3

        for frame_num, shot_info in shot_classifications.items():
            if frame_num + display_delay <= i:
                player_id = shot_info['player_id']
                shot_type = shot_info['shot_type']
                
                # Add to player's shot history (newest first)
                player_shots[player_id].insert(0, {'frame': frame_num, 'type': shot_type})
                
                # Keep only the most recent shots
                if len(player_shots[player_id]) > max_shots_to_display:
                    player_shots[player_id] = player_shots[player_id][:max_shots_to_display]
        
        # Create shot statistics board - positioned at TOP LEFT (not bottom!)
        # CAMERA-ROBUST: Move to top-left to avoid overlap with Player Stats
        board_width = max(350, int(width * 0.32))  # Slightly smaller for top-left
        board_height = 120  # Compact height
        board_x = 10  # Left edge with small padding
        # Top-left: the court's far baseline sits well below this in a broadcast
        # frame, and the player-stats panel now owns the bottom-left corner.
        board_y = 45
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y), 
                     (board_x + board_width, board_y + board_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw board title
        cv2.rectangle(frame, (board_x, board_y), 
                     (board_x + board_width, board_y + 35), 
                     (40, 40, 100), -1)
        cv2.putText(frame, "SHOT ANALYSIS", (board_x + 180, board_y + 25), 
                   font, 0.8, (255, 255, 255), thickness)
        
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
            cv2.putText(frame, player_text, (board_x + 20, y_pos), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Recent shots with colors (smaller balls)
            shots = player_shots.get(player_id, [])
            
            if not shots:
                # If no shots yet, display N/A
                cv2.putText(frame, "N/A", (board_x + 150, y_pos),
                           font, font_scale, (150, 150, 150), 1)
            else:
                # Display smaller shot indicators - adjust spacing to fit in board
                for col, shot in enumerate(shots):
                    shot_type = shot['type']
                    shot_color = shot_classifier.get_shot_color(shot_type)
                    
                    # Shot bubble - reduced spacing (50px instead of 80px)
                    bubble_radius = 14
                    bubble_x = board_x + 150 + (col * 55)  # Start at 150, space by 55
                    bubble_y = y_pos - 5
                    
                    # Ensure bubble stays within board bounds
                    if bubble_x + bubble_radius < board_x + board_width - 10:
                        # Draw filled circle behind text
                        cv2.circle(frame, (bubble_x, bubble_y), bubble_radius, shot_color, -1)
                        cv2.circle(frame, (bubble_x, bubble_y), bubble_radius, (255, 255, 255), 1)
                        
                        # Draw abbreviated shot text
                        short_text = shot_type[:2].upper()
                        text_size = cv2.getTextSize(short_text, font, font_scale-0.1, thickness)[0]
                        text_x = bubble_x - text_size[0]//2
                        text_y = bubble_y + text_size[1]//2
                        cv2.putText(frame, short_text, (text_x, text_y), 
                                  font, font_scale-0.1, (0, 0, 0), thickness)
        
        # Add a legend for shot types at the bottom right 
        # CAMERA-ROBUST: Position to avoid overlapping with mini court and show all items
        # Size the panel to its own title rather than a guessed constant — at 150px
        # the title overflowed the box and was clipped by the frame edge.
        legend_title = "SHOT TYPE LEGEND"
        (title_w, _), _ = cv2.getTextSize(legend_title, font, 0.65, thickness)
        legend_width = max(150, title_w + 30)
        legend_height = 165  # Increased to fit 5 shot types
        legend_x = width - legend_width - 10  # Right edge
        legend_y = height - legend_height - 50  # Move up to avoid cutoff
        
        # Draw semi-transparent background for legend
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add legend title
        cv2.putText(frame, legend_title,
                   (legend_x + (legend_width - title_w) // 2, legend_y + 25),
                   font, 0.65, (255, 255, 255), thickness)
        
        # Add each shot type with its color
        shot_types = [("SM", "Smash",    shot_classifier.get_shot_color("Smash")),
                      ("BH", "Backhand", shot_classifier.get_shot_color("Backhand")),
                      ("FH", "Forehand", shot_classifier.get_shot_color("Forehand")),
                      ("SE", "Serve",    shot_classifier.get_shot_color("Serve")),
                      ("VO", "Volley",   shot_classifier.get_shot_color("Volley"))]
        
        for idx, (abbr, name, color) in enumerate(shot_types):
            y_offset = legend_y + 55 + idx * 25
            
            # Draw color indicator
            circle_x = legend_x + 20
            cv2.circle(frame, (circle_x, y_offset - 5), 10, color, -1)
            cv2.circle(frame, (circle_x, y_offset - 5), 10, (255, 255, 255), 1)
            
            # Draw abbreviation in circle
            text_size = cv2.getTextSize(abbr, font, font_scale-0.2, thickness)[0]
            text_x = circle_x - text_size[0]//2
            text_y = y_offset - 5 + text_size[1]//2
            cv2.putText(frame, abbr, (text_x, text_y), 
                       font, font_scale-0.2, (0, 0, 0), thickness)
            
            # Draw full name
            cv2.putText(frame, name, (legend_x + 40, y_offset), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Show shot announcement banner for `banner_duration` frames after detection
        # (delayed by display_delay so it coincides with visible racket contact)
        active_banner = None
        for sf in ball_shot_frames:
            show_start = sf + display_delay
            show_end   = sf + display_delay + banner_duration
            if show_start <= i < show_end and sf in shot_classifications:
                active_banner = shot_classifications[sf]
                break

        if active_banner is not None:
            player_id = active_banner['player_id']
            shot_type = active_banner['shot_type']
            shot_message = f"Player {player_id}: {shot_type.upper()}"
            shot_color = shot_classifier.get_shot_color(shot_type)

            notification_width = 300
            notification_x = (width - notification_width) // 2
            notification_y = 20

            cv2.rectangle(frame,
                         (notification_x, notification_y),
                         (notification_x + notification_width, notification_y + 40),
                         shot_color, -1)
            cv2.rectangle(frame,
                         (notification_x, notification_y),
                         (notification_x + notification_width, notification_y + 40),
                         (255, 255, 255), 2)
            cv2.putText(frame, shot_message,
                       (notification_x + 20, notification_y + 28),
                       font, 0.8, (0, 0, 0), thickness + 1)
    
    return frames 