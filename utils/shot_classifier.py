import numpy as np
from utils import measure_distance_between_points, measure_xy_distance
import pandas as pd
from typing import Dict, List, Optional, Any

class ShotClassifier:
    """
    A professional shot classifier for tennis match analysis.
    Categorizes shots as serve, forehand, backhand, volley, or smash.
    Based on player position, ball trajectory, and optionally pose data.
    
    Version 2.0: Supports pose-aware classification for improved accuracy.
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
        
        # Shot classification thresholds (position-based)
        self.VOLLEY_DISTANCE_THRESHOLD = 150  # Distance from net for volley detection
        self.SMASH_HEIGHT_THRESHOLD = 0.7     # Relative height threshold for smash detection
        self.NET_Y_POSITION_RELATIVE = 0.5    # Relative position of the net (middle of court)
        
        # Pose-aware thresholds (NEW in v2.0)
        self.ARM_OVERHEAD_THRESHOLD = 0.2     # Wrist above head threshold (relative to body height)
        self.ARM_EXTENDED_ANGLE_MIN = 140     # Arm angle for extended position (degrees)
        self.BACKHAND_CROSS_BODY_RATIO = 0.3  # Wrist crosses body center ratio
        
        # Keypoint indices (COCO format)
        self.KPT_NOSE = 0
        self.KPT_LEFT_SHOULDER = 5
        self.KPT_RIGHT_SHOULDER = 6
        self.KPT_LEFT_ELBOW = 7
        self.KPT_RIGHT_ELBOW = 8
        self.KPT_LEFT_WRIST = 9
        self.KPT_RIGHT_WRIST = 10
        self.KPT_LEFT_HIP = 11
        self.KPT_RIGHT_HIP = 12
        
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
    
    def classify_shots_with_pose(
        self,
        player_mini_court_detections: Dict[int, Dict[int, Any]],
        ball_mini_court_detections: Dict[int, Dict[int, Any]],
        ball_shot_frames: List[int],
        mini_court_height: float,
        player_poses: Optional[List[Dict[int, np.ndarray]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Classify shots using both position and pose data for improved accuracy.
        
        This method enhances the basic position-based classification by analyzing
        player pose keypoints to detect:
        - Serve/Smash: Arm in overhead position
        - Backhand: Wrist crosses body center
        - Forehand: Arm extended on dominant side
        
        Args:
            player_mini_court_detections: Player positions on mini court
            ball_mini_court_detections: Ball positions on mini court
            ball_shot_frames: List of frame numbers where shots occur
            mini_court_height: Height of mini court for relative positioning
            player_poses: Optional pose data from PoseTracker
            
        Returns:
            Dictionary mapping shot frame to classification with confidence
        """
        # If no pose data, fall back to position-based classification
        if player_poses is None:
            return self.classify_shots(
                player_mini_court_detections,
                ball_mini_court_detections,
                ball_shot_frames,
                mini_court_height
            )
        
        shot_classifications = {}
        
        if len(ball_shot_frames) <= 1:
            return shot_classifications
        
        for i in range(len(ball_shot_frames) - 1):
            shot_frame = ball_shot_frames[i]
            next_shot_frame = ball_shot_frames[i + 1]
            
            # Get player who made the shot
            player_positions = player_mini_court_detections.get(shot_frame, {})
            ball_data = ball_mini_court_detections.get(shot_frame, {})
            
            if not player_positions or not ball_data.get(1):
                continue
            
            ball_pos = ball_data[1]
            player_shot_id = min(
                player_positions.keys(),
                key=lambda x: measure_distance_between_points(player_positions[x], ball_pos)
            )
            
            player_pos = player_positions[player_shot_id]
            player_y = player_pos[1]
            
            # Get ball trajectory
            ball_trajectory_y = 0
            if shot_frame in ball_mini_court_detections and next_shot_frame in ball_mini_court_detections:
                ball_start = ball_mini_court_detections[shot_frame].get(1)
                ball_end = ball_mini_court_detections[next_shot_frame].get(1)
                if ball_start and ball_end:
                    ball_trajectory_y = ball_end[1] - ball_start[1]
            
            # Get pose data if available
            player_keypoints = None
            if shot_frame < len(player_poses) and player_shot_id in player_poses[shot_frame]:
                player_keypoints = player_poses[shot_frame][player_shot_id]
            
            # Classify with pose enhancement
            shot_type, confidence, method = self._determine_shot_type_with_pose(
                i=i,
                player_id=player_shot_id,
                player_y=player_y,
                ball_trajectory_y=ball_trajectory_y,
                mini_court_height=mini_court_height,
                is_first_shot=(i == 0),
                keypoints=player_keypoints
            )
            
            shot_classifications[shot_frame] = {
                'shot_type': shot_type,
                'player_id': player_shot_id,
                'frame_index': i,
                'confidence': confidence,
                'method': method  # 'pose' or 'position'
            }
        
        return shot_classifications
    
    def _determine_shot_type_with_pose(
        self,
        i: int,
        player_id: int,
        player_y: float,
        ball_trajectory_y: float,
        mini_court_height: float,
        is_first_shot: bool,
        keypoints: Optional[np.ndarray] = None,
    ):
        """
        Determine shot type using pose data when available.
        
        Returns:
            Tuple of (shot_type, confidence, method)
        """
        net_y = mini_court_height * self.NET_Y_POSITION_RELATIVE
        
        # First shot is always a serve
        if is_first_shot:
            return self.SHOT_TYPES['SERVE'], 0.95, 'position'
        
        # If we have pose data, use it for enhanced classification
        if keypoints is not None and self._has_valid_keypoints(keypoints):
            pose_result = self._analyze_pose_for_shot(keypoints)
            
            if pose_result['arm_overhead']:
                # Overhead arm = serve or smash
                if is_first_shot:
                    return self.SHOT_TYPES['SERVE'], 0.98, 'pose'
                else:
                    return self.SHOT_TYPES['SMASH'], 0.92, 'pose'
            
            if pose_result['backhand_detected']:
                return self.SHOT_TYPES['BACKHAND'], 0.88, 'pose'
            
            if pose_result['forehand_detected']:
                return self.SHOT_TYPES['FOREHAND'], 0.88, 'pose'
        
        # Fall back to position-based classification
        # Check for volley
        if abs(player_y - net_y) < self.VOLLEY_DISTANCE_THRESHOLD:
            return self.SHOT_TYPES['VOLLEY'], 0.85, 'position'
        
        # Check for smash
        if ball_trajectory_y > 0 and ball_trajectory_y > mini_court_height * self.SMASH_HEIGHT_THRESHOLD:
            return self.SHOT_TYPES['SMASH'], 0.80, 'position'
        
        # Determine forehand/backhand based on position
        if player_id == 1:
            if player_y > net_y and ball_trajectory_y < 0:
                return self.SHOT_TYPES['BACKHAND'], 0.75, 'position'
            else:
                return self.SHOT_TYPES['FOREHAND'], 0.75, 'position'
        else:
            if player_y < net_y and ball_trajectory_y > 0:
                return self.SHOT_TYPES['BACKHAND'], 0.75, 'position'
            else:
                return self.SHOT_TYPES['FOREHAND'], 0.75, 'position'
    
    def _has_valid_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.3) -> bool:
        """Check if keypoints have sufficient confidence for analysis."""
        # Need at least shoulders and one wrist
        required_indices = [
            self.KPT_LEFT_SHOULDER, self.KPT_RIGHT_SHOULDER,
        ]
        optional_indices = [
            self.KPT_LEFT_WRIST, self.KPT_RIGHT_WRIST
        ]
        
        # All required must be visible
        for idx in required_indices:
            if keypoints[idx][2] < min_confidence:
                return False
        
        # At least one wrist must be visible
        wrist_visible = any(keypoints[idx][2] >= min_confidence for idx in optional_indices)
        return wrist_visible
    
    def _analyze_pose_for_shot(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Analyze pose keypoints to determine shot characteristics.
        
        Returns:
            Dictionary with pose analysis results
        """
        result = {
            'arm_overhead': False,
            'backhand_detected': False,
            'forehand_detected': False,
            'dominant_arm': None,
            'arm_extension': 0.0,
        }
        
        # Get body landmarks
        nose = keypoints[self.KPT_NOSE][:2]
        left_shoulder = keypoints[self.KPT_LEFT_SHOULDER][:2]
        right_shoulder = keypoints[self.KPT_RIGHT_SHOULDER][:2]
        left_wrist = keypoints[self.KPT_LEFT_WRIST]
        right_wrist = keypoints[self.KPT_RIGHT_WRIST]
        left_hip = keypoints[self.KPT_LEFT_HIP][:2] if keypoints[self.KPT_LEFT_HIP][2] > 0.3 else None
        right_hip = keypoints[self.KPT_RIGHT_HIP][:2] if keypoints[self.KPT_RIGHT_HIP][2] > 0.3 else None
        
        # Calculate body center
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Determine which wrist is more active (higher confidence = more visible)
        left_conf = left_wrist[2]
        right_conf = right_wrist[2]
        
        if right_conf > left_conf:
            active_wrist = right_wrist[:2]
            active_wrist_conf = right_conf
            result['dominant_arm'] = 'right'
        else:
            active_wrist = left_wrist[:2]
            active_wrist_conf = left_conf
            result['dominant_arm'] = 'left'
        
        if active_wrist_conf < 0.3:
            return result  # Not enough confidence
        
        # Check for overhead position (serve/smash)
        # Wrist Y is above (smaller than) nose Y
        if active_wrist[1] < nose[1]:
            result['arm_overhead'] = True
            return result  # Overhead takes priority
        
        # Check for backhand (wrist crosses body center)
        # For right-handed: left wrist on right side of body
        # For left-handed: right wrist on left side of body
        cross_body_threshold = shoulder_width * self.BACKHAND_CROSS_BODY_RATIO
        
        if result['dominant_arm'] == 'right':
            # Check if right wrist is significantly to the left of body center
            if active_wrist[0] < body_center_x - cross_body_threshold:
                result['backhand_detected'] = True
            else:
                result['forehand_detected'] = True
        else:
            # Check if left wrist is significantly to the right of body center
            if active_wrist[0] > body_center_x + cross_body_threshold:
                result['backhand_detected'] = True
            else:
                result['forehand_detected'] = True
        
        return result


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
    font_scale = 0.6
    thickness = 1
    
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
        
        # Create shot statistics board - positioned at TOP LEFT (not bottom!)
        # CAMERA-ROBUST: Move to top-left to avoid overlap with Player Stats
        board_width = max(350, int(width * 0.32))  # Slightly smaller for top-left
        board_height = 120  # Compact height
        board_x = 10  # Left edge with small padding
        board_y = height - 200  # Position above player stats area
        
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
        legend_width = 150
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
        cv2.putText(frame, "SHOT TYPE LEGEND", (legend_x + 35, legend_y + 25), 
                   font, 0.65, (255, 255, 255), thickness)
        
        # Add each shot type with its color
        shot_types = [("SM", "Smash", shot_classifier.get_shot_color("smash")),
                     ("BH", "Backhand", shot_classifier.get_shot_color("backhand")), 
                     ("FH", "Forehand", shot_classifier.get_shot_color("forehand")),
                     ("SE", "Serve", shot_classifier.get_shot_color("serve")),
                     ("VO", "Volley", shot_classifier.get_shot_color("volley")),]
        
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
                notification_width = 300
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
                           font, 0.8, (0, 0, 0), thickness+1)
    
    return frames 