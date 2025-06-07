import cv2
import sys
import numpy as np
import random
sys.path.append("../")
import cricket_constants 
from utils import (convert_meters_to_pixel_distance, convert_pixel_distance_to_meters, 
                   get_foot_position, get_closest_keypoint_index, get_height_of_bbox, 
                   measure_xy_distance, get_center_of_bbox, measure_distance_between_points)

class MiniCricketPitch:
    def __init__(self, frame, mini_pitch_width=None, mini_pitch_height=None):
        """
        Initialize mini cricket pitch with enhanced styling and dimensions
        """
        frame_height, frame_width = frame.shape[:2]
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Set pitch dimensions for coordinate calculations
        self.pitch_width = frame_width
        self.pitch_height = frame_height
        
        # Mini pitch dimensions (cricket pitch is more oval/circular)
        self.mini_pitch_width = mini_pitch_width if mini_pitch_width else int(frame_width * 0.25)
        self.mini_pitch_height = mini_pitch_height if mini_pitch_height else int(self.mini_pitch_width * 1.2)
        
        self.drawing_rectangle_width = 300
        self.drawing_rectangle_height = 360
        self.buffer = 50 
        self.padding_pitch = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_pitch_position()
        self.set_cricket_pitch_drawing_key_points()
        self.set_cricket_pitch_lines()

    def convert_meters_to_pixels(self, meters): 
        return convert_meters_to_pixel_distance(meters,
                                               cricket_constants.PITCH_LENGTH,
                                               self.pitch_drawing_length)

    def set_cricket_pitch_drawing_key_points(self):
        """Set cricket-specific keypoints for the mini pitch"""
        drawing_key_points = [0] * 32  # 16 keypoints for cricket pitch

        # Pitch rectangle (main playing area)
        # Point 0 - Top left of pitch
        drawing_key_points[0] = int(self.pitch_start_x)
        drawing_key_points[1] = int(self.pitch_start_y)

        # Point 1 - Top right of pitch  
        drawing_key_points[2] = int(self.pitch_end_x)
        drawing_key_points[3] = int(self.pitch_start_y)

        # Point 2 - Bottom right of pitch
        drawing_key_points[4] = int(self.pitch_end_x)
        drawing_key_points[5] = int(self.pitch_end_y)

        # Point 3 - Bottom left of pitch
        drawing_key_points[6] = int(self.pitch_start_x)
        drawing_key_points[7] = int(self.pitch_end_y)

        # Cricket pitch center line
        pitch_center_y = (self.pitch_start_y + self.pitch_end_y) / 2
        
        # Striker's end crease (points 4-5)
        striker_crease_y = self.pitch_start_y + self.convert_meters_to_pixels(2)
        drawing_key_points[8] = int(self.pitch_start_x + self.convert_meters_to_pixels(0.5))
        drawing_key_points[9] = int(striker_crease_y)
        drawing_key_points[10] = int(self.pitch_end_x - self.convert_meters_to_pixels(0.5))
        drawing_key_points[11] = int(striker_crease_y)

        # Bowler's end crease (points 6-7)
        bowler_crease_y = self.pitch_end_y - self.convert_meters_to_pixels(2)
        drawing_key_points[12] = int(self.pitch_start_x + self.convert_meters_to_pixels(0.5))
        drawing_key_points[13] = int(bowler_crease_y)
        drawing_key_points[14] = int(self.pitch_end_x - self.convert_meters_to_pixels(0.5))
        drawing_key_points[15] = int(bowler_crease_y)

        # Wickets (points 8-11)
        wicket_x = (self.pitch_start_x + self.pitch_end_x) / 2
        
        # Striker's wicket
        drawing_key_points[16] = int(wicket_x - self.convert_meters_to_pixels(0.1))
        drawing_key_points[17] = int(striker_crease_y)
        drawing_key_points[18] = int(wicket_x + self.convert_meters_to_pixels(0.1))
        drawing_key_points[19] = int(striker_crease_y)

        # Bowler's wicket  
        drawing_key_points[20] = int(wicket_x - self.convert_meters_to_pixels(0.1))
        drawing_key_points[21] = int(bowler_crease_y)
        drawing_key_points[22] = int(wicket_x + self.convert_meters_to_pixels(0.1))
        drawing_key_points[23] = int(bowler_crease_y)

        # Boundary circle points (points 12-15) - simplified as 4 points
        boundary_radius = min(self.pitch_drawing_width, self.pitch_drawing_height) * 0.4
        center_x = (self.pitch_start_x + self.pitch_end_x) / 2
        center_y = (self.pitch_start_y + self.pitch_end_y) / 2

        # Top boundary
        drawing_key_points[24] = int(center_x)
        drawing_key_points[25] = int(center_y - boundary_radius)

        # Right boundary
        drawing_key_points[26] = int(center_x + boundary_radius)
        drawing_key_points[27] = int(center_y)

        # Bottom boundary
        drawing_key_points[28] = int(center_x)
        drawing_key_points[29] = int(center_y + boundary_radius)

        # Left boundary
        drawing_key_points[30] = int(center_x - boundary_radius)
        drawing_key_points[31] = int(center_y)

        self.drawing_key_points = drawing_key_points

    def set_cricket_pitch_lines(self):
        """Define cricket pitch lines to be drawn"""
        self.lines = [
            # Pitch boundary rectangle
            (0, 1), (1, 2), (2, 3), (3, 0),
            
            # Creases
            (4, 5),  # Striker's crease
            (6, 7),  # Bowler's crease
            
            # Wickets
            (8, 9),   # Striker's wicket
            (10, 11), # Bowler's wicket
        ]
        
        # Boundary circle (connecting boundary points)
        self.boundary_lines = [
            (12, 13), (13, 14), (14, 15), (15, 12)
        ]

    def set_mini_pitch_position(self):
        self.pitch_start_x = self.start_x + self.padding_pitch
        self.pitch_start_y = self.start_y + self.padding_pitch
        self.pitch_end_x = self.end_x - self.padding_pitch
        self.pitch_end_y = self.end_y - self.padding_pitch
        self.pitch_drawing_width = self.pitch_end_x - self.pitch_start_x
        self.pitch_drawing_length = self.pitch_end_y - self.pitch_start_y

    def set_canvas_background_box_position(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height 
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_cricket_pitch(self, frame):
        """Draw the cricket pitch with proper styling"""
        # Draw pitch boundary
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), 
                          int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), 
                        int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

        # Draw boundary circle
        center_x = int((self.pitch_start_x + self.pitch_end_x) / 2)
        center_y = int((self.pitch_start_y + self.pitch_end_y) / 2)
        radius = int(min(self.pitch_drawing_width, self.pitch_drawing_length) * 0.4)
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)

        # Draw pitch center line
        cv2.line(frame, 
                (int(self.drawing_key_points[16]), int(self.drawing_key_points[17])),
                (int(self.drawing_key_points[20]), int(self.drawing_key_points[21])),
                (128, 128, 128), 1)

        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            
            # Different colors for different elements
            if i < 8:  # Pitch corners
                color = (255, 0, 0)  # Blue
            elif i < 16:  # Creases
                color = (0, 255, 0)  # Green
            elif i < 24:  # Wickets
                color = (0, 0, 255)  # Red
            else:  # Boundaries
                color = (255, 255, 0)  # Cyan
                
            cv2.circle(frame, (x, y), 4, color, -1)

        return frame

    def draw_background_rectangle(self, frame):
        """Draw semi-transparent background for mini pitch"""
        roi = frame[self.start_y:self.end_y, self.start_x:self.end_x].copy()
        
        # Create cricket field green background
        green_bg = np.ones_like(roi)
        green_bg[:, :] = [34, 139, 34]  # Forest green for cricket field
        
        # Blend with original
        alpha = 0.6
        blended_roi = cv2.addWeighted(roi, alpha, green_bg, 1 - alpha, 0)
        
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = blended_roi
        
        return frame

    def draw_mini_cricket_pitch(self, frames):
        """Draw mini cricket pitch on all frames"""
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_cricket_pitch_with_styling(frame)
            output_frames.append(frame)
        return output_frames

    def draw_cricket_pitch_with_styling(self, frame):
        """Draw cricket pitch with professional styling"""
        # Draw cricket field background
        pitch_surface = np.zeros_like(frame)
        
        # Create oval/circular field shape
        center_x = int((self.pitch_start_x + self.pitch_end_x) / 2)
        center_y = int((self.pitch_start_y + self.pitch_end_y) / 2)
        
        # Draw field as ellipse
        axes = (int(self.pitch_drawing_width//2), int(self.pitch_drawing_length//2))
        cv2.ellipse(pitch_surface, (center_x, center_y), axes, 0, 0, 360, (34, 139, 34), -1)
        
        # Blend with frame
        mask = np.any(pitch_surface != [0, 0, 0], axis=-1)
        frame[mask] = cv2.addWeighted(frame, 0.3, pitch_surface, 0.7, 0)[mask]

        # Draw pitch rectangle (22-yard strip)
        pitch_rect_points = np.array([
            [self.pitch_start_x + 50, self.pitch_start_y + 80],
            [self.pitch_end_x - 50, self.pitch_start_y + 80],
            [self.pitch_end_x - 50, self.pitch_end_y - 80],
            [self.pitch_start_x + 50, self.pitch_end_y - 80]
        ], np.int32)
        
        cv2.fillPoly(frame, [pitch_rect_points], (101, 67, 33))  # Brown pitch color

        # Draw cricket lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), 
                          int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), 
                        int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

        # Draw boundary circle
        cv2.circle(frame, (center_x, center_y), 
                  int(min(self.pitch_drawing_width, self.pitch_drawing_length) * 0.4), 
                  (255, 255, 255), 2)

        # Draw wickets as small rectangles
        wicket_positions = [(16, 17), (20, 21)]  # Striker and bowler wickets
        for wx, wy in wicket_positions:
            wicket_x = int(self.drawing_key_points[wx])
            wicket_y = int(self.drawing_key_points[wy])
            cv2.rectangle(frame, (wicket_x-3, wicket_y-8), (wicket_x+3, wicket_y+8), (139, 69, 19), -1)

        return frame

    def get_striker_end_position(self):
        """Get the striker's end position for player identification"""
        return (self.drawing_key_points[16], self.drawing_key_points[17])

    def get_pitch_length(self):
        """Get the length of the mini pitch for calculations"""
        return self.pitch_drawing_length

    def convert_bounding_boxes_to_mini_pitch_coordinates(self, player_boxes, ball_boxes, pitch_keypoints):
        """Convert bounding boxes to mini cricket pitch coordinates"""
        output_player_boxes_dict = {}
        output_ball_boxes_dict = {}
        
        for frame_num in range(len(player_boxes)):
            frame_player_boxes = player_boxes[frame_num]
            frame_ball_boxes = ball_boxes[frame_num] if frame_num < len(ball_boxes) else {}
            
            output_player_boxes_dict[frame_num] = {}
            output_ball_boxes_dict[frame_num] = {}
            
            # Process player bounding boxes
            for player_id, player_info in frame_player_boxes.items():
                try:
                    if isinstance(player_info, dict):
                        bbox = player_info['bbox']
                    else:
                        bbox = player_info
                        
                    foot_x, foot_y = self.get_foot_position(bbox)
                    
                    # Find closest pitch keypoint
                    closest_keypoint_index = self.get_closest_keypoint_index(
                        (foot_x, foot_y), 
                        pitch_keypoints, 
                        allowed_indices=range(min(8, len(pitch_keypoints) // 2))
                    )
                    
                    # Get mini pitch coordinate
                    mini_pitch_x = self.drawing_key_points[closest_keypoint_index * 2]
                    mini_pitch_y = self.drawing_key_points[closest_keypoint_index * 2 + 1]
                    
                    # Calculate offset
                    if len(pitch_keypoints) > closest_keypoint_index * 2 + 1:
                        kp_x = pitch_keypoints[closest_keypoint_index * 2]
                        kp_y = pitch_keypoints[closest_keypoint_index * 2 + 1]
                        
                        offset_x_norm = (foot_x - kp_x) / max(self.pitch_width, 1)
                        offset_y_norm = (foot_y - kp_y) / max(self.pitch_height, 1)
                        
                        offset_x_mini = offset_x_norm * self.mini_pitch_width
                        offset_y_mini = offset_y_norm * self.mini_pitch_height
                        
                        mini_pitch_x += offset_x_mini
                        mini_pitch_y += offset_y_mini
                    
                    # Constrain to pitch boundaries
                    mini_pitch_x = max(self.start_x, min(self.end_x, mini_pitch_x))
                    mini_pitch_y = max(self.start_y, min(self.end_y, mini_pitch_y))
                    
                    output_player_boxes_dict[frame_num][player_id] = (mini_pitch_x, mini_pitch_y)
                    
                except (ValueError, TypeError, IndexError):
                    # Default position
                    default_x = self.start_x + self.mini_pitch_width // 2
                    default_y = self.start_y + self.mini_pitch_height // 2
                    output_player_boxes_dict[frame_num][player_id] = (default_x, default_y)
            
            # Process ball bounding boxes
            for ball_id, bbox in frame_ball_boxes.items():
                try:
                    ball_x = (bbox[0] + bbox[2]) / 2
                    ball_y = (bbox[1] + bbox[3]) / 2
                    
                    # Similar processing as players but for ball
                    closest_keypoint_index = self.get_closest_keypoint_index(
                        (ball_x, ball_y), 
                        pitch_keypoints, 
                        allowed_indices=range(min(8, len(pitch_keypoints) // 2))
                    )
                    
                    mini_pitch_x = self.drawing_key_points[closest_keypoint_index * 2]
                    mini_pitch_y = self.drawing_key_points[closest_keypoint_index * 2 + 1]
                    
                    if len(pitch_keypoints) > closest_keypoint_index * 2 + 1:
                        kp_x = pitch_keypoints[closest_keypoint_index * 2]
                        kp_y = pitch_keypoints[closest_keypoint_index * 2 + 1]
                        
                        offset_x_norm = (ball_x - kp_x) / max(self.pitch_width, 1)
                        offset_y_norm = (ball_y - kp_y) / max(self.pitch_height, 1)
                        
                        offset_x_mini = offset_x_norm * self.mini_pitch_width
                        offset_y_mini = offset_y_norm * self.mini_pitch_height
                        
                        mini_pitch_x += offset_x_mini
                        mini_pitch_y += offset_y_mini
                    
                    mini_pitch_x = max(self.start_x, min(self.end_x, mini_pitch_x))
                    mini_pitch_y = max(self.start_y, min(self.end_y, mini_pitch_y))
                    
                    output_ball_boxes_dict[frame_num][ball_id] = (int(mini_pitch_x), int(mini_pitch_y))
                    
                except (ValueError, TypeError, IndexError):
                    # Default to center
                    default_x = self.start_x + self.mini_pitch_width // 2
                    default_y = self.start_y + self.mini_pitch_height // 2
                    output_ball_boxes_dict[frame_num][ball_id] = (default_x, default_y)
        
        return output_player_boxes_dict, output_ball_boxes_dict

    def get_foot_position(self, bbox):
        """Get foot position from bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, y2

    def get_closest_keypoint_index(self, position, keypoints, allowed_indices=None):
        """Find closest keypoint to position"""
        x, y = position
        min_dist = float('inf')
        closest_index = 0
        
        allowed_indices = allowed_indices if allowed_indices is not None else range(len(keypoints) // 2)
        
        for i in allowed_indices:
            if i*2+1 < len(keypoints):
                kp_x = keypoints[i*2]
                kp_y = keypoints[i*2+1]
                
                dist = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
        
        return closest_index

    def draw_cricket_players_on_mini_pitch(self, frames, positions):
        """Draw cricket players on mini pitch with role-specific colors"""
        for frame_num, frame in enumerate(frames):
            if frame_num in positions:
                for player_id, position in positions[frame_num].items():
                    try:
                        x, y = position
                        if np.isnan(x) or np.isnan(y):
                            continue
                            
                        x, y = int(x), int(y)
                        
                        # Cricket player colors based on role
                        if player_id == 1:  # Batsman
                            color = (0, 255, 0)  # Green
                        elif player_id == 2:  # Bowler
                            color = (255, 0, 0)  # Blue
                        elif player_id == 3:  # Wicket keeper
                            color = (0, 0, 255)  # Red
                        else:  # Fielders
                            color = (255, 255, 0)  # Cyan
                        
                        # Draw player
                        cv2.circle(frame, (x, y), 12, (0, 0, 0), -1)  # Black outline
                        cv2.circle(frame, (x, y), 10, color, -1)  # Colored fill
                        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)  # White center
                        
                    except (ValueError, TypeError):
                        continue
        
        return frames

    def draw_points_on_mini_pitch(self, frames, positions, color=(255, 255, 0)):
        """Draw points (like ball) on mini pitch"""
        for frame_num, frame in enumerate(frames):
            if frame_num in positions:
                for obj_id, position in positions[frame_num].items():
                    try:
                        x, y = position
                        if np.isnan(x) or np.isnan(y):
                            continue
                            
                        x, y = int(x), int(y)
                        
                        # Draw cricket ball
                        cv2.circle(frame, (x, y), 8, (0, 0, 0), -1)  # Black outline
                        cv2.circle(frame, (x, y), 6, color, -1)  # Colored ball
                        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # White center
                        
                    except (ValueError, TypeError):
                        continue
        
        return frames

    def draw_ball_trajectory(self, frames, positions):
        """Draw ball trajectory on mini pitch"""
        # Store previous positions for trajectory
        trajectory_points = []
        
        for frame_num, frame in enumerate(frames):
            if frame_num in positions and 1 in positions[frame_num]:
                try:
                    x, y = positions[frame_num][1]
                    if not (np.isnan(x) or np.isnan(y)):
                        trajectory_points.append((int(x), int(y)))
                        
                        # Keep only last 10 points for trajectory
                        if len(trajectory_points) > 10:
                            trajectory_points.pop(0)
                        
                        # Draw trajectory line
                        if len(trajectory_points) > 1:
                            for i in range(1, len(trajectory_points)):
                                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], 
                                        (0, 255, 255), 2)
                                
                except (ValueError, TypeError):
                    continue
        
        return frames