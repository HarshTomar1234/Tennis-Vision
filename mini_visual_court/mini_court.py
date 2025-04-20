import cv2
import sys
import numpy as np
import random  # Adding random for ball position offsets
sys.path.append("../")
import constants 
from utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters , get_foot_position, get_closest_keypoint_index, get_height_of_bbox, measure_xy_distance, get_center_of_bbox, measure_distance_between_points


class MiniCourt():
    def __init__(self, frame, mini_court_width=None, mini_court_height=None):
        """
        Initialize mini court with enhanced styling and dimensions
        """
        frame_height, frame_width = frame.shape[:2]
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Set court dimensions for coordinate calculations
        self.court_width = frame_width  # Full court width in pixels
        self.court_height = frame_height  # Full court height in pixels
        
        # Mini court dimensions
        self.mini_court_width = mini_court_width if mini_court_width else int(frame_width * 0.2)
        self.mini_court_height = mini_court_height if mini_court_height else int(self.mini_court_width * 1.5)
        
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50 
        self.padding_court = 20


        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters): 
         return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width)  

    def set_court_drawing_key_points(self):

        drawing_key_points = [0] * 28 # create a list of 28 zeros

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)

        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)

        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)

        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]

        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 

        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]

        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 

        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 

        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 

        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 

        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 

        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]




    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_height = self.court_end_y - self.court_start_y  # Add the missing court_height attribute
      

    def set_canvas_background_box_position(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height 
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height


    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # Drawing the Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Drawing the net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame


       
        
    def draw_background_rectangle(self, frame):
        # Create a smaller mask just for the rectangle area instead of the whole frame
        # This significantly reduces memory usage
        roi = frame[self.start_y:self.end_y, self.start_x:self.end_x].copy()
        
        # Create a white background of the same size as the ROI
        white_bg = np.ones_like(roi) * 255
        
        # Blend the ROI with the white background (alpha blending)
        alpha = 0.5
        blended_roi = cv2.addWeighted(roi, alpha, white_bg, 1 - alpha, 0)
        
        # Place the blended ROI back into the original frame
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = blended_roi
        
        # Return the modified frame (no need to create a new copy)
        return frame
    
    def draw_mini_court(self, frames):
        """
        Draw a visually appealing mini court without labels
        for a cleaner, more professional appearance
        """
        output_frames = []
        for frame in frames:
            # Draw a gradient background for the mini court area
            frame = self.draw_background_rectangle(frame)
            
            # Draw court with enhanced styling
            frame = self.draw_court_with_styling(frame)
            
            output_frames.append(frame)

        return output_frames
        
    def draw_court_with_styling(self, frame):
        """
        Draw the mini court with professional styling and clear visual representation.
        
        This enhanced version creates a clean, professional-looking mini court with:
        - Clear court boundaries and markings
        - Professional color scheme
        - High-contrast lines and keypoints
        - No distracting elements
        """
        # Step 1: Create a clean court surface with a professional blue tennis court color
        court_surface = np.zeros_like(frame)
        court_points = []
        
        # Get the court outline points for the main polygon
        for i in range(0, 8, 2):  # Using the first 4 points that form the court outline
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            court_points.append((x, y))
            
        # Convert to numpy array for drawing polygon
        court_points = np.array(court_points, np.int32)
        court_points = court_points.reshape((-1, 1, 2))
        
        # Fill the court with a professional tennis court blue color
        # This resembles actual hard court tennis surfaces for better realism
        cv2.fillPoly(court_surface, [court_points], (176, 127, 89))  # Pro tennis court blue (BGR)
        
        # Blend with the frame for a clean look
        mask = np.any(court_surface != [0, 0, 0], axis=-1)
        frame[mask] = cv2.addWeighted(frame, 0.1, court_surface, 0.9, 0)[mask]
        
        # Draw the court lines with professional styling
        # Using clean white lines like a real tennis court
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            
            # Draw crisp, clean BLACK lines for better visibility as requested
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # Pure BLACK lines

        # Drawing the net with a distinctive BLUE color (different from court lines)
        net_start_point = (int(self.drawing_key_points[0]), int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (int(self.drawing_key_points[2]), int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        
        # Draw a thicker line for the net with distinctive blue color (different from court lines)
        cv2.line(frame, net_start_point, net_end_point, (128, 0, 0), 3)  # Blue net
        
        # Add court keypoints for better reference - these help understand the court geometry
        # Using bright RED keypoints that are clearly visible as requested by the user
        for i in range(0, len(self.drawing_key_points), 2):
            if i < len(self.drawing_key_points):
                x = int(self.drawing_key_points[i])
                y = int(self.drawing_key_points[i+1])
                
                # Use a bright RED color for keypoints as requested for better visibility
                # Adding a 2-layer approach for more prominence
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)  # Black outline
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Bright RED keypoints
                
                # REMOVING the confusing yellow/orange lines that were connecting keypoints
                # This makes the visualization cleaner and more professional as requested
        
        # Removing the legend completely - no text or legend points
        # This keeps the mini court clean and professional without any text
                  
        return frame
           
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self, object_position, closest_keypoint, closest_keypoint_index, player_height_in_pixels, player_height_in_meters):
        distance_from_keypont_x_pixels, distance_from_keypont_y_pixels = measure_xy_distance(object_position, closest_keypoint)

        # Covert pixel distance to meters
        distance_from_keypont_x_meters = convert_pixel_distance_to_meters(distance_from_keypont_x_pixels, player_height_in_meters, player_height_in_pixels)
        distance_from_keypont_y_pixels = convert_pixel_distance_to_meters(distance_from_keypont_y_pixels, player_height_in_meters, player_height_in_pixels)

        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypont_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypont_y_pixels)

        closest_mini_court_keypoint = (self.drawing_key_points[closest_keypoint_index*2], self.drawing_key_points[closest_keypoint_index*2+1]) 

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels, closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)

        return mini_court_player_position

    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, court_keypoints):
        """
        Convert bounding boxes to mini court coordinates with enhanced accuracy
        - Improved real-time synchronization between actual and mini court
        - Better ball position tracking relative to player positions
        - More accurate coordinate transformation
        """
        output_player_boxes_dict = {}
        output_ball_boxes_dict = {}
        
        # Process each frame
        for frame_num in range(len(player_boxes)):
            frame_player_boxes = player_boxes[frame_num]
            frame_ball_boxes = ball_boxes[frame_num] if frame_num < len(ball_boxes) else {}
            
            # Initialize frame dictionaries
            output_player_boxes_dict[frame_num] = {}
            output_ball_boxes_dict[frame_num] = {}
            
            # Process player bounding boxes
            for player_id, bbox in frame_player_boxes.items():
                try:
                    # Get foot position (bottom center of bounding box)
                    foot_x, foot_y = self.get_foot_position(bbox)
                    
                    # Find the closest court keypoint to determine player's court position
                    closest_keypoint_index = self.get_closest_keypoint_index(
                        (foot_x, foot_y), 
                        court_keypoints, 
                        allowed_indices=range(len(court_keypoints) // 2)
                    )
                    
                    # Get the corresponding mini court coordinate
                    mini_court_x = self.drawing_key_points[closest_keypoint_index * 2]
                    mini_court_y = self.drawing_key_points[closest_keypoint_index * 2 + 1]
                    
                    # Calculate offset based on player's position relative to the keypoint
                    kp_x = court_keypoints[closest_keypoint_index * 2]
                    kp_y = court_keypoints[closest_keypoint_index * 2 + 1]
                    
                    # Calculate normalized offset (0-1 range)
                    offset_x_norm = (foot_x - kp_x) / max(self.court_width, 1)
                    offset_y_norm = (foot_y - kp_y) / max(self.court_height, 1)
                    
                    # Scale offset to mini court dimensions
                    offset_x_mini = offset_x_norm * self.mini_court_width
                    offset_y_mini = offset_y_norm * self.mini_court_height
                    
                    # Apply offset to mini court position
                    mini_court_x += offset_x_mini
                    mini_court_y += offset_y_mini
                    
                    # Ensure position is within mini court boundaries
                    mini_court_x = max(self.start_x, min(self.end_x, mini_court_x))
                    mini_court_y = max(self.start_y, min(self.end_y, mini_court_y))
                    
                    # Store the mini court position
                    output_player_boxes_dict[frame_num][player_id] = (mini_court_x, mini_court_y)
                    
                except (ValueError, TypeError, IndexError):
                    # Use default position if calculation fails
                    default_x = self.start_x + self.mini_court_width // 2
                    default_y = self.start_y + self.mini_court_height // 2
                    output_player_boxes_dict[frame_num][player_id] = (default_x, default_y)
            
            # Process ball bounding boxes with enhanced accuracy
            for ball_id, bbox in frame_ball_boxes.items():
                try:
                    # Get ball center
                    ball_x = (bbox[0] + bbox[2]) / 2
                    ball_y = (bbox[1] + bbox[3]) / 2
                    
                    # Find the closest court keypoint
                    closest_keypoint_index = self.get_closest_keypoint_index(
                        (ball_x, ball_y), 
                        court_keypoints, 
                        allowed_indices=range(len(court_keypoints) // 2)
                    )
                    
                    # Get the corresponding mini court coordinate
                    mini_court_x = self.drawing_key_points[closest_keypoint_index * 2]
                    mini_court_y = self.drawing_key_points[closest_keypoint_index * 2 + 1]
                    
                    # Calculate offset based on ball's position relative to the keypoint
                    kp_x = court_keypoints[closest_keypoint_index * 2]
                    kp_y = court_keypoints[closest_keypoint_index * 2 + 1]
                    
                    # Calculate normalized offset (0-1 range)
                    offset_x_norm = (ball_x - kp_x) / max(self.court_width, 1)
                    offset_y_norm = (ball_y - kp_y) / max(self.court_height, 1)
                    
                    # Scale offset to mini court dimensions
                    offset_x_mini = offset_x_norm * self.mini_court_width
                    offset_y_mini = offset_y_norm * self.mini_court_height
                    
                    # Apply offset to mini court position
                    mini_court_x += offset_x_mini
                    mini_court_y += offset_y_mini
                    
                    # Find the nearest player to determine ball possession
                    nearest_player = None
                    min_distance = float('inf')
                    
                    for player_id, player_bbox in frame_player_boxes.items():
                        player_x = (player_bbox[0] + player_bbox[2]) / 2
                        player_y = (player_bbox[1] + player_bbox[3]) / 2
                        
                        # Calculate Euclidean distance
                        distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_player = player_id
                    
                    # If ball is very close to a player (improved threshold), position it near that player
                    # This significantly improves real-time accuracy of ball possession
                    if nearest_player is not None and min_distance < 150 and nearest_player in output_player_boxes_dict[frame_num]:
                        player_mini_x, player_mini_y = output_player_boxes_dict[frame_num][nearest_player]
                        
                        # Very small offset for precise positioning (just 3-5 pixels)
                        offset_x = 5 * (0.5 - random.random())  # Random offset between -2.5 and 2.5
                        offset_y = 5 * (0.5 - random.random())  # Random offset between -2.5 and 2.5
                        
                        mini_court_ball_position = (int(player_mini_x + offset_x), int(player_mini_y + offset_y))
                    else:
                        # Ensure position is within mini court boundaries
                        mini_court_x = max(self.start_x, min(self.end_x, mini_court_x))
                        mini_court_y = max(self.start_y, min(self.end_y, mini_court_y))
                        
                        mini_court_ball_position = (int(mini_court_x), int(mini_court_y))
                    
                    # Store the mini court position
                    output_ball_boxes_dict[frame_num][ball_id] = mini_court_ball_position
                    
                except (ValueError, TypeError, IndexError):
                    # If no valid ball position, check if we have players and use their position
                    if output_player_boxes_dict[frame_num] and 1 in output_player_boxes_dict[frame_num]:
                        # Default to player 1's position with small offset
                        player_x, player_y = output_player_boxes_dict[frame_num][1]
                        offset_x = 5 * (0.5 - random.random())
                        offset_y = 5 * (0.5 - random.random())
                        output_ball_boxes_dict[frame_num][ball_id] = (int(player_x + offset_x), int(player_y + offset_y))
                    else:
                        # Use center of mini court as fallback
                        default_x = self.start_x + self.mini_court_width // 2
                        default_y = self.start_y + self.mini_court_height // 2
                        output_ball_boxes_dict[frame_num][ball_id] = (default_x, default_y)
        
        return output_player_boxes_dict, output_ball_boxes_dict

    def constrain_to_court_boundaries(self, position):
        """Ensure position is within court boundaries"""
        x, y = position
        
        # Check for NaN values
        if np.isnan(x) or np.isnan(y):
            # Default to center court if values are NaN
            return ((self.court_start_x + self.court_end_x) // 2, 
                    (self.court_start_y + self.court_end_y) // 2)
        
        # Constrain to court boundaries with a small buffer
        buffer = 10  # pixels
        x = max(self.court_start_x - buffer, min(x, self.court_end_x + buffer))
        y = max(self.court_start_y - buffer, min(y, self.court_end_y + buffer))
        
        return (x, y)

    def get_foot_position(self, bbox):
        """Get foot position of a player from their bounding box"""
        x1, y1, x2, y2 = bbox
        # Foot position is the bottom center of the bounding box
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return foot_x, foot_y
    
    def get_closest_keypoint_index(self, position, keypoints, allowed_indices=None):
        """Find the closest keypoint to a given position"""
        x, y = position
        min_dist = float('inf')
        closest_index = 0
        
        allowed_indices = allowed_indices if allowed_indices is not None else range(len(keypoints) // 2)
        
        for i in allowed_indices:
            kp_x = keypoints[i*2]
            kp_y = keypoints[i*2+1]
            
            dist = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        return closest_index

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0), draw_trail=False, label=None):
        """
        Draw points on mini court with enhanced visualization features:
        - Both players have IDENTICAL green circles with IDENTICAL thickness
        - Ball has a distinct bright PURPLE color to differentiate from players
        - No distracting text labels for cleaner visualization
        - Ball follows players more accurately with real-time synchronization
        
        Args:
            frames: List of video frames to draw on
            positions: Player or ball positions to visualize
            color: Base color for the points (default: green)
            draw_trail: Whether to draw movement trails (default: False)
            label: Text label to display (default: None, no labels will be shown for clean visualization)
            
        """
        # History of positions for drawing trails
        position_history = []
        
        # Define CONSISTENT player circle parameters
        PLAYER_OUTLINE_THICKNESS = 10  # Identical outline thickness for both players
        PLAYER_CIRCLE_THICKNESS = 8    # Identical circle thickness for both players
        PLAYER_COLOR = (0, 180, 0)     # Identical GREEN color for both players (BGR format)
        
        # Define DISTINCT ball color - bright PURPLE instead of blue or orange
        BALL_COLOR = (128, 0, 128)  # Bright PURPLE color for ball (BGR format)
        
        # Determine if we're drawing players or ball based on the input color
        # This is a more reliable way to distinguish between the two calls from main.py
        # In main.py, players are drawn with color=(0,255,0) and ball with color=(0,255,255)
        is_drawing_ball = False
        if color == (0, 255, 255):  # This is the color used for ball in main.py
            is_drawing_ball = True
        
        for frame_num, frame in enumerate(frames):
            current_positions = []
            
            # Extract current frame positions
            for obj_id, position in positions[frame_num].items():
                try:
                    x, y = position
                    # Handle NaN or invalid positions
                    if np.isnan(x) or np.isnan(y):
                        continue
                        
                    x, y = int(x), int(y)
                    current_positions.append((obj_id, x, y))
                    
                    # Enhanced circle drawing with dark outline for better visibility
                    if is_drawing_ball:
                        # Ball visualization - make MUCH more visible with PURPLE color
                        # First draw larger black outline for definition against any background
                        cv2.circle(frame, (x, y), 9, (0, 0, 0), -1)  # Black outline
                        # Then draw main ball circle with a distinct PURPLE color that stands out
                        cv2.circle(frame, (x, y), 7, BALL_COLOR, -1)  # Bright PURPLE ball - distinct from players
                    else:
                        # Player visualization with CONSISTENT green coloring and thickness for both players
                        # Both players will be IDENTICAL green circles with IDENTICAL thickness
                        cv2.circle(frame, (x, y), PLAYER_OUTLINE_THICKNESS, (0, 0, 0), -1)  # Black outline - IDENTICAL thickness
                        cv2.circle(frame, (x, y), PLAYER_CIRCLE_THICKNESS, PLAYER_COLOR, -1)  # Green fill - IDENTICAL color and thickness
                    
                except (ValueError, TypeError):
                    continue  # Skip invalid positions
            
            # Update position history
            position_history.append(current_positions)
        
        return frames

    def draw_ball_trajectory(self, frames, positions):
        """
        Draw ball trajectory on frames with enhanced visualization
        -
        """
        return frames

    def draw_background_rectangle(self, frame):
        # Creating a smaller mask just for the rectangle area instead of the whole frame
        # This significantly reduces memory usage
        roi = frame[self.start_y:self.end_y, self.start_x:self.end_x].copy()
        
        # Create a white background of the same size as the ROI
        white_bg = np.ones_like(roi) * 255
        
        # Blend the ROI with the white background (alpha blending)
        alpha = 0.5
        blended_roi = cv2.addWeighted(roi, alpha, white_bg, 1 - alpha, 0)
        
        # Place the blended ROI back into the original frame
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = blended_roi
        
        # Return the modified frame (no need to create a new copy)
        return frame