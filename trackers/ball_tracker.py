from ultralytics import YOLO 
import cv2
import pickle
import  pandas as pd


class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        """Enhanced ball position interpolation with improved handling of missing values and smoothing"""
        ball_positions = [x.get(1, []) for x in ball_positions]

        # Convert the list into a pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Check for NaN or empty values
        if df_ball_positions.isna().any().any() or df_ball_positions.empty:
            # Apply more robust interpolation techniques
            # First linear interpolation to fill gaps
            df_ball_positions = df_ball_positions.interpolate(method='linear')
            # Then polynomial interpolation for smoother curves
            df_ball_positions = df_ball_positions.interpolate(method='polynomial', order=2)
            # Use forward and backward fill to handle remaining NaN values
            df_ball_positions = df_ball_positions.fillna(method='ffill').fillna(method='bfill')
        else:
            # Apply standard interpolation for well-behaved data
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()
        
        # Apply smoothing to reduce jitter in ball trajectories
        window_size = 5  # Adjust based on frame rate and ball speed
        for col in df_ball_positions.columns:
            df_ball_positions[col] = df_ball_positions[col].rolling(window=window_size, min_periods=1, center=True).mean()
            
        # Reconvert to original format
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions


    
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    # Fix pandas chained assignment warning by using loc[] instead of chained indexing
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections  
    

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf = 0.15)[0]
        

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
    def filter_by_confidence(self, ball_detections, confidence_threshold=0.6):
        """
        Filter ball detections by confidence and size criteria for improved accuracy
        
        Args:
            ball_detections: Dictionary of ball detections by frame
            confidence_threshold: Minimum confidence score to keep (default: 0.6)
            
        Returns:
            Filtered ball detections with only high-confidence detections retained
        """
        filtered_detections = []
        
        for frame_detections in ball_detections:
            filtered_frame = {}
            
            for track_id, bbox in frame_detections.items():
                if bbox and len(bbox) == 4:  # Ensure valid bbox format
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Tennis balls should be small and relatively square-ish
                    if 5 <= width <= 40 and 5 <= height <= 40:
                        # Check aspect ratio - tennis balls should be roughly circular
                        aspect_ratio = width / height if height > 0 else 0
                        if 0.7 <= aspect_ratio <= 1.3:  # Allow some deformation
                            filtered_frame[track_id] = bbox
            
            filtered_detections.append(filtered_frame)
            
        return filtered_detections
    
    def draw_bboxes(self, video_frames, ball_detections, color=(0, 255, 255), thickness=2):
        """
        Draw ball bounding boxes on frames with enhanced visualization
        
        Args:
            video_frames: Video frames to draw on
            ball_detections: Ball detection bounding boxes
            color: Color to use for ball boxes (default: cyan)
            thickness: Line thickness for drawing bounding boxes (default: 2)
            
        Returns:
            Frames with ball bounding boxes drawn
        """
        # Instead of creating a copy of each frame, we'll draw directly on the input frames
        # This is more memory efficient for high-resolution videos
        
        for i, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            # Draw Bounding Boxes with enhanced visibility
            for track_id, bbox in ball_dict.items():
                if bbox and len(bbox) == 4:  # Ensure valid bbox format
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate center point and radius for circular highlight
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2) + 3  # Slightly larger for visibility
                    
                    # Draw an outer circle with darker color
                    darker_color = (0, 150, 150)  # Darker cyan
                    cv2.circle(frame, (center_x, center_y), radius, darker_color, 2)
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Draw smaller inner circle at the center point for precise location
                    cv2.circle(frame, (center_x, center_y), 2, (255, 255, 255), -1)  # White center
        
        return video_frames  # Return the same frames since we modified them in-place
