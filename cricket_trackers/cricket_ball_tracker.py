from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Cricket-specific ball event types
        self.BALL_EVENTS = {
            'DELIVERY': 'Delivery',
            'SHOT': 'Shot',
            'BOUNDARY': 'Boundary',
            'WICKET': 'Wicket',
            'CATCH': 'Catch'
        }

    def interpolate_ball_positions(self, ball_positions):
        """Enhanced ball position interpolation for cricket"""
        ball_positions = [x.get(1, []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        if df_ball_positions.isna().any().any() or df_ball_positions.empty:
            # Cricket-specific interpolation - more aggressive smoothing for fast ball
            df_ball_positions = df_ball_positions.interpolate(method='linear')
            df_ball_positions = df_ball_positions.interpolate(method='polynomial', order=3)
            df_ball_positions = df_ball_positions.fillna(method='ffill').fillna(method='bfill')
        else:
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()
        
        # Cricket ball smoothing - smaller window for faster ball
        window_size = 3
        for col in df_ball_positions.columns:
            df_ball_positions[col] = df_ball_positions[col].rolling(
                window=window_size, min_periods=1, center=True).mean()
            
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def get_cricket_ball_events(self, ball_positions):
        """
        Detect cricket-specific ball events like deliveries, shots, boundaries
        """
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_event'] = 0
        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        
        # Calculate velocity and acceleration for cricket ball
        df_ball_positions['velocity_x'] = df_ball_positions['mid_x'].diff()
        df_ball_positions['velocity_y'] = df_ball_positions['mid_y'].diff()
        df_ball_positions['speed'] = np.sqrt(
            df_ball_positions['velocity_x']**2 + df_ball_positions['velocity_y']**2
        )
        
        # Rolling mean for smoothing
        df_ball_positions['speed_rolling'] = df_ball_positions['speed'].rolling(
            window=5, min_periods=1, center=False).mean()
        df_ball_positions['acceleration'] = df_ball_positions['speed_rolling'].diff()
        
        # Cricket-specific event detection
        minimum_change_frames = 15  # Shorter for cricket
        
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames * 1.2)):
            # Detect sudden direction changes (shots)
            speed_change = abs(df_ball_positions['acceleration'].iloc[i])
            direction_change_x = (df_ball_positions['velocity_x'].iloc[i] > 0) != (df_ball_positions['velocity_x'].iloc[i+1] > 0)
            direction_change_y = (df_ball_positions['velocity_y'].iloc[i] > 0) != (df_ball_positions['velocity_y'].iloc[i+1] > 0)
            
            if speed_change > 2 or direction_change_x or direction_change_y:
                # Verify sustained change
                change_count = 0
                for change_frame in range(i+1, min(i + minimum_change_frames + 1, len(df_ball_positions))):
                    if abs(df_ball_positions['acceleration'].iloc[change_frame]) > 1:
                        change_count += 1
                
                if change_count > minimum_change_frames // 2:
                    df_ball_positions.loc[i, 'ball_event'] = 1

        event_frames = df_ball_positions[df_ball_positions['ball_event'] == 1].index.tolist()
        
        return event_frames

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections, color=(255, 255, 0), thickness=2):
        """
        Draw cricket ball bounding boxes with enhanced visibility
        """
        for i, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            for track_id, bbox in ball_dict.items():
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate center and radius
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2) + 4
                    
                    # Draw cricket ball with distinctive styling
                    # Outer circle
                    cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), 3)
                    # Inner circle with cricket ball color
                    cv2.circle(frame, (center_x, center_y), radius-1, color, -1)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Add "BALL" label
                    cv2.putText(frame, "BALL", (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return video_frames