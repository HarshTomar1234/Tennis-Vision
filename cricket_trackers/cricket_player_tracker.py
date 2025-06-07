from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance_between_points
import cricket_constants

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Cricket-specific player roles
        self.PLAYER_ROLES = {
            'BATSMAN': 'Batsman',
            'BOWLER': 'Bowler', 
            'WICKET_KEEPER': 'Wicket Keeper',
            'FIELDER': 'Fielder'
        }
        
        # Colors for different player roles (BGR format)
        self.ROLE_COLORS = {
            'BATSMAN': (0, 255, 0),      # Green
            'BOWLER': (255, 0, 0),       # Blue  
            'WICKET_KEEPER': (0, 0, 255), # Red
            'FIELDER': (255, 255, 0)     # Cyan
        }

    def identify_cricket_players(self, player_detections, pitch_keypoints):
        """
        Identify cricket players based on their positions relative to the pitch
        """
        if not player_detections or not pitch_keypoints:
            return player_detections
            
        first_frame = player_detections[0]
        player_roles = self.assign_cricket_roles(first_frame, pitch_keypoints)
        
        # Filter and assign roles to all frames
        filtered_detections = []
        for frame_detections in player_detections:
            filtered_frame = {}
            for track_id, bbox in frame_detections.items():
                if track_id in player_roles:
                    filtered_frame[track_id] = {
                        'bbox': bbox,
                        'role': player_roles[track_id]
                    }
            filtered_detections.append(filtered_frame)
            
        return filtered_detections

    def assign_cricket_roles(self, player_dict, pitch_keypoints):
        """
        Assign cricket roles based on player positions
        """
        if len(pitch_keypoints) < 8:  # Need at least 4 keypoints (8 coordinates)
            return {}
            
        roles = {}
        
        # Get pitch center and ends
        pitch_center_x = (pitch_keypoints[0] + pitch_keypoints[2]) / 2
        pitch_center_y = (pitch_keypoints[1] + pitch_keypoints[3]) / 2
        
        striker_end_y = min(pitch_keypoints[1], pitch_keypoints[3])
        bowler_end_y = max(pitch_keypoints[1], pitch_keypoints[3])
        
        player_positions = []
        for track_id, bbox in player_dict.items():
            center = get_center_of_bbox(bbox)
            distance_to_pitch = measure_distance_between_points(center, (pitch_center_x, pitch_center_y))
            player_positions.append((track_id, center, distance_to_pitch))
        
        # Sort by distance to pitch
        player_positions.sort(key=lambda x: x[2])
        
        # Assign roles based on position
        for i, (track_id, center, distance) in enumerate(player_positions):
            if i == 0:  # Closest to pitch - likely batsman
                roles[track_id] = self.PLAYER_ROLES['BATSMAN']
            elif i == 1:  # Second closest - could be bowler or wicket keeper
                if center[1] < pitch_center_y:  # Towards bowler's end
                    roles[track_id] = self.PLAYER_ROLES['BOWLER']
                else:  # Behind wickets
                    roles[track_id] = self.PLAYER_ROLES['WICKET_KEEPER']
            elif i < 6:  # Next few are fielders
                roles[track_id] = self.PLAYER_ROLES['FIELDER']
        
        return roles

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == "person":
                    player_dict[track_id] = result
        
        return player_dict

    def draw_cricket_bboxes(self, video_frames, player_detections, thickness=2):
        """
        Draw cricket player bounding boxes with role-specific colors and labels
        """
        for i, (frame, player_dict) in enumerate(zip(video_frames, player_detections)):
            for track_id, player_info in player_dict.items():
                if isinstance(player_info, dict):
                    bbox = player_info['bbox']
                    role = player_info.get('role', 'FIELDER')
                else:
                    bbox = player_info
                    role = 'FIELDER'
                
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Get role-specific color
                    box_color = self.ROLE_COLORS.get(role, (255, 255, 255))
                    
                    # Draw darker outline
                    darker_color = tuple(max(0, c//2) for c in box_color)
                    cv2.rectangle(frame, 
                               (int(x1)-1, int(y1)-1), 
                               (int(x2)+1, int(y2)+1), 
                               darker_color, thickness+2)
                    
                    # Draw main rectangle
                    cv2.rectangle(frame, 
                               (int(x1), int(y1)), 
                               (int(x2), int(y2)), 
                               box_color, thickness)
                    
                    # Add role label
                    label = role
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw background for text
                    cv2.rectangle(frame, 
                               (int(x1), int(y1) - text_size[1] - 5),
                               (int(x1) + text_size[0] + 5, int(y1)),
                               (50, 50, 50), -1)
                    
                    # Draw text
                    cv2.putText(frame, label, 
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return video_frames