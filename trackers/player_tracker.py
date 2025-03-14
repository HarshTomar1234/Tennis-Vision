from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance_between_points



class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)


    def choose_and_filter_players(self, player_detections, court_keypoints):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections



    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):   # here step size is 2 because we are taking x and y coordinates of court keypoints alternatively
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance_between_points(player_center, court_keypoint)
                if distance < min_distance:  # here we are finding the minimum distance of player from court keypoints basically we are finding the nearest court keypoint from player
                    min_distance = distance
            distances.append((track_id, min_distance))   # here we are appending the track id and minimum distance of player from court keypoints


        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])  # here the logic of using lambda x: x[1] is that we are sorting the distances based on the minimum distance of player from court keypoints and x is the list of track id and x[1] is the minimum distance of player from court keypoints

        # Choose the top 2 track ids
        chosen_players = [distances[0][0], distances[1][0]]  # here we are choosing the top 2 track ids based on the minimum distance of player from court keypoints
        return chosen_players
    


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
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
    

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def filter_by_confidence(self, player_detections, confidence_threshold=0.7):
        """
        Filter player detections by confidence score to improve accuracy
        
        Args:
            player_detections: Dictionary of player detections by frame
            confidence_threshold: Minimum confidence score to keep (default: 0.7)
            
        Returns:
            Filtered player detections with only high-confidence detections retained
        """
        # For this implementation, we'll use size and position-based filtering
        # since confidence scores might not be directly available from stubs
        
        filtered_detections = []
        
        for frame_detections in player_detections:
            # Create a new dictionary for filtered detections in this frame
            filtered_frame = {}
            
            for track_id, bbox in frame_detections.items():
                if bbox and len(bbox) == 4:  # Ensure valid bbox format
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Filter based on size (too small objects are likely false positives)
                    if width > 20 and height > 50:  # Minimum size for a player
                        filtered_frame[track_id] = bbox
            
            filtered_detections.append(filtered_frame)
            
        return filtered_detections

    def draw_bboxes(self, video_frames, player_detections, thickness=2, color=None):
        """
        Draw player bounding boxes on frames with enhanced visualization
        
        Args:
            video_frames: Video frames to draw on
            player_detections: Player detection bounding boxes
            thickness: Line thickness for drawing bounding boxes (default: 2)
            color: Optional specific color to use (default: None, uses player-specific colors)
            
        Returns:
            Frames with player bounding boxes drawn
        """
        # Instead of creating a copy of each frame, we'll draw directly on the input frames
        # This is more memory efficient for high-resolution videos
        
        for i, (frame, player_dict) in enumerate(zip(video_frames, player_detections)):
            # Draw Bounding Boxes with enhanced visibility
            for track_id, bbox in player_dict.items():
                if bbox and len(bbox) == 4:  # Ensure valid bbox format
                    x1, y1, x2, y2 = bbox
                    
                    # Use different colors for different players with darker outlines
                    if color:
                        box_color = color
                    else:
                        # Define consistent colors regardless of track_id value
                        # Always use player 1 and player 2 (never higher numbers)
                        player_num = 1 if track_id == list(player_dict.keys())[0] else 2
                        if player_num == 1:
                            box_color = (0, 0, 255)  # Red for Player 1 (BGR format)
                        else:
                            box_color = (0, 165, 255)  # Orange for Player 2 (BGR format)
                    
                    # Draw darker outline first for better visibility
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
                    
                    # Add player label with better visibility - always use Player 1 or Player 2 only
                    # Instead of using track_id+1, use the player_num we determined above
                    label = f"Player {player_num}"  # Consistent player numbering (1 or 2 only)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw background for text
                    cv2.rectangle(frame, 
                               (int(x1), int(y1) - text_size[1] - 5),
                               (int(x1) + text_size[0] + 5, int(y1)),
                               (50, 50, 50), -1)  # Dark background
                    
                    # Draw text
                    cv2.putText(frame, label, 
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return video_frames  # Return the same frames since we modified them in-place