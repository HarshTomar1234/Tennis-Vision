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
        """
        CAMERA-ROBUST V2: Enhanced player selection with OPPONENT SEPARATION.
        
        Key insight: Tennis players are at OPPOSITE ENDS of the court (top and bottom),
        while ball boys and line judges are at the SIDES.
        
        New approach:
        1. Score each candidate on multiple criteria
        2. Select the best player from the BOTTOM half of court (near baseline)
        3. Select the best player from the TOP half of court (far baseline)
        4. This ensures we get actual opponents, not two ball boys!
        """
        if len(player_dict) < 2:
            return list(player_dict.keys())
        
        # Calculate court bounds
        court_bounds = self._estimate_court_bounds(court_keypoints)
        court_mid_y = (court_bounds['top'] + court_bounds['bottom']) / 2
        
        # Score all candidates
        candidates = []
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            player_center = get_center_of_bbox(bbox)
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            bbox_area = bbox_height * bbox_width
            
            score = 0
            
            # ====== CRITERION 1: Inside court bounds (+80 points) ======
            if self._is_inside_court(player_center, court_bounds, margin=50):
                score += 80
            elif self._is_inside_court(player_center, court_bounds, margin=150):
                score += 40
            
            # ====== CRITERION 2: Bounding box size (+60 points max) ======
            # Real players appear LARGER than ball boys due to camera focus
            area_score = min(bbox_area / 800, 60)
            score += area_score
            
            # ====== CRITERION 3: Aspect ratio (+30 points) ======
            aspect_ratio = bbox_height / max(bbox_width, 1)
            if 1.5 <= aspect_ratio <= 4.0:
                score += 30
            elif 1.0 <= aspect_ratio <= 1.5:
                score += 15
            
            # ====== CRITERION 4: Near baseline position (+50 points) ======
            # Real players are at TOP or BOTTOM of court (baselines)
            # Ball boys are at LEFT/RIGHT SIDES
            player_y = player_center[1]
            player_x = player_center[0]
            
            # Distance from horizontal center line (net)
            distance_from_net_line = abs(player_y - court_mid_y)
            court_half_height = (court_bounds['bottom'] - court_bounds['top']) / 2
            
            # Players at baseline have high Y-distance from net
            baseline_score = 50 * (distance_from_net_line / court_half_height)
            baseline_score = min(baseline_score, 50)  # Cap at 50
            score += baseline_score
            
            # ====== CRITERION 5: X position near center (+40 points) ======
            # Real players move along CENTER of court (left-right)
            # Ball boys are at EXTREME left or right edges
            court_center_x = (court_bounds['left'] + court_bounds['right']) / 2
            court_half_width = (court_bounds['right'] - court_bounds['left']) / 2
            distance_from_center_x = abs(player_x - court_center_x)
            
            # Lower distance from center X = higher score
            x_center_score = 40 * (1 - min(distance_from_center_x / court_half_width, 1))
            score += x_center_score
            
            # ====== CRITERION 6: Minimum size requirement (+20 points) ======
            if bbox_height > 80 and bbox_width > 30:
                score += 20
            
            # Determine if player is in top or bottom half
            is_bottom_half = player_y > court_mid_y
            
            candidates.append({
                'id': track_id,
                'score': score,
                'bbox': bbox,
                'center': player_center,
                'is_bottom_half': is_bottom_half
            })
        
        # Debug output
        print(f"  [PLAYER SELECTION V2] All candidates:")
        for c in candidates:
            half = "BOTTOM" if c['is_bottom_half'] else "TOP"
            print(f"    ID {c['id']}: score={c['score']:.1f}, half={half}, center={c['center']}")
        
        # ====== OPPONENT SEPARATION: Select one from each half ======
        bottom_candidates = [c for c in candidates if c['is_bottom_half']]
        top_candidates = [c for c in candidates if not c['is_bottom_half']]
        
        # Sort each group by score
        bottom_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        chosen_players = []
        
        # Best from bottom half (near player)
        if bottom_candidates:
            chosen_players.append(bottom_candidates[0]['id'])
            print(f"  [PLAYER SELECTION V2] Bottom half winner: ID {bottom_candidates[0]['id']} (score={bottom_candidates[0]['score']:.1f})")
        
        # Best from top half (far player)
        if top_candidates:
            chosen_players.append(top_candidates[0]['id'])
            print(f"  [PLAYER SELECTION V2] Top half winner: ID {top_candidates[0]['id']} (score={top_candidates[0]['score']:.1f})")
        
        # Fallback: if we don't have players in both halves, take top 2 overall
        if len(chosen_players) < 2:
            print(f"  [PLAYER SELECTION V2] WARNING: Using fallback - not enough separation")
            all_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
            chosen_players = [c['id'] for c in all_sorted[:2]]
        
        print(f"  [PLAYER SELECTION V2] Final chosen: {chosen_players}")
        return chosen_players
    
    def _estimate_court_bounds(self, court_keypoints):
        """
        Estimate the bounding box of the court from keypoints.
        
        Returns dict with: left, right, top, bottom, center, width, height
        """
        x_coords = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
        y_coords = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
        
        left = min(x_coords)
        right = max(x_coords)
        top = min(y_coords)
        bottom = max(y_coords)
        
        return {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'center': ((left + right) / 2, (top + bottom) / 2),
            'width': right - left,
            'height': bottom - top
        }
    
    def _is_inside_court(self, point, court_bounds, margin=0):
        """
        Check if a point is inside the court bounds (with optional margin).
        
        Args:
            point: (x, y) tuple
            court_bounds: dict from _estimate_court_bounds
            margin: pixels of margin outside court to still consider "inside"
        """
        x, y = point
        return (court_bounds['left'] - margin <= x <= court_bounds['right'] + margin and
                court_bounds['top'] - margin <= y <= court_bounds['bottom'] + margin)
    


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