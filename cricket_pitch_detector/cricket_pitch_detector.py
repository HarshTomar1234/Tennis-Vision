import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CricketPitchDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        # Cricket pitch has different keypoints than tennis court
        # 16 keypoints: 4 corners of pitch, 4 crease points, 4 wicket points, 4 boundary points
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 16*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """Predict cricket pitch keypoints"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        
        # Scale keypoints back to original image size
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """Draw cricket pitch keypoints on image"""
        keypoint_labels = [
            "Pitch Corner 1", "Pitch Corner 2", "Pitch Corner 3", "Pitch Corner 4",
            "Striker Crease 1", "Striker Crease 2", "Bowler Crease 1", "Bowler Crease 2", 
            "Striker Wicket 1", "Striker Wicket 2", "Bowler Wicket 1", "Bowler Wicket 2",
            "Boundary 1", "Boundary 2", "Boundary 3", "Boundary 4"
        ]
        
        colors = [
            (255, 0, 0),   # Pitch corners - Blue
            (255, 0, 0), (255, 0, 0), (255, 0, 0),
            (0, 255, 0),   # Creases - Green  
            (0, 255, 0), (0, 255, 0), (0, 255, 0),
            (0, 0, 255),   # Wickets - Red
            (0, 0, 255), (0, 0, 255), (0, 0, 255),
            (255, 255, 0), # Boundaries - Cyan
            (255, 255, 0), (255, 255, 0), (255, 255, 0)
        ]
        
        for i in range(0, len(keypoints), 2):
            if i//2 < len(colors):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                color = colors[i//2]
                
                # Draw keypoint
                cv2.circle(image, (x, y), 8, color, -1)
                cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
                
                # Draw label
                if i//2 < len(keypoint_labels):
                    cv2.putText(image, str(i//2), (x-5, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints, point_color=(0, 255, 0), radius=6):
        """
        Draw cricket pitch keypoints on all video frames with enhanced visibility
        """
        for frame in video_frames:
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                
                # Draw larger outline for cricket pitch visibility
                cv2.circle(frame, (x, y), radius+3, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), radius, point_color, -1)
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
                
        return video_frames

    def draw_pitch_lines(self, image, keypoints):
        """Draw cricket pitch lines connecting the keypoints"""
        if len(keypoints) < 32:  # Need 16 points (32 coordinates)
            return image
            
        # Draw pitch boundary (rectangle connecting first 4 points)
        pitch_points = []
        for i in range(0, 8, 2):
            pitch_points.append((int(keypoints[i]), int(keypoints[i+1])))
        
        # Draw pitch rectangle
        for i in range(len(pitch_points)):
            start_point = pitch_points[i]
            end_point = pitch_points[(i+1) % len(pitch_points)]
            cv2.line(image, start_point, end_point, (255, 255, 255), 3)
        
        # Draw creases
        for i in range(8, 16, 4):  # Crease points
            if i+2 < len(keypoints):
                start = (int(keypoints[i]), int(keypoints[i+1]))
                end = (int(keypoints[i+2]), int(keypoints[i+3]))
                cv2.line(image, start, end, (0, 255, 0), 2)
        
        # Draw wickets
        for i in range(16, 24, 4):  # Wicket points
            if i+2 < len(keypoints):
                start = (int(keypoints[i]), int(keypoints[i+1]))
                end = (int(keypoints[i+2]), int(keypoints[i+3]))
                cv2.line(image, start, end, (0, 0, 255), 3)
        
        return image