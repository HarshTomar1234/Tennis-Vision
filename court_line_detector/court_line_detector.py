import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints, point_color=(0, 0, 255), radius=8):
        """
        Draw keypoints on all video frames with enhanced visibility and professional appearance
        
        Args:
            video_frames: List of video frames to draw on
            keypoints: Court keypoints to draw
            point_color: Color to use for keypoints (default: bright red)
            radius: Radius of keypoint circles (increased for better visibility)
            
        Returns:
            Video frames with prominent, highly visible keypoints drawn
        """
        # Always use bright red for maximum visibility of court keypoints
        # This is critical for professional presentation
        bright_red = (0, 0, 255)  # BGR format: bright red
        
        # We'll modify frames in-place for memory efficiency
        for frame in video_frames:
            # Draw each keypoint with enhanced visibility using a multi-layer approach
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                
                # Draw larger black outline for better contrast against any background
                cv2.circle(frame, (x, y), radius+2, (0, 0, 0), -1)
                
                # Draw main colored circle (bright red for maximum visibility)
                cv2.circle(frame, (x, y), radius, bright_red, -1)
                
                # Add a small white center dot for precision and professional appearance
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                
        return video_frames