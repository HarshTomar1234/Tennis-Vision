import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: path to the trained ResNet-50 keypoint regression weights.
            device:     torch device string. Defaults to CUDA when available -
                        keypoint detection runs once per frame and was the slowest
                        stage of the pipeline by a wide margin while pinned to CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
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
        
        bright_red = (0, 0, 255)  # BGR format: bright red
        
        # We'll modify frames in-place for memory efficiency
        for frame in video_frames:
            # Draw each keypoint with enhanced visibility using a multi-layer approach
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                
                # Drawing larger black outline for better contrast against any background
                cv2.circle(frame, (x, y), radius+2, (0, 0, 0), -1)
                
                # Drawing main colored circle (bright red for maximum visibility)
                cv2.circle(frame, (x, y), radius, bright_red, -1)
                
                # Adding a small white center dot for precision and professional appearance
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                
        return video_frames
    
    def predict_all_frames(self, video_frames, smooth=True, window_size=5):
        """
        Detect court keypoints for ALL frames in the video.
        
        This is crucial for handling camera motion - instead of detecting keypoints
        once (frame 0), we detect them every frame to handle camera shifts.
        
        Args:
            video_frames: List of video frames
            smooth: Whether to apply temporal smoothing (reduces jitter)
            window_size: Size of smoothing window (higher = smoother but more lag)
            
        Returns:
            List of keypoint arrays, one per frame
            
        Concept Explanation:
        -------------------
        When a camera moves, the court appears to move in the video. By detecting
        keypoints every frame, we track this "apparent motion" and can correctly
        map player/ball positions even when the camera is not perfectly static.
        """
        all_keypoints = []
        
        print(f"Detecting court keypoints for {len(video_frames)} frames...")
        
        for i, frame in enumerate(video_frames):
            keypoints = self.predict(frame)
            all_keypoints.append(keypoints)
            
            # Progress indicator every 100 frames
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(video_frames)} frames")
        
        if smooth:
            print("Applying temporal smoothing to keypoints...")
            all_keypoints = self.smooth_keypoints(all_keypoints, window_size)
        
        return all_keypoints
    
    def smooth_keypoints(self, keypoints_list, window_size=5):
        """
        Apply temporal smoothing to reduce jitter in keypoint detection.
        
        Uses a moving average filter to smooth keypoint positions across frames.
        This helps with:
        - Slight detection variations between frames
        - Minor camera vibrations
        - Model prediction noise
        
        Args:
            keypoints_list: List of keypoint arrays (one per frame)
            window_size: Number of frames to average (odd number recommended)
            
        Returns:
            Smoothed keypoint list
            
        Technical Note:
        --------------
        We use numpy's convolve with 'same' mode to maintain the same output length.
        Edge frames use partial windows (fewer frames for averaging).
        """
        if len(keypoints_list) < window_size:
            return keypoints_list
        
        # Convert to numpy array for efficient processing
        # Shape: (num_frames, num_keypoint_coords)
        keypoints_array = np.array(keypoints_list)
        smoothed_array = np.zeros_like(keypoints_array)
        
        # Create averaging kernel
        kernel = np.ones(window_size) / window_size
        
        # Smooth each keypoint coordinate independently
        num_coords = keypoints_array.shape[1]
        for coord_idx in range(num_coords):
            coord_series = keypoints_array[:, coord_idx]
            
            # Apply convolution for smoothing
            # 'same' mode keeps output length equal to input
            smoothed_coords = np.convolve(coord_series, kernel, mode='same')
            
            # Handle edge cases where we have fewer samples
            # First few frames: use progressively larger windows
            half_window = window_size // 2
            for i in range(half_window):
                start_idx = 0
                end_idx = i + half_window + 1
                smoothed_coords[i] = np.mean(coord_series[start_idx:end_idx])
            
            # Last few frames: use progressively smaller windows
            for i in range(len(coord_series) - half_window, len(coord_series)):
                start_idx = i - half_window
                end_idx = len(coord_series)
                smoothed_coords[i] = np.mean(coord_series[start_idx:end_idx])
            
            smoothed_array[:, coord_idx] = smoothed_coords
        
        # Convert back to list of arrays
        return [smoothed_array[i] for i in range(len(smoothed_array))]
    
    def draw_keypoints_on_video_dynamic(self, video_frames, all_keypoints, point_color=(0, 0, 255), radius=8):
        """
        Draw keypoints on video using per-frame keypoint detections.
        
        Unlike draw_keypoints_on_video which uses the same keypoints for all frames,
        this method uses different keypoints for each frame - essential for videos
        with camera motion.
        
        Args:
            video_frames: List of video frames to draw on
            all_keypoints: List of keypoint arrays (one per frame)
            point_color: Color to use for keypoints
            radius: Radius of keypoint circles
            
        Returns:
            Video frames with keypoints drawn (per-frame keypoints)
        """
        bright_red = (0, 0, 255)
        
        for frame_idx, frame in enumerate(video_frames):
            # Get keypoints for THIS specific frame
            if frame_idx < len(all_keypoints):
                keypoints = all_keypoints[frame_idx]
            else:
                # Fallback to last available keypoints
                keypoints = all_keypoints[-1]
            
            # Draw each keypoint
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                
                # Multi-layer approach for visibility
                cv2.circle(frame, (x, y), radius+2, (0, 0, 0), -1)  # Black outline
                cv2.circle(frame, (x, y), radius, bright_red, -1)   # Red fill
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)   # White center
                
        return video_frames