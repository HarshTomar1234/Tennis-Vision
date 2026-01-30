"""
PoseTracker: BBoxMaskPose Integration for Tennis-Vision

This module wraps the BBoxMaskPose library to provide human pose estimation
for tennis player analysis. It supports multiple pose estimation models
and provides keypoint data for shot classification and technique analysis.

Author: Harsh Tomar
Date: January 2026
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

# Add BBoxMaskPose to path
BBOXMASKPOSE_PATH = Path(__file__).parent.parent / "external" / "BBoxMaskPose"
sys.path.insert(0, str(BBOXMASKPOSE_PATH))
sys.path.insert(0, str(BBOXMASKPOSE_PATH / "demo"))

# COCO keypoint definitions (17 keypoints)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint indices for easy access
KEYPOINT_INDICES = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Color scheme for skeleton parts (BGR)
SKELETON_COLORS = {
    'head': (255, 200, 100),      # Light blue
    'left_arm': (255, 0, 0),      # Blue
    'right_arm': (0, 0, 255),     # Red
    'torso': (0, 165, 255),       # Orange
    'left_leg': (0, 255, 0),      # Green
    'right_leg': (0, 255, 255),   # Yellow
}

# Model configurations
MODEL_CONFIGS = {
    'vitpose-b': {
        'name': 'ViTPose-b',
        'description': 'Fast and lightweight, good for tennis',
        'pose_config': 'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192.py',
        'pose_checkpoint': 'https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/ViTPose-b-multi_mmpose20.pth',
    },
    'maskpose-b': {
        'name': 'MaskPose-b',
        'description': 'Mask-conditioned pose, better for occlusions',
        'pose_config': 'mmpose/configs/MaskPose/ViTb-multi_mask.py',
        'pose_checkpoint': 'https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose-b.pth',
    },
    'pmpose': {
        'name': 'PMPose (BMP-v2)',
        'description': 'Cutting-edge, best accuracy',
        'pose_config': None,  # Will be available in BMP-v2 branch
        'pose_checkpoint': None,
    },
}


class PoseTracker:
    """
    Wrapper for BBoxMaskPose pose estimation in tennis context.
    
    Provides human pose estimation for tennis players, extracting 17 COCO
    keypoints per player per frame. Supports multiple model backends.
    
    Usage:
        pose_tracker = PoseTracker(model='vitpose-b')
        player_poses = pose_tracker.detect_poses(frames, player_detections)
        output_frames = pose_tracker.draw_skeletons(frames, player_poses)
    """
    
    def __init__(
        self,
        model: str = 'vitpose-b',
        device: str = 'cuda:0',
        confidence_threshold: float = 0.3,
        use_sam_refinement: bool = False,
    ):
        """
        Initialize PoseTracker with specified model.
        
        Args:
            model: Model to use ('vitpose-b', 'maskpose-b', 'pmpose')
            device: Inference device ('cuda:0', 'cpu')
            confidence_threshold: Minimum keypoint confidence to keep
            use_sam_refinement: Whether to use SAM2 mask refinement (slower)
        """
        self.model_name = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_sam_refinement = use_sam_refinement
        
        # Model references (lazy loading)
        self._pose_estimator = None
        self._detector = None
        self._sam_model = None
        self._initialized = False
        
        # Get model config
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_CONFIGS.keys())}")
        self.config = MODEL_CONFIGS[model]
        
        print(f"[PoseTracker] Initialized with {self.config['name']}")
        print(f"[PoseTracker] Device: {device}, Confidence threshold: {confidence_threshold}")
    
    def _lazy_init(self):
        """Lazy initialization of models (on first use)."""
        if self._initialized:
            return
        
        try:
            from mmpose.apis import init_model as init_pose_estimator
            from mmdet.apis import init_detector
            from mmpose.utils import adapt_mmdet_pipeline
            
            # Resolve config paths relative to BBoxMaskPose
            pose_config = str(BBOXMASKPOSE_PATH / self.config['pose_config'])
            pose_checkpoint = self.config['pose_checkpoint']
            
            # Initialize pose estimator
            print(f"[PoseTracker] Loading {self.config['name']}...")
            self._pose_estimator = init_pose_estimator(
                pose_config,
                pose_checkpoint,
                device=self.device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
            )
            
            self._initialized = True
            print(f"[PoseTracker] Model loaded successfully!")
            
        except ImportError as e:
            print(f"[PoseTracker] ERROR: Missing dependencies - {e}")
            print("[PoseTracker] Please install: pip install mmpose mmdet mmengine")
            raise
        except Exception as e:
            print(f"[PoseTracker] ERROR: Failed to load model - {e}")
            raise
    
    def detect_poses(
        self,
        frames: List[np.ndarray],
        player_detections: List[Dict[int, List[float]]],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[Dict[int, np.ndarray]]:
        """
        Detect poses for all players in all frames.
        
        Args:
            frames: List of video frames (BGR format)
            player_detections: List of dicts mapping player_id -> bbox [x1,y1,x2,y2]
            read_from_stub: If True, load from pickle file
            stub_path: Path to pickle file for caching
            
        Returns:
            List of dicts mapping player_id -> keypoints array (17, 3)
            Where each keypoint is (x, y, confidence)
        """
        import pickle
        
        # Load from stub if available
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[PoseTracker] Loading poses from {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        # Initialize models on first use
        self._lazy_init()
        
        all_poses = []
        total_frames = len(frames)
        
        print(f"[PoseTracker] Processing {total_frames} frames...")
        
        for frame_idx, (frame, player_dict) in enumerate(zip(frames, player_detections)):
            frame_poses = {}
            
            for player_id, bbox in player_dict.items():
                if bbox and len(bbox) == 4:
                    keypoints = self._estimate_pose_for_bbox(frame, bbox)
                    if keypoints is not None:
                        frame_poses[player_id] = keypoints
            
            all_poses.append(frame_poses)
            
            # Progress indicator
            if (frame_idx + 1) % 50 == 0:
                print(f"[PoseTracker] Processed {frame_idx + 1}/{total_frames} frames")
        
        print(f"[PoseTracker] Pose detection complete!")
        
        # Save to stub if path provided
        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(all_poses, f)
            print(f"[PoseTracker] Saved poses to {stub_path}")
        
        return all_poses
    
    def _estimate_pose_for_bbox(
        self,
        frame: np.ndarray,
        bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Estimate pose for a single bounding box.
        
        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Keypoints array (17, 3) or None if failed
        """
        try:
            from mmpose.apis import inference_topdown
            from mmpose.structures import PoseDataSample
            from mmengine.structures import InstanceData
            
            # Create detection result for MMPose
            x1, y1, x2, y2 = bbox
            instance = InstanceData()
            instance.bboxes = np.array([[x1, y1, x2, y2]])
            instance.bbox_scores = np.array([1.0])
            
            # Run pose estimation
            results = inference_topdown(
                self._pose_estimator,
                frame,
                bboxes=instance.bboxes,
                bbox_format='xyxy',
            )
            
            if results and len(results) > 0:
                # Extract keypoints (17, 3) - x, y, score
                pred_instances = results[0].pred_instances
                keypoints = pred_instances.keypoints[0]  # (17, 2)
                scores = pred_instances.keypoint_scores[0]  # (17,)
                
                # Combine into (17, 3)
                keypoints_with_scores = np.concatenate(
                    [keypoints, scores[:, None]], axis=-1
                )
                
                return keypoints_with_scores
            
            return None
            
        except Exception as e:
            # Silently fail for individual detections
            return None
    
    def draw_skeletons(
        self,
        frames: List[np.ndarray],
        player_poses: List[Dict[int, np.ndarray]],
        thickness: int = 2,
        radius: int = 4,
    ) -> List[np.ndarray]:
        """
        Draw pose skeletons on video frames.
        
        Args:
            frames: List of video frames
            player_poses: Pose data from detect_poses()
            thickness: Line thickness for skeleton
            radius: Keypoint circle radius
            
        Returns:
            Frames with skeletons drawn
        """
        output_frames = []
        
        for frame, poses in zip(frames, player_poses):
            frame_copy = frame.copy()
            
            for player_id, keypoints in poses.items():
                self._draw_single_skeleton(
                    frame_copy, keypoints, player_id, thickness, radius
                )
            
            output_frames.append(frame_copy)
        
        return output_frames
    
    def _draw_single_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        player_id: int,
        thickness: int = 2,
        radius: int = 4,
    ):
        """Draw skeleton for a single player on frame."""
        # Player-specific color offset
        color_offset = 50 if player_id == 2 else 0
        
        # Draw skeleton connections
        for connection in SKELETON_CONNECTIONS:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # Only draw if both keypoints are confident
            if pt1[2] > self.confidence_threshold and pt2[2] > self.confidence_threshold:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                # Determine line color based on body part
                color = self._get_connection_color(connection, color_offset)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw keypoints
        for idx, kpt in enumerate(keypoints):
            if kpt[2] > self.confidence_threshold:
                x, y = int(kpt[0]), int(kpt[1])
                color = self._get_keypoint_color(idx, color_offset)
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)  # White border
    
    def _get_connection_color(self, connection: Tuple[int, int], offset: int = 0) -> Tuple[int, int, int]:
        """Get color for skeleton connection."""
        pt1_idx, pt2_idx = connection
        
        # Head connections
        if pt1_idx in [0, 1, 2] or pt2_idx in [1, 2, 3, 4]:
            color = SKELETON_COLORS['head']
        # Left arm
        elif pt1_idx in [5, 7] and pt2_idx in [7, 9]:
            color = SKELETON_COLORS['left_arm']
        # Right arm
        elif pt1_idx in [6, 8] and pt2_idx in [8, 10]:
            color = SKELETON_COLORS['right_arm']
        # Torso
        elif (pt1_idx in [5, 6, 11] and pt2_idx in [6, 11, 12]):
            color = SKELETON_COLORS['torso']
        # Left leg
        elif pt1_idx in [11, 13] and pt2_idx in [13, 15]:
            color = SKELETON_COLORS['left_leg']
        # Right leg
        elif pt1_idx in [12, 14] and pt2_idx in [14, 16]:
            color = SKELETON_COLORS['right_leg']
        else:
            color = (200, 200, 200)  # Gray default
        
        # Apply offset for player differentiation
        return tuple(min(255, c + offset) for c in color)
    
    def _get_keypoint_color(self, idx: int, offset: int = 0) -> Tuple[int, int, int]:
        """Get color for keypoint based on body part."""
        if idx < 5:  # Head
            color = SKELETON_COLORS['head']
        elif idx in [5, 7, 9]:  # Left arm
            color = SKELETON_COLORS['left_arm']
        elif idx in [6, 8, 10]:  # Right arm
            color = SKELETON_COLORS['right_arm']
        elif idx in [11, 12]:  # Hips
            color = SKELETON_COLORS['torso']
        elif idx in [13, 15]:  # Left leg
            color = SKELETON_COLORS['left_leg']
        elif idx in [14, 16]:  # Right leg
            color = SKELETON_COLORS['right_leg']
        else:
            color = (200, 200, 200)
        
        return tuple(min(255, c + offset) for c in color)
    
    def get_keypoint(
        self,
        poses: Dict[int, np.ndarray],
        player_id: int,
        keypoint_name: str,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get specific keypoint for a player.
        
        Args:
            poses: Pose dict for a frame
            player_id: Player ID
            keypoint_name: Name of keypoint (e.g., 'left_wrist')
            
        Returns:
            Tuple of (x, y, confidence) or None if not found
        """
        if player_id not in poses:
            return None
        
        if keypoint_name not in KEYPOINT_INDICES:
            return None
        
        idx = KEYPOINT_INDICES[keypoint_name]
        kpt = poses[player_id][idx]
        return (kpt[0], kpt[1], kpt[2])
    
    def get_arm_angle(
        self,
        poses: Dict[int, np.ndarray],
        player_id: int,
        arm: str = 'right',
    ) -> Optional[float]:
        """
        Calculate arm angle (shoulder-elbow-wrist angle).
        
        Useful for detecting serve motion, forehand/backhand position.
        
        Args:
            poses: Pose dict for a frame
            player_id: Player ID
            arm: 'left' or 'right'
            
        Returns:
            Angle in degrees (0-180) or None if keypoints not visible
        """
        if player_id not in poses:
            return None
        
        keypoints = poses[player_id]
        
        # Get arm keypoint indices
        if arm == 'right':
            shoulder_idx = KEYPOINT_INDICES['right_shoulder']
            elbow_idx = KEYPOINT_INDICES['right_elbow']
            wrist_idx = KEYPOINT_INDICES['right_wrist']
        else:
            shoulder_idx = KEYPOINT_INDICES['left_shoulder']
            elbow_idx = KEYPOINT_INDICES['left_elbow']
            wrist_idx = KEYPOINT_INDICES['left_wrist']
        
        # Check confidence
        if (keypoints[shoulder_idx][2] < self.confidence_threshold or
            keypoints[elbow_idx][2] < self.confidence_threshold or
            keypoints[wrist_idx][2] < self.confidence_threshold):
            return None
        
        # Calculate vectors
        shoulder = keypoints[shoulder_idx][:2]
        elbow = keypoints[elbow_idx][:2]
        wrist = keypoints[wrist_idx][:2]
        
        vec1 = shoulder - elbow
        vec2 = wrist - elbow
        
        # Calculate angle using dot product
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def is_arm_overhead(
        self,
        poses: Dict[int, np.ndarray],
        player_id: int,
        arm: str = 'right',
    ) -> bool:
        """
        Check if player's arm is in overhead position (serve/smash).
        
        Args:
            poses: Pose dict for a frame
            player_id: Player ID
            arm: 'left' or 'right'
            
        Returns:
            True if arm is overhead
        """
        if player_id not in poses:
            return False
        
        keypoints = poses[player_id]
        
        # Get keypoint indices
        if arm == 'right':
            wrist_idx = KEYPOINT_INDICES['right_wrist']
            shoulder_idx = KEYPOINT_INDICES['right_shoulder']
        else:
            wrist_idx = KEYPOINT_INDICES['left_wrist']
            shoulder_idx = KEYPOINT_INDICES['left_shoulder']
        
        head_idx = KEYPOINT_INDICES['nose']
        
        # Check confidence
        if (keypoints[wrist_idx][2] < self.confidence_threshold or
            keypoints[head_idx][2] < self.confidence_threshold):
            return False
        
        # Overhead: wrist Y is above (smaller than) head Y
        wrist_y = keypoints[wrist_idx][1]
        head_y = keypoints[head_idx][1]
        
        return wrist_y < head_y
    
    def export_pose_data(
        self,
        player_poses: List[Dict[int, np.ndarray]],
        output_path: str,
        format: str = 'csv',
    ) -> str:
        """
        Export pose data to file for external analysis.
        
        Args:
            player_poses: Pose data from detect_poses()
            output_path: Path to output file
            format: 'csv' or 'json'
            
        Returns:
            Path to saved file
        """
        import pandas as pd
        import json
        
        rows = []
        for frame_idx, poses in enumerate(player_poses):
            for player_id, keypoints in poses.items():
                row = {
                    'frame': frame_idx,
                    'player_id': player_id,
                }
                
                for kpt_idx, kpt_name in enumerate(COCO_KEYPOINTS):
                    row[f'{kpt_name}_x'] = keypoints[kpt_idx][0]
                    row[f'{kpt_name}_y'] = keypoints[kpt_idx][1]
                    row[f'{kpt_name}_conf'] = keypoints[kpt_idx][2]
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if format == 'csv':
            output_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
            df.to_csv(output_path, index=False)
        elif format == 'json':
            output_path = output_path if output_path.endswith('.json') else output_path + '.json'
            df.to_json(output_path, orient='records', indent=2)
        
        print(f"[PoseTracker] Exported pose data to {output_path}")
        return output_path
