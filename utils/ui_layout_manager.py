"""
UI Layout Manager for Tennis Vision
====================================

This module provides intelligent, dynamic positioning of UI elements
based on the video frame dimensions and court position.

Key Features:
- Automatic scaling based on frame resolution
- Collision-free placement of UI elements
- Responsive positioning that adapts to camera angle

Concept Explanation:
-------------------
Traditional approach: Hardcode positions like "stats at y=450"
Problem: Different videos have different resolutions and camera angles

Our approach: Calculate positions based on:
1. Frame dimensions (width, height)
2. Court keypoints position (where the court appears in frame)
3. UI element sizes (calculated as percentage of frame)
4. Collision avoidance (ensure nothing overlaps)
"""

import numpy as np


class UILayoutManager:
    """
    Manages the layout of all UI elements on the tennis video frame.
    
    This ensures that:
    - Mini court doesn't overlap with stats panels
    - Shot analysis doesn't overlap with player stats
    - Everything scales with video resolution
    - UI elements are positioned around the court, not on it
    """
    
    def __init__(self, frame_shape, court_keypoints=None):
        """
        Initialize the layout manager with frame dimensions.
        
        Args:
            frame_shape: Tuple of (height, width, channels)
            court_keypoints: Optional court keypoints to determine court position
        """
        self.frame_height = frame_shape[0]
        self.frame_width = frame_shape[1]
        
        # Store court bounds if keypoints provided
        self.court_bounds = None
        if court_keypoints is not None:
            self.court_bounds = self._calculate_court_bounds(court_keypoints)
        
        # Calculate base sizes as percentages of frame dimensions
        # These percentages work well across different resolutions
        self._calculate_base_dimensions()
        
        # Calculate all zone positions
        self._calculate_zones()
    
    def _calculate_base_dimensions(self):
        """Calculate base dimensions for UI elements as frame percentages."""
        # Mini court dimensions (about 15% of frame width)
        self.mini_court_width = int(self.frame_width * 0.15)
        self.mini_court_height = int(self.mini_court_width * 2.0)  # 2:1 aspect ratio for court
        
        # Ensure minimum sizes for readability
        self.mini_court_width = max(self.mini_court_width, 150)
        self.mini_court_height = max(self.mini_court_height, 300)
        
        # Stats panel dimensions
        self.stats_panel_width = int(self.frame_width * 0.25)
        self.stats_panel_height = int(self.frame_height * 0.35)
        
        # Ensure minimum sizes
        self.stats_panel_width = max(self.stats_panel_width, 250)
        self.stats_panel_height = max(self.stats_panel_height, 180)
        
        # Shot analysis panel dimensions
        self.shot_panel_width = int(self.frame_width * 0.2)
        self.shot_panel_height = int(self.frame_height * 0.25)
        
        # Padding between elements
        self.padding = int(min(self.frame_width, self.frame_height) * 0.02)
        self.padding = max(self.padding, 10)
    
    def _calculate_court_bounds(self, court_keypoints):
        """
        Calculate bounding box of the court from keypoints.
        
        Returns dict with keys: left, right, top, bottom
        """
        # Extract all x and y coordinates
        x_coords = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
        y_coords = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
        
        return {
            'left': min(x_coords),
            'right': max(x_coords),
            'top': min(y_coords),
            'bottom': max(y_coords)
        }
    
    def _calculate_zones(self):
        """
        Calculate positions for all UI zones.
        
        Layout strategy:
        - Mini court: Top-right corner
        - Player stats: Bottom-center (below court)
        - Shot analysis: Bottom-left
        - Shot type legend: Next to mini court
        """
        # ========== MINI COURT ZONE ==========
        # Position in top-right corner
        self.mini_court_zone = {
            'x': self.frame_width - self.mini_court_width - self.padding,
            'y': self.padding,
            'width': self.mini_court_width,
            'height': self.mini_court_height
        }
        
        # ========== PLAYER STATS ZONE ==========
        # Position at bottom center, but check for overlap with court
        stats_y = self.frame_height - self.stats_panel_height - self.padding
        
        # If court bounds available, ensure stats don't overlap court
        if self.court_bounds:
            # Stats should be below the court
            min_stats_y = int(self.court_bounds['bottom'] + self.padding)
            stats_y = max(stats_y, min_stats_y)
            
            # But also make sure it's visible in frame
            stats_y = min(stats_y, self.frame_height - self.stats_panel_height - self.padding)
        
        # Left-aligned, not centred: the near player stands and plays in the
        # bottom-centre of a broadcast frame, so a centred panel sits directly on
        # top of the action it is describing. The bottom-left corner is over the
        # doubles alley, which is empty for most of a rally.
        self.stats_zone = {
            'x': self.padding,
            'y': stats_y,
            'width': self.stats_panel_width,
            'height': self.stats_panel_height
        }
        
        # ========== SHOT ANALYSIS ZONE ==========
        # Position at bottom-left, not overlapping with stats
        self.shot_analysis_zone = {
            'x': self.padding,
            'y': self.frame_height - self.shot_panel_height - self.padding,
            'width': self.shot_panel_width,
            'height': self.shot_panel_height
        }
        
        # Check for overlap with stats panel
        if self._zones_overlap(self.shot_analysis_zone, self.stats_zone):
            # Move shot analysis above stats
            self.shot_analysis_zone['y'] = self.stats_zone['y'] - self.shot_panel_height - self.padding
        
        # ========== SHOT TYPE LEGEND ZONE ==========
        # Position below mini court
        legend_height = 120
        self.legend_zone = {
            'x': self.mini_court_zone['x'],
            'y': self.mini_court_zone['y'] + self.mini_court_zone['height'] + self.padding,
            'width': self.mini_court_width,
            'height': legend_height
        }
    
    def _zones_overlap(self, zone1, zone2):
        """Check if two rectangular zones overlap."""
        # Zone format: {'x', 'y', 'width', 'height'}
        left1 = zone1['x']
        right1 = zone1['x'] + zone1['width']
        top1 = zone1['y']
        bottom1 = zone1['y'] + zone1['height']
        
        left2 = zone2['x']
        right2 = zone2['x'] + zone2['width']
        top2 = zone2['y']
        bottom2 = zone2['y'] + zone2['height']
        
        # Check for no overlap conditions
        if right1 < left2 or right2 < left1:
            return False
        if bottom1 < top2 or bottom2 < top1:
            return False
        
        return True
    
    def get_mini_court_params(self):
        """
        Get parameters for mini court placement.
        
        Returns:
            dict with keys: start_x, start_y, end_x, end_y, width, height
        """
        zone = self.mini_court_zone
        return {
            'start_x': zone['x'],
            'start_y': zone['y'],
            'end_x': zone['x'] + zone['width'],
            'end_y': zone['y'] + zone['height'],
            'width': zone['width'],
            'height': zone['height']
        }
    
    def get_stats_panel_params(self):
        """
        Get parameters for player stats panel placement.
        
        Returns:
            dict with keys: start_x, start_y, end_x, end_y, width, height
        """
        zone = self.stats_zone
        return {
            'start_x': zone['x'],
            'start_y': zone['y'],
            'end_x': zone['x'] + zone['width'],
            'end_y': zone['y'] + zone['height'],
            'width': zone['width'],
            'height': zone['height']
        }
    
    def get_shot_analysis_params(self):
        """
        Get parameters for shot analysis panel placement.
        
        Returns:
            dict with keys: start_x, start_y, end_x, end_y, width, height
        """
        zone = self.shot_analysis_zone
        return {
            'start_x': zone['x'],
            'start_y': zone['y'],
            'end_x': zone['x'] + zone['width'],
            'end_y': zone['y'] + zone['height'],
            'width': zone['width'],
            'height': zone['height']
        }
    
    def get_legend_params(self):
        """
        Get parameters for shot type legend placement.
        
        Returns:
            dict with keys: start_x, start_y, end_x, end_y, width, height
        """
        zone = self.legend_zone
        return {
            'start_x': zone['x'],
            'start_y': zone['y'],
            'end_x': zone['x'] + zone['width'],
            'end_y': zone['y'] + zone['height'],
            'width': zone['width'],
            'height': zone['height']
        }
    
    def get_all_zones_debug(self):
        """
        Get all zones for debugging/visualization.
        
        Returns:
            dict of all zones with their parameters
        """
        return {
            'mini_court': self.mini_court_zone,
            'stats': self.stats_zone,
            'shot_analysis': self.shot_analysis_zone,
            'legend': self.legend_zone
        }
    
    def update_with_keypoints(self, court_keypoints):
        """
        Update layout based on new court keypoints.
        
        Call this when keypoints change (new frame with camera motion).
        Only recalculates zones that depend on court position.
        """
        self.court_bounds = self._calculate_court_bounds(court_keypoints)
        self._calculate_zones()  # Recalculate zones with new bounds


def create_layout_for_frame(frame, court_keypoints=None):
    """
    Convenience function to create a layout manager for a frame.
    
    Args:
        frame: Video frame (numpy array)
        court_keypoints: Optional court keypoints
        
    Returns:
        UILayoutManager instance
    """
    return UILayoutManager(frame.shape, court_keypoints)
