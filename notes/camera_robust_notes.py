# =====================================================================
#                  CAMERA-ROBUST TENNIS VISION SYSTEM
#                       Technical Notes & Guide
# =====================================================================
# Author: Harsh Tomar (with AI assistance)
# Date: December 2024
# Version: 2.0 - Camera Motion Robust Edition
# =====================================================================

"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎾 TENNIS VISION: CAMERA MOTION ROBUSTNESS                        ║
║                                                                      ║
║   This file contains comprehensive technical notes explaining        ║
║   how we handle camera motion in tennis video analysis.             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

TABLE OF CONTENTS:
==================
1. THE CORE PROBLEM: Why Camera Position Breaks Everything
2. SOLUTION 1: Per-Frame Keypoint Detection
3. SOLUTION 2: Temporal Smoothing Algorithm
4. SOLUTION 3: Dynamic UI Layout System
5. SOLUTION 4: Multi-Criteria Player Selection
6. HOMOGRAPHY: The Math Behind Court Mapping
7. IMPLEMENTATION DETAILS: Files & Functions
8. HOW TO RUN AND TEST
9. SPEED OPTIMIZATION STRATEGIES (Future Work)
10. TROUBLESHOOTING GUIDE
"""


# =====================================================================
# SECTION 1: THE CORE PROBLEM
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 1: THE CORE PROBLEM - Why Camera Position Breaks Everything █
█████████████████████████████████████████████████████████████████████████

UNDERSTANDING THE ISSUE:
========================

When analyzing tennis footage, our system needs to:
    1. Detect where the court is (via keypoints - corners and line intersections)
    2. Detect where players and ball are (via bounding boxes)
    3. Map their pixel positions to real-world court coordinates
    4. Visualize everything on a mini court overlay

THE ASSUMPTION THAT BREAKS:
===========================

The original system made a CRITICAL assumption:
    
    "The camera is PERFECTLY STATIC throughout the entire video"

This meant we only detected court keypoints ONCE (on frame 0) and used those
same keypoints for ALL subsequent frames.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ORIGINAL CODE (main.py, line ~45):                             ║
    ║                                                                  ║
    ║  court_keypoints = court_line_detector.predict(video_frames[0]) ║
    ║                                                                  ║
    ║  # ^ This was the ONLY keypoint detection!                      ║
    ╚══════════════════════════════════════════════════════════════════╝

WHY THIS FAILS:
===============

    Frame 0:   Camera at position A → Keypoints detected correctly
    Frame 100: Camera shifts 5 pixels → Keypoints are now 5 pixels off!
    Frame 500: Camera vibration → Court appears to "jitter" in analysis
    Frame 1000: Cameraman adjusts → Complete mapping failure!

Even TINY camera movements (due to vibration, zoom adjustments, or tripod 
settling) cause the coordinate transformation to produce wrong results.

VISUAL REPRESENTATION:
======================

    STATIC CAMERA (Works):          MOVING CAMERA (Breaks):
    ┌─────────────────┐             ┌─────────────────┐
    │  Court          │             │    Court        │  ← Court appears
    │  ╔═══════════╗  │             │  ╔═══════════╗  │    to have moved
    │  ║           ║  │             │  ║           ║  │
    │  ║   Court   ║  │             │  ║   Court   ║  │
    │  ║           ║  │             │  ║           ║  │
    │  ╚═══════════╝  │             │  ╚═══════════╝  │
    └─────────────────┘             └─────────────────┘
    ^ Same position                 ^ Different position!
    
    Keypoints from Frame 0          Keypoints from Frame 0 don't match!
    still valid  ✓                  Player appears in wrong mini court spot ✗

CHAIN REACTION OF PROBLEMS:
===========================

    Camera Moves
        │
        ▼
    Court Keypoints Drift from Reality
        │
        ├──► Wrong Player Position on Mini Court
        │
        ├──► Wrong Ball Position on Mini Court
        │
        ├──► Incorrect Speed Calculations (distance / time)
        │
        └──► Complete Analysis Failure

"""


# =====================================================================
# SECTION 2: PER-FRAME KEYPOINT DETECTION
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 2: PER-FRAME KEYPOINT DETECTION                              █
█████████████████████████████████████████████████████████████████████████

THE SOLUTION:
=============

Instead of detecting keypoints ONCE, detect them for EVERY FRAME.

    ╔════════════════════════════════════════════════════════════════════╗
    ║  NEW APPROACH:                                                     ║
    ║                                                                    ║
    ║  all_keypoints = court_line_detector.predict_all_frames(frames)   ║
    ║                                                                    ║
    ║  # This returns a LIST of keypoints, one per frame!              ║
    ╚════════════════════════════════════════════════════════════════════╝

ALGORITHM (PSEUDO-CODE):
========================

    def predict_all_frames(video_frames):
        all_keypoints = []
        
        for frame in video_frames:
            # Run the keypoint detection neural network on THIS frame
            keypoints = neural_network.predict(frame)
            all_keypoints.append(keypoints)
        
        # Optionally smooth to reduce jitter (see Section 3)
        all_keypoints = smooth_keypoints(all_keypoints)
        
        return all_keypoints

IMPLEMENTATION (court_line_detector.py):
========================================
"""

def predict_all_frames_algorithm(video_frames, smooth=True, window_size=5):
    """
    Detect court keypoints for ALL frames in the video.
    
    This is crucial for handling camera motion - instead of detecting keypoints
    once (frame 0), we detect them every frame to handle camera shifts.
    
    PARAMETERS:
    -----------
    video_frames : list of numpy arrays
        All video frames to process
    smooth : bool
        Whether to apply temporal smoothing (reduces jitter)
    window_size : int
        Size of smoothing window (higher = smoother but more lag)
        
    RETURNS:
    --------
    list of numpy arrays
        One keypoint array per frame
    """
    all_keypoints = []
    
    print(f"Detecting court keypoints for {len(video_frames)} frames...")
    
    for i, frame in enumerate(video_frames):
        # Run neural network prediction on this frame
        keypoints = predict_single_frame(frame)  # ResNet-50 backbone
        all_keypoints.append(keypoints)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(video_frames)} frames")
    
    if smooth:
        print("Applying temporal smoothing to keypoints...")
        all_keypoints = smooth_keypoints(all_keypoints, window_size)
    
    return all_keypoints

"""
TRADE-OFF ANALYSIS:
===================

    ┌────────────────────┬──────────────────┬──────────────────────┐
    │ Aspect             │ Per-Frame        │ Single-Frame         │
    ├────────────────────┼──────────────────┼──────────────────────┤
    │ Accuracy           │ ✅ Excellent      │ ❌ Breaks with motion │
    │ Speed              │ ❌ ~5x slower     │ ✅ Very fast          │
    │ Memory             │ ⚠️ Higher         │ ✅ Minimal            │
    │ Camera motion      │ ✅ Handled        │ ❌ Not handled        │
    │ Jitter (raw)       │ ⚠️ Present        │ ❌ None (but wrong!)  │
    │ Jitter (smoothed)  │ ✅ Minimal        │ N/A                  │
    └────────────────────┴──────────────────┴──────────────────────┘

TOGGLE FLAG:
============

In main.py, you can switch between modes:

    ENABLE_PER_FRAME_KEYPOINTS = True   # Camera-robust (slower)
    ENABLE_PER_FRAME_KEYPOINTS = False  # Fast mode (original)

"""


# =====================================================================
# SECTION 3: TEMPORAL SMOOTHING ALGORITHM
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 3: TEMPORAL SMOOTHING ALGORITHM                              █
█████████████████████████████████████████████████████████████████████████

THE PROBLEM:
============

Even with per-frame detection, you get JITTER (tiny random variations):

    Frame 1: keypoint X = 100.0
    Frame 2: keypoint X = 101.2   ← Slight variation
    Frame 3: keypoint X = 99.5    ← More variation
    Frame 4: keypoint X = 100.8
    Frame 5: keypoint X = 98.9

This causes the mini court visualization to "shake" or "vibrate".

    Why does jitter happen?
    - Neural network outputs vary slightly between frames
    - Pixel-level ambiguity at court line edges
    - Compression artifacts in video
    - Lighting changes

THE SOLUTION: MOVING AVERAGE FILTER
====================================

Smooth the keypoints over a window of consecutive frames:

    new_value[frame] = average(values[frame - window/2 : frame + window/2])

ALGORITHM VISUALIZATION:
========================

    Window size = 5 (we look at 2 frames before and 2 after)
    
    Raw values:    [100, 101, 99, 102, 98, 100, 103, 97, 101, 99]
                         ↓
    For frame 3:   average([101, 99, 102, 98, 100]) = 100.0
    For frame 4:   average([99, 102, 98, 100, 103]) = 100.4
    For frame 5:   average([102, 98, 100, 103, 97]) = 100.0
                         ↓
    Smoothed:      [~100, ~100, ~100, ~100, ~100, ~100, ~100, ~100, ~100, ~100]
                    ↑
                    Much more stable!

MATHEMATICAL FORMULATION:
=========================

    Let K[t] = keypoint value at time t
    Let w = window_size (e.g., 5)
    Let h = w // 2 (half window, e.g., 2)
    
    Smoothed[t] = (1/w) * Σ K[i]  for i from (t-h) to (t+h)
    
    Or using convolution:
    
    Smoothed = K ⊛ [1/w, 1/w, 1/w, 1/w, 1/w]
             = convolve(K, ones(w) / w)

IMPLEMENTATION:
===============
"""

import numpy as np

def smooth_keypoints_algorithm(keypoints_list, window_size=5):
    """
    Apply temporal smoothing to reduce jitter in keypoint detection.
    
    Uses a moving average filter to smooth keypoint positions across frames.
    
    PARAMETERS:
    -----------
    keypoints_list : list of numpy arrays
        Raw keypoints for each frame
    window_size : int
        Number of frames to average (odd number recommended)
        
    RETURNS:
    --------
    list of numpy arrays
        Smoothed keypoints
        
    ALGORITHM:
    ----------
    1. Convert list to 2D array (frames x coordinates)
    2. For each coordinate, apply 1D convolution with averaging kernel
    3. Handle edge cases (first/last few frames)
    4. Convert back to list format
    """
    if len(keypoints_list) < window_size:
        return keypoints_list  # Not enough frames to smooth
    
    # Convert to numpy array: shape = (num_frames, num_coordinates)
    # Example: 500 frames, 28 coordinates → (500, 28)
    keypoints_array = np.array(keypoints_list)
    smoothed_array = np.zeros_like(keypoints_array)
    
    # Create averaging kernel: [0.2, 0.2, 0.2, 0.2, 0.2] for window_size=5
    kernel = np.ones(window_size) / window_size
    
    # Smooth each coordinate independently
    num_coords = keypoints_array.shape[1]
    for coord_idx in range(num_coords):
        coord_series = keypoints_array[:, coord_idx]
        
        # Apply convolution (this is the core smoothing operation)
        smoothed_coords = np.convolve(coord_series, kernel, mode='same')
        
        # Handle edge cases where we have fewer samples
        # First few frames: use smaller windows
        half_window = window_size // 2
        for i in range(half_window):
            smoothed_coords[i] = np.mean(coord_series[0:i + half_window + 1])
        
        # Last few frames: use smaller windows
        for i in range(len(coord_series) - half_window, len(coord_series)):
            smoothed_coords[i] = np.mean(coord_series[i - half_window:])
        
        smoothed_array[:, coord_idx] = smoothed_coords
    
    # Convert back to list of arrays
    return [smoothed_array[i] for i in range(len(smoothed_array))]

"""
PARAMETER TUNING GUIDE:
=======================

    window_size = 3:  Minimal smoothing, very responsive
                      Good for: Handheld cameras with rapid movement
                      
    window_size = 5:  ★ RECOMMENDED ★
                      Good balance between smoothing and responsiveness
                      
    window_size = 7:  More smoothing, slower response
                      Good for: Tripod cameras with occasional bumps
                      
    window_size = 11: Heavy smoothing
                      Good for: Very noisy keypoint detection
                      Risk: May smooth out actual camera movements

VISUAL COMPARISON:
==================

    Raw (no smoothing):
    ───╱╲─╱╲──╱╲╱╲───╱╲╱╲──╱╲───   (jittery)
    
    window_size = 3:
    ───╱╲──╱╲───╱╲────╱╲────╱╲──   (some jitter remains)
    
    window_size = 5:
    ────────────────────────────   (smooth, stable) ★
    
    window_size = 11:
    ─────────────────────────__    (very smooth, might miss quick changes)

"""


# =====================================================================
# SECTION 4: DYNAMIC UI LAYOUT SYSTEM
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 4: DYNAMIC UI LAYOUT SYSTEM                                  █
█████████████████████████████████████████████████████████████████████████

THE PROBLEM:
============

Original code had HARDCODED UI positions:

    # In player_stats_drawer_utils.py:
    start_y = 450  # Always at pixel 450!
    width = 350    # Always 350 pixels!
    
    # In mini_court.py:
    self.drawing_rectangle_width = 250   # Always 250 pixels!
    self.drawing_rectangle_height = 500  # Always 500 pixels!

This causes problems when:
    - Video has different resolution (720p vs 1080p vs 4K)
    - Camera angle shows more/less of the court
    - Different broadcast formats
    - Mini court overlaps with stats panel

EXAMPLE OF FAILURE:
==================

    1080p Video (1920 x 1080):
    ┌────────────────────────────────────────┐
    │                                        │
    │           Court View                   │
    │                                        │
    │                             ┌──────┐   │ ← Mini court at right
    │                             │ Mini │   │
    │  ┌────────────┐             │ Court│   │
    │  │Player Stats│ ← At y=450  └──────┘   │
    │  └────────────┘                        │
    └────────────────────────────────────────┘
    Looks fine! ✓

    720p Video (1280 x 720):
    ┌──────────────────────────────┐
    │                              │
    │        Court View            │
    │                     ┌──────┐ │
    │  ┌────────────┐     │ Mini │ │ ← OVERLAPPING!
    │  │Player Stats│     │ Court│ │
    │  └────────────┘     └──────┘ │
    │  │Player Stats│ ← y=450 is  │
    │  │  (cut off) │   below     │
    │                  the frame!  │
    └──────────────────────────────┘
    Broken! ✗

THE SOLUTION: PERCENTAGE-BASED LAYOUT
=====================================

Calculate all sizes and positions as PERCENTAGES of frame dimensions:

    mini_court_width = frame_width * 0.15   # 15% of width
    mini_court_height = mini_court_width * 2.0  # 2:1 aspect ratio
    
    stats_panel_y = frame_height * 0.65   # At 65% down the frame

IMPLEMENTATION (ui_layout_manager.py):
======================================
"""

class UILayoutManagerExplained:
    """
    Manages the layout of all UI elements on the tennis video frame.
    
    DESIGN PRINCIPLES:
    ==================
    1. All sizes are percentages of frame dimensions
    2. Positions are calculated relative to frame, not absolute pixels
    3. Collision detection prevents overlapping elements
    4. Minimum sizes ensure readability at any resolution
    """
    
    def __init__(self, frame_shape, court_keypoints=None):
        """
        Initialize layout based on frame dimensions.
        
        PARAMETERS:
        -----------
        frame_shape : tuple
            (height, width, channels) of video frame
        court_keypoints : array, optional
            Court keypoints for smart positioning around court
        """
        self.frame_height = frame_shape[0]
        self.frame_width = frame_shape[1]
        
        # Calculate court bounds for smart positioning
        if court_keypoints is not None:
            self.court_bounds = self._calculate_court_bounds(court_keypoints)
        
        # Calculate UI zone sizes (as percentages)
        self._calculate_base_dimensions()
        
        # Calculate positions (with collision avoidance)
        self._calculate_zones()
    
    def _calculate_base_dimensions(self):
        """
        Calculate UI element sizes as percentages of frame.
        
        RATIONALE:
        ----------
        - Mini court: 15% of width gives good visibility without blocking view
        - Stats panel: 25% width, 35% height provides readable text
        - Padding: 2% of smaller dimension ensures consistent margins
        """
        # Mini court: 15% of frame width, 2:1 aspect ratio (portrait)
        self.mini_court_width = int(self.frame_width * 0.15)
        self.mini_court_height = int(self.mini_court_width * 2.0)
        
        # Ensure minimum sizes for readability
        self.mini_court_width = max(self.mini_court_width, 150)
        self.mini_court_height = max(self.mini_court_height, 300)
        
        # Stats panel: 25% width, 35% height
        self.stats_panel_width = int(self.frame_width * 0.25)
        self.stats_panel_height = int(self.frame_height * 0.35)
        
        # Minimum sizes
        self.stats_panel_width = max(self.stats_panel_width, 250)
        self.stats_panel_height = max(self.stats_panel_height, 180)
        
        # Padding: 2% of smaller dimension
        self.padding = int(min(self.frame_width, self.frame_height) * 0.02)
        self.padding = max(self.padding, 10)  # At least 10 pixels
    
    def _zones_overlap(self, zone1, zone2):
        """
        Check if two rectangular zones overlap.
        
        ALGORITHM (Standard Rectangle Collision):
        ==========================================
        Two rectangles DON'T overlap if:
            - rect1 is completely to the left of rect2, OR
            - rect1 is completely to the right of rect2, OR
            - rect1 is completely above rect2, OR
            - rect1 is completely below rect2
        
        Two rectangles DO overlap if none of the above are true.
        """
        left1, right1 = zone1['x'], zone1['x'] + zone1['width']
        top1, bottom1 = zone1['y'], zone1['y'] + zone1['height']
        
        left2, right2 = zone2['x'], zone2['x'] + zone2['width']
        top2, bottom2 = zone2['y'], zone2['y'] + zone2['height']
        
        # Check for no overlap conditions
        if right1 < left2:  return False  # zone1 is left of zone2
        if right2 < left1:  return False  # zone2 is left of zone1
        if bottom1 < top2:  return False  # zone1 is above zone2
        if bottom2 < top1:  return False  # zone2 is above zone1
        
        return True  # They overlap!

"""
LAYOUT ZONES:
=============

    ┌────────────────────────────────────────────────────┐
    │                                   ┌─────────────┐  │
    │                                   │ Mini Court  │  │ ← Top-right
    │             COURT                 │  (15% x 30%)│  │
    │              VIEW                 └─────────────┘  │
    │                                   ┌─────────────┐  │
    │                                   │Shot Legend  │  │ ← Below mini court
    │                                   └─────────────┘  │
    │  ┌───────────────┐  ┌─────────────────────────┐   │
    │  │ Shot Analysis │  │     Player Stats        │   │ ← Bottom
    │  │  (bottom-left)│  │    (bottom-center)      │   │
    │  └───────────────┘  └─────────────────────────┘   │
    └────────────────────────────────────────────────────┘

"""


# =====================================================================
# SECTION 5: MULTI-CRITERIA PLAYER SELECTION
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 5: MULTI-CRITERIA PLAYER SELECTION                           █
█████████████████████████████████████████████████████████████████████████

THE PROBLEM:
============

Original algorithm: "Player = person closest to ANY court keypoint"

    def choose_players_OLD(court_keypoints, player_dict):
        distances = []
        for person in detected_persons:
            min_distance = min(distance(person, kp) for kp in keypoints)
            distances.append((person_id, min_distance))
        
        distances.sort(by=min_distance)
        return distances[:2]  # Top 2 closest

WHY THIS FAILS:
===============

    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │    Line Judge     ●───● ← Very close to court keypoint!   │
    │                   │                                        │
    │    ┌─────────────────────────────────────┐                 │
    │    │                COURT                 │                 │
    │    │                                      │                 │
    │    │   ○ Player 1              Player 2 ○ │                 │
    │    │                                      │                 │
    │    └─────────────────────────────────────┘                 │
    │                   │                                        │
    │    Ball Boy       ●───● ← Also close to court keypoint!   │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

    Line judges and ball boys are CLOSER to keypoint corners than
    actual players who are at baseline/service line!

THE SOLUTION: MULTI-CRITERIA SCORING
====================================

Instead of just distance, score each detected person on MULTIPLE criteria:

    Total Score = Criterion 1 + Criterion 2 + Criterion 3 + ...

SCORING CRITERIA EXPLAINED:
===========================
"""

def choose_players_algorithm(court_keypoints, player_dict):
    """
    Enhanced player selection using multiple criteria.
    
    CRITERIA BREAKDOWN:
    ===================
    
    1. COURT POSITION (max 100 points)
       --------------------------------
       Real players are INSIDE or NEAR the court boundaries.
       Line judges are outside, ball boys are at corners.
       
       Inside court bounds:     +100 points
       Within 200px of court:   +50 points
       Outside:                 +0 points
    
    2. BOUNDING BOX SIZE (max 50 points)
       ----------------------------------
       Real players appear LARGER in frame because camera focuses on them.
       Line judges/ball boys appear smaller as they're peripheral.
       
       Score = min(bbox_area / 1000, 50)
       
       Example:
         Player bbox (50 x 200 = 10000 px²) → 10000/1000 = 50 points ★
         Line judge bbox (30 x 100 = 3000 px²) → 3000/1000 = 3 points
    
    3. ASPECT RATIO (max 30 points)
       ----------------------------
       Standing humans are TALLER than WIDE.
       Aspect ratio = height / width
       
       Ratio in [1.3, 4.0]:  +30 points (standing)
       Ratio in [1.0, 1.3]:  +15 points (crouching)
       Otherwise:           +0 points
       
       Example:
         Player (50 wide x 150 tall): 150/50 = 3.0 → +30 points ★
         Seated judge (100 x 80): 80/100 = 0.8 → +0 points
    
    4. DISTANCE FROM COURT CENTER (max 40 points)
       ------------------------------------------
       Real players are near baseline or service line, which are
       closer to court CENTER than the corners where judges sit.
       
       Score = 40 * (1 - normalized_distance)
       
       Example:
         Player at baseline: distance ~200px → score ~35 points ★
         Line judge at corner: distance ~500px → score ~10 points
    
    5. VERTICAL POSITION (max 20 points)
       ---------------------------------
       Players are within the court's vertical boundaries.
       
       If player_y is between court_top and court_bottom: +20 points
    
    6. MINIMUM HEIGHT (max 20 points)
       ------------------------------
       Real players should be reasonably tall in frame (at least 5% 
       of frame height).
       
       If bbox_height > frame_height * 0.05: +20 points
    
    TOTAL MAXIMUM SCORE: 100 + 50 + 30 + 40 + 20 + 20 = 260 points
    """
    
    # Calculate court bounds
    court_bounds = estimate_court_bounds(court_keypoints)
    
    candidates = []
    for track_id, bbox in player_dict.items():
        score = 0
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # CRITERION 1: Court position (+100 max)
        if is_inside_court((center_x, center_y), court_bounds, margin=100):
            score += 100
        elif is_inside_court((center_x, center_y), court_bounds, margin=200):
            score += 50
        
        # CRITERION 2: Bounding box size (+50 max)
        bbox_area = (x2 - x1) * (y2 - y1)
        score += min(bbox_area / 1000, 50)
        
        # CRITERION 3: Aspect ratio (+30 max)
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        aspect_ratio = bbox_height / max(bbox_width, 1)
        if 1.3 <= aspect_ratio <= 4.0:
            score += 30
        elif 1.0 <= aspect_ratio <= 1.3:
            score += 15
        
        # CRITERION 4: Distance from center (+40 max)
        court_center = court_bounds['center']
        distance = euclidean_distance((center_x, center_y), court_center)
        max_distance = max(court_bounds['width'], court_bounds['height'])
        score += 40 * (1 - min(distance / max_distance, 1))
        
        # CRITERION 5: Vertical position (+20)
        if court_bounds['top'] < center_y < court_bounds['bottom']:
            score += 20
        
        # CRITERION 6: Minimum height (+20)
        if bbox_height > court_bounds['height'] * 0.05:
            score += 20
        
        candidates.append((track_id, score, bbox))
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 2
    return [c[0] for c in candidates[:2]]

"""
EXAMPLE SCORING:
================

Detected persons in frame:
    - Person A: Player at baseline
    - Person B: Player at service line
    - Person C: Line judge at corner
    - Person D: Ball boy near net post

    ┌──────────────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┐
    │ Person       │ Crit1│ Crit2│ Crit3│ Crit4│ Crit5│ Crit6│ TOTAL │
    ├──────────────┼──────┼──────┼──────┼──────┼──────┼──────┼───────┤
    │ A (Player 1) │  100 │   45 │   30 │   35 │   20 │   20 │  250  │ ★
    │ B (Player 2) │  100 │   40 │   30 │   38 │   20 │   20 │  248  │ ★
    │ C (Judge)    │   50 │   15 │    0 │   10 │    0 │    0 │   75  │
    │ D (Ball boy) │   50 │   20 │   30 │   20 │   20 │   10 │  150  │
    └──────────────┴──────┴──────┴──────┴──────┴──────┴──────┴───────┘

    Winners: Person A (250) and Person B (248) → Correct players! ✓

"""


# =====================================================================
# SECTION 6: HOMOGRAPHY - The Math Behind Court Mapping
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 6: HOMOGRAPHY - The Math Behind Court Mapping               █
█████████████████████████████████████████████████████████████████████████

WHAT IS HOMOGRAPHY?
===================

A homography is a transformation that maps points from one plane to another.

    Real Court (as seen by camera)  →  Mini Court (on screen overlay)
    
    Player at position (x, y)  →  Mini court position (x', y')

VISUAL EXPLANATION:
===================

    REAL COURT VIEW                      MINI COURT
    (perspective distorted)              (top-down view)
    
         ╱─────────────╲                ┌─────────────┐
        ╱               ╲               │             │
       ╱     ○ Player    ╲              │      ●      │
      │                   │             │             │
      │                   │      →      │             │
      │                   │      H      │             │
       ╲                 ╱              │             │
        ╲               ╱               │             │
         ╲─────────────╱                └─────────────┘
         
    H = Homography matrix (3x3)

THE MATHEMATICS:
================

Given:
    - Source point: (x, y) on real court
    - Destination point: (x', y') on mini court
    - Homography matrix H (3x3)

The transformation is:
    
    ┌   ┐   ┌           ┐   ┌   ┐
    │ x'│   │ h11 h12 h13│   │ x │
    │ y'│ = │ h21 h22 h23│ × │ y │
    │ w │   │ h31 h32 h33│   │ 1 │
    └   ┘   └           ┘   └   ┘
    
    Final coordinates: (x'/w, y'/w)

HOW TO COMPUTE HOMOGRAPHY:
==========================

We need at least 4 corresponding points:

    Source (real court keypoints):      Destination (mini court corners):
    ┌──────────────────────────────┐    ┌────────────────────────────┐
    │ Point 0: (x0, y0) top-left  │    │ (mini_start_x, mini_start_y)│
    │ Point 1: (x1, y1) top-right │ →  │ (mini_end_x, mini_start_y)  │
    │ Point 2: (x2, y2) bot-left  │    │ (mini_start_x, mini_end_y)  │
    │ Point 3: (x3, y3) bot-right │    │ (mini_end_x, mini_end_y)    │
    └──────────────────────────────┘    └────────────────────────────┘

OpenCV computes H using:

    H, status = cv2.findHomography(source_points, destination_points)

IMPLEMENTATION:
===============
"""

import cv2
import numpy as np

def compute_homography_example(court_keypoints, mini_court_corners):
    """
    Compute the homography matrix from court keypoints to mini court.
    
    PARAMETERS:
    -----------
    court_keypoints : array
        [x0, y0, x1, y1, ...] detected keypoints on real court
    mini_court_corners : dict
        Mini court corner positions
        
    RETURNS:
    --------
    H : 3x3 numpy array
        Homography matrix
    """
    # Source: 4 corners of real court from keypoints
    src_points = np.array([
        [court_keypoints[0], court_keypoints[1]],   # Top-left
        [court_keypoints[2], court_keypoints[3]],   # Top-right
        [court_keypoints[4], court_keypoints[5]],   # Bottom-left
        [court_keypoints[6], court_keypoints[7]],   # Bottom-right
    ], dtype=np.float32)
    
    # Destination: 4 corners of mini court
    dst_points = np.array([
        [mini_court_corners['start_x'], mini_court_corners['start_y']],
        [mini_court_corners['end_x'],   mini_court_corners['start_y']],
        [mini_court_corners['start_x'], mini_court_corners['end_y']],
        [mini_court_corners['end_x'],   mini_court_corners['end_y']],
    ], dtype=np.float32)
    
    # Compute homography
    H, status = cv2.findHomography(src_points, dst_points)
    
    return H


def transform_point_example(point, H):
    """
    Transform a point using homography matrix.
    
    PARAMETERS:
    -----------
    point : tuple
        (x, y) position on real court
    H : 3x3 array
        Homography matrix
        
    RETURNS:
    --------
    tuple
        (x', y') position on mini court
    """
    x, y = point
    
    # Create homogeneous coordinates
    src = np.array([[x], [y], [1]])
    
    # Apply transformation
    dst = H @ src
    
    # Convert back from homogeneous coordinates
    x_prime = dst[0, 0] / dst[2, 0]
    y_prime = dst[1, 0] / dst[2, 0]
    
    return (x_prime, y_prime)

"""
WHY PER-FRAME HOMOGRAPHY MATTERS:
=================================

The homography H is computed from court keypoints.

If keypoints change (due to camera motion), H must be recomputed!

    Frame 0: Keypoints K0 → Homography H0
    Frame 1: Keypoints K1 → Homography H1  (K1 ≠ K0 if camera moved)
    Frame 2: Keypoints K2 → Homography H2
    ...

Using H0 for all frames = WRONG mapping when camera moves!

    ┌───────────────────────────────────────────────────────────┐
    │ CURRENT IMPLEMENTATION:                                   │
    │                                                           │
    │ We don't explicitly compute H matrix.                     │
    │ Instead, we use per-frame keypoints directly in the       │
    │ coordinate conversion function.                           │
    │                                                           │
    │ Future optimization: Pre-compute H for each frame         │
    │ for faster batch transformations.                         │
    └───────────────────────────────────────────────────────────┘

"""


# =====================================================================
# SECTION 7: IMPLEMENTATION DETAILS
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 7: IMPLEMENTATION DETAILS - Files & Functions                █
█████████████████████████████████████████████████████████████████████████

FILES MODIFIED:
===============

┌──────────────────────────────────────────────────────────────────────┐
│ FILE                              │ CHANGES                          │
├──────────────────────────────────────────────────────────────────────┤
│ main.py                           │ • Added ENABLE_PER_FRAME_KEYPOINTS│
│                                   │ • Integrated UILayoutManager     │
│                                   │ • Per-frame keypoint detection   │
├──────────────────────────────────────────────────────────────────────┤
│ court_line_detector/              │ • Added predict_all_frames()     │
│   court_line_detector.py          │ • Added smooth_keypoints()       │
│                                   │ • Added draw_keypoints_dynamic() │
├──────────────────────────────────────────────────────────────────────┤
│ mini_visual_court/                │ • Dynamic layout support         │
│   mini_court.py                   │ • Per-frame keypoint usage       │
│                                   │ • Responsive sizing              │
├──────────────────────────────────────────────────────────────────────┤
│ trackers/                         │ • Multi-criteria choose_players()│
│   player_tracker.py               │ • _estimate_court_bounds()       │
│                                   │ • _is_inside_court()             │
├──────────────────────────────────────────────────────────────────────┤
│ utils/                            │ • Dynamic position support       │
│   player_stats_drawer_utils.py    │ • layout_params parameter        │
└──────────────────────────────────────────────────────────────────────┘

NEW FILES CREATED:
==================

┌──────────────────────────────────────────────────────────────────────┐
│ FILE                              │ PURPOSE                          │
├──────────────────────────────────────────────────────────────────────┤
│ utils/                            │ Dynamic UI positioning           │
│   ui_layout_manager.py            │ Collision detection              │
│                                   │ Responsive sizing                │
└──────────────────────────────────────────────────────────────────────┘

KEY FUNCTIONS:
==============

1. court_line_detector.predict_all_frames(frames, smooth, window_size)
   → Returns list of keypoints, one per frame

2. court_line_detector.smooth_keypoints(keypoints_list, window_size)
   → Applies moving average filter to reduce jitter

3. UILayoutManager(frame_shape, court_keypoints)
   → Creates layout manager for dynamic UI positioning

4. player_tracker.choose_players(court_keypoints, player_dict)
   → Multi-criteria player selection (NEW algorithm)

5. mini_court.convert_bounding_boxes_to_mini_court_coordinates(
       player_boxes, ball_boxes, all_court_keypoints)
   → Now accepts per-frame keypoints (list of arrays)

CONFIGURATION FLAGS:
====================

In main.py:

    ENABLE_SHOT_CLASSIFICATION = True   # Shot type detection
    ENABLE_PER_FRAME_KEYPOINTS = True   # Camera-robust mode (slower)
    ENABLE_PER_FRAME_KEYPOINTS = False  # Fast mode (original behavior)

"""


# =====================================================================
# SECTION 8: HOW TO RUN AND TEST
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 8: HOW TO RUN AND TEST                                       █
█████████████████████████████████████████████████████████████████████████

STEP-BY-STEP TESTING GUIDE:
===========================

STEP 1: PREPARE FOR FRESH DETECTION
------------------------------------
Since you have old tracker stubs from before, you need to regenerate them.

In main.py, find these lines and change read_from_stub to False:

    # BEFORE (using old stubs):
    player_detections = player_tracker.detect_frames(
        video_frames, 
        read_from_stub=True,  # ← Change to False
        stub_path="tracker_stubs/player_detections.pkl"
    )
    
    ball_detections = ball_tracker.detect_frames(
        video_frames, 
        read_from_stub=True,  # ← Change to False
        stub_path="tracker_stubs/ball_detections.pkl"
    )
    
    # AFTER (regenerate fresh):
    player_detections = player_tracker.detect_frames(
        video_frames, 
        read_from_stub=False,  # ← Fresh detection
        stub_path="tracker_stubs/player_detections.pkl"
    )

NOTE: After first run, you can change back to True to save time.


STEP 2: CHOOSE YOUR MODE
-------------------------
In main.py, set the flag:

    # For ACCURATE mode (handles camera motion, slower):
    ENABLE_PER_FRAME_KEYPOINTS = True
    
    # For FAST mode (original behavior, may break with camera motion):
    ENABLE_PER_FRAME_KEYPOINTS = False

I recommend starting with False to quickly verify everything works,
then switch to True for the full camera-robust experience.


STEP 3: SELECT INPUT VIDEO
--------------------------
In main.py:

    # Test with the working video first:
    input_video_path = "input_videos/input_video.mp4"  # or input_video_1.mp4
    
    # Then test with the problematic video:
    input_video_path = "input_videos/input_video_2.mp4"


STEP 4: RUN THE SCRIPT
----------------------
Open terminal in D:\Tennis-Vision directory:

    # Activate virtual environment (if using one)
    .\\venv\\Scripts\\activate
    
    # Run the script
    python main.py


STEP 5: MONITOR OUTPUT
----------------------
Watch for these console messages:

    [CAMERA-ROBUST MODE] Detecting keypoints for ALL frames...
    Note: This takes longer but handles camera motion correctly.
    Detecting court keypoints for 500 frames...
      Processed 100/500 frames
      Processed 200/500 frames
      ...
    Applying temporal smoothing to keypoints...
    
      [PLAYER SELECTION] Candidates scored: [(1, 248.5), (2, 245.2), ...]
      [PLAYER SELECTION] Chosen players: [1, 2]
    
    [CAMERA-ROBUST] Using per-frame keypoints for coordinate transformation


STEP 6: CHECK OUTPUT VIDEO
--------------------------
Output is saved to: output_videos/output_video.avi

Watch for:
    ✓ Court keypoints track correctly with camera motion
    ✓ Player bounding boxes are on actual players (not line judges)
    ✓ Mini court shows players in correct positions
    ✓ Stats panel doesn't overlap with other elements
    ✓ Mini court is appropriately sized


EXPECTED PROCESSING TIMES:
==========================

For a 500-frame video (~20 seconds at 24fps):

    Mode                    │ Keypoint Time │ Total Time
    ────────────────────────┼───────────────┼────────────
    ENABLE_PER_FRAME=False  │ ~2 seconds    │ ~3-5 minutes
    ENABLE_PER_FRAME=True   │ ~3-5 minutes  │ ~10-15 minutes


TROUBLESHOOTING:
================

Problem: "ModuleNotFoundError: No module named 'utils.ui_layout_manager'"
Solution: The file wasn't created. Check D:\Tennis-Vision\utils\ui_layout_manager.py exists.

Problem: Script crashes with KeyError in player_detections
Solution: Set read_from_stub=False to regenerate fresh detections.

Problem: Mini court is in wrong position
Solution: Check that UILayoutManager is being used. Look for:
          "Setting up dynamic UI layout..." in console output.

Problem: Still detecting line judges instead of players
Solution: The multi-criteria scoring should handle this now.
          Check console for "[PLAYER SELECTION] Candidates scored" to verify.

"""


# =====================================================================
# SECTION 9: SPEED OPTIMIZATION STRATEGIES
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 9: SPEED OPTIMIZATION STRATEGIES (Future Work)              █
█████████████████████████████████████████████████████████████████████████

CURRENT BOTTLENECK:
===================

Per-frame keypoint detection is ~5x slower because:
    - Neural network inference (ResNet-50) runs on EVERY frame
    - Feature extraction repeated for every frame
    - No parallelization in current implementation

OPTIMIZATION STRATEGY 1: KEYFRAME DETECTION
============================================

Instead of detecting every frame, detect every Nth frame and interpolate.

    Frame 0:  Detect  → K0                 Keyframe
    Frame 1:  Skip    → interpolate(K0, K10)
    Frame 2:  Skip    → interpolate(K0, K10)
    ...
    Frame 9:  Skip    → interpolate(K0, K10)
    Frame 10: Detect  → K10                Keyframe
    Frame 11: Skip    → interpolate(K10, K20)
    ...

    def interpolate_keypoints(kp1, kp2, alpha):
        '''Linear interpolation between two keypoint sets'''
        return kp1 * (1 - alpha) + kp2 * alpha

Speed improvement: ~N times faster (e.g., 5x for every 5th frame)


OPTIMIZATION STRATEGY 2: MOTION-TRIGGERED DETECTION
====================================================

Only re-detect keypoints when camera motion exceeds threshold.

    def should_redetect(prev_frame, curr_frame, motion_threshold=50):
        '''Check if camera has moved significantly'''
        # Method 1: Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, ...)
        motion_magnitude = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
        
        return motion_magnitude > motion_threshold

When motion is below threshold, reuse previous keypoints.


OPTIMIZATION STRATEGY 3: GPU BATCHING
=====================================

Process multiple frames in parallel using GPU batch inference.

    # Instead of:
    for frame in frames:
        keypoints = model(frame)
    
    # Do:
    batch_size = 32
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        keypoints_batch = model(batch)  # GPU processes all at once


OPTIMIZATION STRATEGY 4: CACHING
================================

Cache keypoints when camera is stable, invalidate on detected motion.

    class KeypointCache:
        def __init__(self):
            self.cached_keypoints = None
            self.cache_valid = False
        
        def get_keypoints(self, frame, prev_frame):
            if self.cache_valid and not self.detect_motion(prev_frame, frame):
                return self.cached_keypoints  # Use cache
            
            # Recompute
            keypoints = self.detect_keypoints(frame)
            self.cached_keypoints = keypoints
            self.cache_valid = True
            return keypoints


PRIORITY FOR IMPLEMENTATION:
============================

    1. Keyframe Detection (easiest, good speedup)
    2. Motion-Triggered Detection (moderate, smart optimization)
    3. Caching (easy, helps with stable cameras)
    4. GPU Batching (harder, requires GPU memory management)

"""


# =====================================================================
# SECTION 10: TROUBLESHOOTING GUIDE
# =====================================================================
"""
█████████████████████████████████████████████████████████████████████████
█ SECTION 10: TROUBLESHOOTING GUIDE                                    █
█████████████████████████████████████████████████████████████████████████

COMMON ISSUES AND SOLUTIONS:
============================

ISSUE 1: KeyError when accessing player_detections[frame_num][player_id]
------------------------------------------------------------------------
Cause: Stale tracker stubs from before the player ID normalization

Solution:
    1. Delete old stubs:
       - tracker_stubs/player_detections.pkl
       - tracker_stubs/ball_detections.pkl
    2. Or change read_from_stub=False in main.py
    3. Run again to regenerate fresh detections


ISSUE 2: Mini court appears too large or too small
--------------------------------------------------
Cause: Layout manager not being used, or frame resolution issue

Solution:
    1. Check that this print appears: "Setting up dynamic UI layout..."
    2. Verify UILayoutManager is imported in main.py
    3. Check get_mini_court_params() is being called


ISSUE 3: Player stats panel overlaps with mini court
----------------------------------------------------
Cause: Collision detection not working properly

Solution:
    1. Check UILayoutManager._zones_overlap() is called
    2. Verify stats_params is passed to draw_player_stats()
    3. Reduce panel sizes if needed in ui_layout_manager.py


ISSUE 4: Keypoints still jittering despite smoothing
-----------------------------------------------------
Cause: Window size too small, or very noisy detections

Solution:
    1. Increase window_size in predict_all_frames():
       all_court_keypoints = court_line_detector.predict_all_frames(
           video_frames, smooth=True, window_size=7  # Try 7 instead of 5
       )


ISSUE 5: Wrong players selected (still getting line judges)
------------------------------------------------------------
Cause: Scoring weights need adjustment for your video

Solution:
    1. Check debug output: "[PLAYER SELECTION] Candidates scored: ..."
    2. Adjust weights in choose_players() in player_tracker.py
    3. Increase court position weight (currently +100)
    4. Increase size weight (currently max +50)


ISSUE 6: Script runs very slowly (even slower than expected)
-------------------------------------------------------------
Cause: Processing is happening on CPU instead of GPU

Solution:
    1. Verify CUDA is installed: python -c "import torch; print(torch.cuda.is_available())"
    2. If False, install PyTorch with CUDA support
    3. Check GPU memory is available


ISSUE 7: "FutureWarning: Calling int on a single element Series"
-----------------------------------------------------------------
Cause: Pandas version incompatibility

Solution:
    Already handled in previous fixes. If it appears:
    1. Use .iloc[0] instead of direct indexing
    2. Or update pandas: pip install --upgrade pandas


DEBUG MODE:
===========

Add these prints to troubleshoot:

    # In main.py, after layout manager creation:
    print(f"[DEBUG] Mini court params: {mini_court_params}")
    print(f"[DEBUG] Stats params: {stats_params}")
    
    # In court_line_detector.py, in predict_all_frames:
    print(f"[DEBUG] Keypoints variance: {np.var(all_keypoints, axis=0).mean():.2f}")
    
    # In player_tracker.py, in choose_players:
    for c in candidates:
        print(f"[DEBUG] Candidate {c[0]}: score={c[1]:.1f}, bbox={c[2]}")

"""


# =====================================================================
# END OF NOTES
# =====================================================================
"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎾 END OF CAMERA-ROBUST TENNIS VISION NOTES                       ║
║                                                                      ║
║   Created: December 2024                                            ║
║   Version: 2.0                                                       ║
║                                                                      ║
║   These notes document the complete camera-robust implementation    ║
║   for handling variable camera positions in tennis video analysis.  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

QUICK REFERENCE COMMANDS:
=========================

# Activate environment and run:
cd D:\\Tennis-Vision
.\\venv\\Scripts\\activate
python main.py

# Check for output:
# Output video: output_videos/output_video.avi

# Toggle modes in main.py:
ENABLE_PER_FRAME_KEYPOINTS = True   # Accurate (slower)
ENABLE_PER_FRAME_KEYPOINTS = False  # Fast (may break with motion)

# For fresh detection (first time after changes):
# Change read_from_stub=False in main.py, run once, then change back to True

"""

print("Camera-Robust Notes loaded! Check docstrings for detailed documentation.")
