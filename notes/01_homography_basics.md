# Homography Basics
## Understanding Court Coordinate Mapping in Tennis Vision

---

## What is Homography?

**Homography** is a mathematical transformation that maps points from one plane to another. In Tennis Vision, we use it to:

- Map real court coordinates (meters) → Mini court coordinates (pixels)
- Map player positions on video → Positions on 2D court diagram

### Visual Representation

```
Real Video Frame                    Mini Court Diagram
+------------------+                +--------+
|   * Player 1     |   Homography   |   •    |
|                  |  ============> |        |
|        ⚪        |                |   ⚫  |
|   * Player 2     |                |   •    |
+------------------+                +--------+
```

---

## The Math (Simplified)

A homography is represented by a 3x3 matrix **H**:

```
     | h11  h12  h13 |
H =  | h21  h22  h23 |
     | h31  h32  h33 |
```

To transform a point (x, y) to (x', y'):

```
| x' |       | x |
| y' | = H × | y |
| 1  |       | 1 |
```

---

## How We Use It in Tennis Vision

### Step 1: Identify Court Keypoints
We detect 14 keypoints on the tennis court:
- 4 corners of the court
- 4 corners of the service boxes
- 6 intersection points

### Step 2: Define Target Coordinates
We know the real-world positions of these keypoints on a standard tennis court (in meters).

### Step 3: Compute Homography Matrix
Using OpenCV:
```python
import cv2
import numpy as np

# Source points (from video frame)
src_points = np.array([
    [x1, y1],  # Court corner 1
    [x2, y2],  # Court corner 2
    [x3, y3],  # Court corner 3
    [x4, y4],  # Court corner 4
], dtype=np.float32)

# Destination points (mini court pixels)
dst_points = np.array([
    [0, 0],
    [250, 0],
    [250, 500],
    [0, 500],
], dtype=np.float32)

# Compute homography
H, _ = cv2.findHomography(src_points, dst_points)
```

### Step 4: Transform Points
```python
# Transform player position to mini court
player_pos = np.array([[px, py]], dtype=np.float32)
mini_court_pos = cv2.perspectiveTransform(player_pos.reshape(-1, 1, 2), H)
```

---

## Why Per-Frame Keypoints Matter

**Problem**: Camera moves during the match
- Static keypoints become inaccurate
- Homography becomes wrong
- Mini court positions drift

**Solution**: Detect keypoints EVERY frame
- Recalculate homography per frame
- Accurate mapping even with camera motion

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **Keypoints** | Known reference points on court |
| **Homography Matrix** | The 3x3 transformation matrix |
| **Perspective Transform** | Applying the matrix to map points |
| **Temporal Smoothing** | Averaging keypoints across frames to reduce jitter |

---

## In Our Code

**File**: `mini_visual_court/mini_court.py`

```python
def convert_bounding_boxes_to_mini_court_coordinates(
    self, player_boxes, ball_boxes, all_court_keypoints
):
    # For each frame, get current keypoints
    current_keypoints = all_court_keypoints[frame_num]
    
    # Calculate player position on mini court using homography
    mini_court_x, mini_court_y = self.get_mini_court_coordinates(
        player_center_x, player_center_y, 
        current_keypoints
    )
```

---

## Further Reading

- [OpenCV Homography Tutorial](https://docs.opencv.org/master/d9/dab/tutorial_homography.html)
- [Wikipedia: Homography](https://en.wikipedia.org/wiki/Homography_(computer_vision))
