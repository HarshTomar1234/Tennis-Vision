# Temporal Smoothing
## Reducing Jitter in Frame-by-Frame Detection

---

## The Problem

When detecting court keypoints frame-by-frame:

```
Frame 1: Keypoint at (100, 50)
Frame 2: Keypoint at (102, 49)  ← Small jitter
Frame 3: Keypoint at (98, 51)   ← Jitters back
Frame 4: Keypoint at (101, 50)
```

Even though the camera is stable, detection has small variations.

**Result**: Mini court wiggle, keypoint lines shake

---

## The Solution: Temporal Smoothing

Instead of using raw detections, we **average over multiple frames**.

### Moving Average

```python
window_size = 5

smoothed_keypoints[frame_5] = average(
    keypoints[frame_1],
    keypoints[frame_2],
    keypoints[frame_3],
    keypoints[frame_4],
    keypoints[frame_5]
)
```

---

## How We Implement It

**File**: `court_line_detector/court_line_detector.py`

```python
def smooth_keypoints(self, all_keypoints, window_size=5):
    """
    Apply temporal smoothing to keypoints.
    
    Uses a moving average to reduce frame-to-frame jitter
    while preserving actual camera movements.
    """
    smoothed = []
    
    for i in range(len(all_keypoints)):
        # Get window of frames around current frame
        start = max(0, i - window_size // 2)
        end = min(len(all_keypoints), i + window_size // 2 + 1)
        
        # Average keypoints in window
        window_keypoints = all_keypoints[start:end]
        avg_keypoints = np.mean(window_keypoints, axis=0)
        
        smoothed.append(avg_keypoints)
    
    return smoothed
```

---

## Visual Comparison

### Without Smoothing
```
Position
   ^
52 |     •
51 |   •   •
50 | •   •   •
49 |         •
   +----------→ Frame
     1 2 3 4 5 (jumpy)
```

### With Smoothing
```
Position
   ^
51 |
50 | • • • • •
49 |
   +----------→ Frame
     1 2 3 4 5 (smooth)
```

---

## Trade-offs

| Factor | Small Window (3) | Large Window (7+) |
|--------|------------------|-------------------|
| **Jitter reduction** | Less | More |
| **Responsiveness** | Fast | Slow |
| **Camera motion tracking** | Good | May lag |

### Our Choice: Window = 5
- Good jitter reduction
- Still responsive to actual camera movement
- Works well for tennis broadcast videos

---

## Types of Smoothing

### 1. Simple Moving Average (What we use)
```python
smoothed = mean(keypoints[i-2:i+3])
```
**Pros**: Simple, effective
**Cons**: All frames weighted equally

### 2. Weighted Moving Average
```python
weights = [0.1, 0.2, 0.4, 0.2, 0.1]
smoothed = weighted_mean(keypoints[i-2:i+3], weights)
```
**Pros**: Current frame has more influence
**Cons**: Slightly more complex

### 3. Exponential Moving Average
```python
smoothed[i] = alpha * keypoint[i] + (1-alpha) * smoothed[i-1]
```
**Pros**: Less memory, adjustable responsiveness
**Cons**: Can lag behind sudden changes

---

## When to Use Temporal Smoothing

| Use Case | Apply Smoothing? |
|----------|-----------------|
| Court keypoints | ✅ Yes |
| Ball position | ⚠️ Careful (Kalman better) |
| Player bounding boxes | ✅ Yes (gentle) |
| Camera motion | ✅ Yes |

---

## In Our Code

```python
# In main.py
if ENABLE_PER_FRAME_KEYPOINTS:
    all_court_keypoints = court_line_detector.predict_all_frames(video_frames)
    # Apply smoothing to reduce jitter
    all_court_keypoints = court_line_detector.smooth_keypoints(
        all_court_keypoints, 
        window_size=5
    )
```

---

## Key Insight

**Temporal smoothing is about trust**:
- We trust that the camera doesn't jump randomly
- Detection noise is the enemy, not real motion
- Averaging assumes adjacent frames are similar

---

## Further Reading

- [Moving Average - Wikipedia](https://en.wikipedia.org/wiki/Moving_average)
- [Video Stabilization Techniques](https://docs.opencv.org/master/d5/d50/tutorial_video_write.html)
