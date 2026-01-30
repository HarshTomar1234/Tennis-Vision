# Kalman Filter
## Prediction and Smoothing for Object Tracking

---

## What is a Kalman Filter?

A **Kalman Filter** is an algorithm that:
1. **Predicts** where an object will be
2. **Updates** its prediction based on new measurements
3. **Smooths** noisy observations

Perfect for tracking a tennis ball that moves fast and gets missed in some frames!

---

## The Problem It Solves

```
Frame 1: Ball detected at (100, 50) ✓
Frame 2: Ball NOT detected ❌ → Where is the ball?
Frame 3: Ball NOT detected ❌ → Still missing!
Frame 4: Ball detected at (200, 80) ✓
```

**Without Kalman**: Ball "teleports" from (100,50) to (200,80)
**With Kalman**: Ball smoothly moves through predicted positions

---

## How It Works (Simplified)

### Two Steps Per Frame:

#### 1. PREDICT (Guess where ball will be)
```
predicted_position = current_position + velocity × time
predicted_velocity = current_velocity
```

#### 2. UPDATE (Correct with measurement)
```
If ball detected:
    actual_position = detection
    correction = actual_position - predicted_position
    final_position = predicted_position + (correction × gain)
Else:
    final_position = predicted_position  # Use prediction!
```

---

## Visual Example

```
Time →
       ●         ●              ●         ●
Frame: 1    2    3    4    5    6    7    8
       ↑    ↓    ↓    ↓    ↑    ↓    ↓    ↑
       Det  Miss Miss Miss Det  Miss Miss Det

Without Kalman:  ●---------●---------●
                 (teleports)

With Kalman:     ●--○--○--○--●--○--○--●
                 (smooth path with predictions ○)
```

---

## The Math (Simplified)

### State Vector
```python
state = [x, y, vx, vy]  # position and velocity
```

### Prediction
```python
# State transition matrix (assumes constant velocity)
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]

predicted_state = F @ current_state
```

### Update
```python
# Kalman gain determines how much to trust measurement vs prediction
K = predicted_uncertainty / (predicted_uncertainty + measurement_noise)

# Correct prediction with measurement
updated_state = predicted_state + K × (measurement - predicted_state)
```

---

## In SORT Tracker (What We'll Use)

The **Roboflow Trackers** library has Kalman Filter built-in!

```python
from trackers import SORTTracker

tracker = SORTTracker(
    lost_track_buffer=10,  # Keep predicting for 10 frames if lost
)

# The tracker internally uses Kalman filter to:
# 1. Predict ball position when not detected
# 2. Smooth ball trajectory
# 3. Maintain tracking ID
```

---

## Key Parameters

| Parameter | Effect |
|-----------|--------|
| **Process Noise** | How much we expect object to change (high = erratic motion) |
| **Measurement Noise** | How noisy our detections are (high = trust predictions more) |
| **lost_track_buffer** | Frames to keep predicting without detection |

---

## Why This Fixes "Teleporting"

| Current System | With Kalman Filter |
|----------------|-------------------|
| Ball not detected → No position | Ball not detected → **Predicted position** |
| Ball appears far away → Jump | Ball appears → **Smooth correction** |
| No velocity model | Uses **velocity for prediction** |

---

## In Tennis Vision

**Current ball tracking**:
```
Frame 10: Ball at (100, 50)
Frame 11: Ball not detected → ???
Frame 12: Ball at (150, 70) → JUMP!
```

**With Kalman (SORT)**:
```
Frame 10: Ball at (100, 50), velocity (10, 4)
Frame 11: Ball not detected → Predict (110, 54)
Frame 12: Ball at (150, 70) → Smooth update
```

---

## Further Reading

- [Kalman Filter Visual Explanation](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [SORT Paper](https://arxiv.org/abs/1602.00763)
