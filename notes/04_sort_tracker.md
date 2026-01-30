# SORT Tracker
## Simple Online and Realtime Tracking

---

## What is SORT?

**SORT** = **S**imple **O**nline and **R**ealtime **T**racking

A fast, efficient algorithm to track multiple objects across video frames.

**Key Components**:
1. **Kalman Filter** - Predicts object motion
2. **Hungarian Algorithm** - Associates detections to tracks

---

## The Problem SORT Solves

Each frame gives us **detections**, but we don't know:
- Which detection is Player 1?
- Which is Player 2?
- Is this the same ball as last frame?

```
Frame N:   [Det A] [Det B] [Det C]
Frame N+1: [Det X] [Det Y]

Question: Is Det X the same as Det A or Det B?
```

SORT **assigns consistent IDs** across frames.

---

## How SORT Works

### Step 1: Predict
Use Kalman filter to predict where each tracked object will be:
```
Track 1: Was at (100, 50) → Predict (110, 55)
Track 2: Was at (300, 200) → Predict (295, 210)
```

### Step 2: Detect
Get new detections from YOLO:
```
Detection A: (112, 54)
Detection B: (290, 215)
Detection C: (500, 100)  ← New object?
```

### Step 3: Associate (Hungarian Algorithm)
Match predictions to detections using IoU (Intersection over Union):
```
Track 1 predict (110, 55) ↔ Det A (112, 54) ✓ Match!
Track 2 predict (295, 210) ↔ Det B (290, 215) ✓ Match!
Det C → No match → Create new Track 3
```

### Step 4: Update
Update Kalman filter with matched detections:
```
Track 1: Update with Det A → Now at (112, 54), velocity updated
Track 2: Update with Det B → Now at (290, 215), velocity updated
Track 3: New track initialized
```

---

## Visual Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PREDICT   │ --> │   DETECT    │ --> │  ASSOCIATE  │
│ (Kalman)    │     │   (YOLO)    │     │ (Hungarian) │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                       │
       │                                       │
       └───────────── UPDATE ──────────────────┘
```

---

## Key Parameters

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `lost_track_buffer` | Frames to keep predicting if lost | 30 |
| `minimum_iou_threshold` | Minimum overlap for match | 0.3 |
| `track_activation_threshold` | Min confidence to start track | 0.25 |
| `minimum_consecutive_frames` | Frames before valid track | 3 |

---

## Using Roboflow's SORT

### Installation
```bash
pip install trackers
```

### Code Example
```python
from trackers import SORTTracker
import supervision as sv

# Initialize tracker
tracker = SORTTracker(
    lost_track_buffer=10,     # Keep tracking for 10 frames if lost
    frame_rate=30.0,          # Video FPS
    minimum_consecutive_frames=3
)

# For each frame:
detections = yolo_model(frame)  # Get YOLO detections
detections = sv.Detections.from_ultralytics(detections)

# SORT magic happens here!
tracked = tracker.update(detections)

# Now tracked.tracker_id contains consistent IDs
for i, track_id in enumerate(tracked.tracker_id):
    bbox = tracked.xyxy[i]
    print(f"Object {track_id} at {bbox}")
```

---

## Why SORT is Perfect for Tennis

| Feature | Benefit for Tennis |
|---------|-------------------|
| **Fast** (~100 FPS) | Real-time processing |
| **Kalman prediction** | Ball trajectory when missed |
| **Consistent IDs** | Player 1 stays Player 1 |
| **Simple** | Easy to integrate |

---

## SORT vs Current System

| Aspect | Current | With SORT |
|--------|---------|-----------|
| Ball missed | Position unknown | **Predicted** |
| Player ID | Manual selection | **Automatic** |
| Trajectory | Raw detections | **Smooth** |
| ID switches | Possible | **Rare** |

---

## Limitations of SORT

| Limitation | Solution |
|------------|----------|
| ID switches during occlusion | Use DeepSORT |
| No appearance matching | Use DeepSORT |
| Needs good detector | Use better YOLO model |

---

## In Tennis Vision

We'll use SORT for:
1. **Ball tracking** - Fix teleporting
2. **Player tracking** - Consistent IDs

```python
# Ball tracker with SORT
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = SORTTracker(lost_track_buffer=10)
    
    def detect_frame_with_tracking(self, frame):
        results = self.model.predict(frame)
        detections = sv.Detections.from_ultralytics(results)
        tracked = self.tracker.update(detections)
        return tracked
```

---

## Further Reading

- [SORT Paper](https://arxiv.org/abs/1602.00763)
- [Roboflow Trackers Docs](https://trackers.roboflow.com)
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)
