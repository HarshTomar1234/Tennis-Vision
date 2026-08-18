# Shot Detection
## Detecting When Players Hit the Ball

---

## The Goal

Detect the exact moment when a **player hits the ball** with their racket.

NOT when:
- Ball bounces on court ❌
- Ball hits the net ❌
- Ball passes by ❌

---

## The Approach

### Key Insight: Ball Changes Direction

When a player hits the ball, it changes direction:
- Ball coming toward player (moving down in frame)
- Player hits → Ball going away (moving up in frame)

We detect this **direction change in Y-axis**.

---

## The Algorithm

### Step 1: Track Ball Y Position
```python
mid_y = (ball_y1 + ball_y2) / 2
```

### Step 2: Calculate Direction (Delta Y)
```python
delta_y = mid_y[frame] - mid_y[frame - 1]

delta_y > 0 → Ball moving DOWN
delta_y < 0 → Ball moving UP
```

### Step 3: Detect Direction Change
```python
# Going down then up (positive to negative delta)
if delta_y[frame] > 0 and delta_y[frame+1] < 0:
    possible_hit = True

# Going up then down (negative to positive delta)  
if delta_y[frame] < 0 and delta_y[frame+1] > 0:
    possible_hit = True
```

---

## The Problem: Court Bounces!

Direction changes happen when:
1. ✅ Player hits ball (near top/bottom of frame)
2. ❌ Ball bounces on court (near middle of frame)

Both cause direction reversal!

---

## The Solution: Zone Filtering

Divide the frame into zones:

```
┌────────────────────────┐
│     TOP ZONE (35%)     │ ← Player 1 here
│     (Near player)      │
├────────────────────────┤
│    MIDDLE ZONE (30%)   │ ← Court here
│    (Bounce zone)       │
├────────────────────────┤
│    BOTTOM ZONE (35%)   │ ← Player 2 here
│    (Near player)       │
└────────────────────────┘
```

**Rule**: Only count direction changes in TOP or BOTTOM zones!

---

## Implementation

**File**: `trackers/ball_tracker.py`

```python
def get_ball_shot_frames(self, ball_positions):
    # Calculate ball Y movement range
    max_y = df_ball_positions['mid_y'].max()
    min_y = df_ball_positions['mid_y'].min()
    frame_range = max_y - min_y
    
    # Define zones (35% top, 30% middle, 35% bottom)
    player_zone_threshold = frame_range * 0.35
    top_player_zone = min_y + player_zone_threshold
    bottom_player_zone = max_y - player_zone_threshold
    
    for i in range(len(df_ball_positions)):
        if direction_changed:
            ball_y = df_ball_positions['mid_y'].iloc[i]
            
            # Only count if near a player (not in middle)
            is_near_top_player = ball_y < top_player_zone
            is_near_bottom_player = ball_y > bottom_player_zone
            
            if not (is_near_top_player or is_near_bottom_player):
                continue  # Skip - it's a court bounce!
            
            # This is a real player hit
            df_ball_positions.loc[i, 'ball_hit'] = 1
```

---

## Visual Explanation

```
Ball Y Position Over Time:

       Player 1
Top    ─────────●───────────●─────────  ← HITS detected here
       \       / \         / \
        \     /   \       /   \
Middle   \   /     \     /     ●   ← BOUNCE ignored (middle)
          \ /       \   /      |
           ●         \ /       |
Bottom    ───────────●────────────────  ← HITS detected here
       Player 2  
       
Frame:  1  2  3  4  5  6  7  8  9  10
```

---

## Validation

We also validate that the direction change persists:

```python
minimum_change_frames = 25  # Direction must stay changed

# Check next 25 frames to confirm consistent direction
for change_frame in range(i+1, i + 25):
    if still_same_direction:
        change_count += 1

if change_count > 24:
    confirmed_hit = True
```

This filters out noisy false positives.

---

## Results

**Before fix**:
- Detected ~10+ "shots" per rally
- Included all court bounces

**After fix**:
- Detected ~5 actual shots per rally
- Only player hits count

---

## Shot Classification

Once we detect a hit, we classify the **shot type**:

```python
SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley', 'smash']

# Based on player position and ball trajectory
if player_at_net and ball_coming_down:
    shot_type = 'volley'
elif ball_very_high:
    shot_type = 'smash'
elif serving_position:
    shot_type = 'serve'
else:
    # Forehand or backhand based on stance
    shot_type = classify_groundstroke(player_pose, ball_position)
```

---

## Shot Notification Display

When hit detected:
1. Show "BALL SHOT!" in red (top-left)
2. Show "Player X: SHOT_TYPE" in green banner (top-center)
3. Record in Shot Analysis panel

---

## Key Takeaways

| Concept | Purpose |
|---------|---------|
| **Delta Y** | Detect direction change |
| **Zone filtering** | Distinguish hits from bounces |
| **Persistence check** | Filter false positives |
| **Classification** | Identify shot type |

---

## Future Improvements

1. **Pose estimation** - Actually see racket contact
2. **Audio analysis** - Sound of ball hitting racket
3. **Deep learning** - Train model on hit moments
4. **Player proximity** - Use actual player positions, not zones
