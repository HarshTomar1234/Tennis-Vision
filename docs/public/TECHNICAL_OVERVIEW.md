# Tennis-Vision - Technical Overview

How the pipeline actually works, stage by stage, for someone reading the code for the
first time or deciding whether to trust its output.

The README says what is measured. This says how it is computed and where it breaks.
Every stage below lists the same five things: **what it does, why it exists, the
algorithm, the key assumption, and the failure mode.** Trivial helpers are skipped.

One theme runs through all of it. Most of these stages can produce a plausible-looking
answer when they have no business producing one at all, and most of the engineering here
is about detecting that case and refusing rather than about squeezing out accuracy.

---

## The pipeline, in order

```
video
  ├─ 1  read_video                    frames + fps from the container header
  ├─ 2  frame-rate support gate       is this rate one we have evidence for?      [GATE]
  ├─ 3  PlayerTracker                 YOLOv8x + ByteTrack, every person
  ├─ 4  TrackNetBallTracker           3-frame stack, heatmap, largest blob
  ├─ 5  CourtLineDetector             ResNet-50, 14 keypoints, per frame
  ├─ 6  assess_court_fit              are those lines on actual paint?            [GATE]
  ├─ 7  select_two_players            six criteria over the whole clip
  ├─ 8  assess_selection              are they on opposite sides of the net?      [GATE]
  ├─ 9  MiniCourt homography          cv2.findHomography, RANSAC, cached
  ├─ 10 derive_shot_frames            candidates → classify → Viterbi decode      [GATE]
  ├─ 11 classify_floor_level          which frames are floor-valid anchors
  ├─ 12 smooth_trajectories           constant-velocity Kalman
  ├─ 13 detect_serve_frames           overhead AND from a baseline                [GATE]
  ├─ 14 classify_from_physics         evidence for Volley/Smash, or downgrade     [GATE]
  ├─ 15 pose forehand/backhand        MediaPipe body geometry
  ├─ 16 serve_speed                   feet → bounce, service-box gated            [GATE]
  ├─ 17 audit_rally                   what the ordering proves is missing
  ├─ 18 reconstruct_rally             closed-form ballistic 3-D                   [GATE]
  └─ 19 outputs                       CSV, summary JSON, 3-D scene, video, viewer
```

---

## 1 · Video ingestion

**What** Decodes every frame into memory and reads the frame rate from the container
header.

**Why** Frame rate is not cosmetic here. Every event threshold downstream is counted in
*frames*, so a wrong rate silently rescales the whole event path. Hardcoding 24 or 30, as
an earlier version did, meant every speed on a 25 fps clip was wrong by 20%.

**Algorithm** OpenCV `VideoCapture`, `CAP_PROP_FPS`.

**Assumption** The whole clip is one continuous camera take at a constant rate.

**Failure mode** A container with a missing or wrong fps header. The pipeline defaults to
30 for arithmetic but marks the clip **unsupported** rather than letting the default pass
as a measurement.

---

## 2 · Frame-rate support gate `utils/fps_support.py`

**What** Classifies the clip's rate as `supported` / `partially_supported` /
`unsupported` and attaches the reason to the log, `summary.json` and the rendered video.

**Why** Every threshold in the event path is in frames, and the classifier's two largest
weights are velocities in **pixels per frame**. Nothing is normalised by rate, so the same
tennis sampled faster or slower produces a different event set. Every clip this project
has measured on runs 23.6-30 fps, so the entire evidence base sits in one narrow band and
until this gate existed nothing said so.

**Algorithm** Threshold comparison against bands derived by resampling a reference clip's
ball track and re-running the real generators over it:

| fps | events/s | vs 30 fps | contact : bounce |
|---|---|---|---|
| 15 | 0.84 | -43% | 10 : 6 |
| 24 | 1.37 | -7% | 15 : 11 |
| **30** | **1.47** | baseline | **14 : 14** |
| 50 | 2.21 | +50% | 15 : 27 |
| 60 | 2.42 | +64% | 11 : 35 |

Supported 23-31, partially supported 18-50, unsupported outside.

**Assumption** The reference clip's event density is representative of tennis generally.

**Failure mode** Above 30 fps the measurement is a **lower bound**: it resamples a 30 fps
clip, so it invents intermediate points a real fast camera would have measured
independently, and it cannot model sharper motion or less blur. Real 60 fps footage is
likely worse than the table says. The clip is analysed anyway, with the caveat attached,
because refusing it outright would be less useful than analysing it honestly.

---

## 3 · Player detection and tracking `trackers/player_tracker.py`

**What** Finds every person in every frame and gives each a persistent track id.

**Why** Shot attribution needs to know *which* player struck the ball, and that requires
identity across frames, not just detection.

**Algorithm** YOLOv8x for detection, ByteTrack for association. ByteTrack matches
detections to existing tracks by IoU, and its distinguishing idea is that it also tries to
match the *low-confidence* detections rather than discarding them - which is what keeps a
partially-occluded player's track alive.

**Assumption** People are visible and separable enough for box-overlap association to work.

**Failure mode** Broadcast tennis puts 12-50 people in frame: players, ball kids, line
judges, the umpire, the crowd. Detection is not the hard part and a bigger detector does
not help - measured, and recorded in the README's rejected experiments. Choosing which two
are the players is the hard part, which is the next stage.

---

## 4 · Ball tracking `trackers/tracknet_ball_tracker.py`

**What** Locates the ball in each frame, or reports that it could not.

**Why** A tennis ball is a few pixels across, travels up to 200 km/h, motion-blurs into a
streak and disappears behind players and the net. A single-frame detector fails on it.

**Algorithm** TrackNet v2 - a VGG-style encoder/decoder taking **three consecutive frames
stacked channel-wise** (9 channels in) and emitting a per-pixel heatmap. The temporal stack
is the whole idea: a smear that is ambiguous in one frame is obvious as motion across
three. The heatmap is reduced to a position by taking the centroid of the **largest
connected component**, not the mean of all responding pixels.

That distinction is worth its own note. When the heatmap responds in two places - the ball
plus a line marking, or a distant shoe - the mean lands *between* them, at a point the ball
never occupied. Measured against the dataset's own labels:

| postprocess | detection rate | within 5px | median | p90 |
|---|---|---|---|---|
| mean of all pixels | 88.7% | 40.8% | 5.8px | 20.2px |
| **largest component** | 88.6% | **42.5%** | **5.4px** | **18.0px** |

The tail improves more than the median, which is the signature of removing blended
two-blob frames rather than of general smoothing.

**Assumption** The ball is the largest coherent moving response in the frame.

**Failure mode** **Detection rate is not accuracy, and the gap is large.** 88.6% of frames
get a position; only 42.5% of visible-ball frames are within 5px of the true centre. That
bounds every speed and landing position downstream. It does *not* cost event recall -
the funnel shows every labelled contact has a detected ball nearby.

---

## 5 · Court geometry `court_line_detector/`

**What** Locates 14 known court points in every frame.

**Why** Everything real-world - position, distance, speed - needs a mapping from pixels to
metres, and that mapping is defined by the court.

**Algorithm** ResNet-50 with a regression head predicting 28 numbers (14 x,y pairs),
re-run per frame so camera pan, tilt and zoom are tracked rather than assumed away.
Fine-tuned with geometric augmentation only (translation, scale, perspective, flip):
median keypoint error 4.03px → **2.90px**.

**Assumption** The camera sees enough of the court for the model to place all 14 points.

**Failure mode** **The model has no way to say "I don't recognise this."** It is a
regression head: given any image it returns 14 numbers. On unfamiliar footage it returns a
tidy quadrilateral that simply is not the court - often on the crowd. Everything
downstream then computes confidently from it. This is the single most dangerous failure in
the system, which is why the next stage exists.

---

## 6 · Court validity gate `utils/court_validity.py`

**What** Decides whether the detected court is trustworthy enough to measure from.

**Why** See above: a wrong court produces plausible numbers, and a plausible wrong number
travels further than a right one.

**Algorithm** Image evidence. A real court line is *painted brighter than the surface
beside it*, so the gate samples along every predicted line and asks whether those pixels
are brighter than pixels a few px to either side, by a margin. The fraction that pass is
the **line support score**. Below `MIN_LINE_SUPPORT = 0.22` the fit is rejected and every
dependent measurement is withheld.

**What does not work, and this is the instructive part.** Homography reprojection error is
useless here. Measured across 9 clips it ranged 1.40-1.88px on correct fits and 2.13px on a
visibly wrong one, with 14/14 RANSAC inliers every time. It measures whether the 14 points
are *self-consistent* - and a tidy quadrilateral on the stands is perfectly
self-consistent. Only evidence from the image itself distinguishes them.

**Assumption** Court lines are brighter than their surroundings. True on hard, clay and
grass in normal light.

**Failure mode** Heavy shadow, worn paint, unusual surfaces, or night matches with strong
line glare could in principle fool it in either direction. The threshold is calibrated on a
small sample and is a heuristic, not a proof.

---

## 7 · Player selection `utils/player_selection.py`

**What** Narrows every detected person to the two players and renumbers them 1 and 2.

**Why** Twelve to fifty people are detected. Two are playing.

**Algorithm** Six criteria scored per track and aggregated **across the whole clip**, then
one winner per court half. Aggregating over time is the key: in any single frame a line
judge can outscore a real player, but players are present for most of a rally and
incidental people are not.

Selecting from frame zero was the original approach and it failed on real footage - on one
clip the true player is a track id that does not exist at frame zero, because they were
off-screen at the serve.

**Assumption** Singles. Two players, one per side, present for most of the clip.

**Failure mode** Selecting a spectator or a line judge, which poisons every per-player
number. That is why the next stage exists.

---

## 8 · Player selection gate `utils/player_selection.assess_selection`

**What** Judges whether the selection could plausibly be two tennis players, and reports
`ok` / `degraded` / `failed`.

**Why** A clip in the evaluation set fits its court comfortably (line support 0.327,
against a 0.22 threshold) and then selects two tracks **on the same side of the net**, one
present for 40% of frames with a 199-frame hole in the middle. Nothing stopped it. The run
completed and published confident per-player statistics.

**Algorithm** Three checks, none of which needs any ground truth:
- **Opposite sides of the net.** Singles is played across the net, so two tracks on one
  half cannot both be players. Decisive on its own.
- **Coverage** - share of frames each player is present.
- **Longest continuous gap** - because ten scattered misses and one 200-frame hole have the
  same average and are completely different problems.

`failed` means the wrong people were selected and nothing per-player is a measurement.
`degraded` means the right people were lost for a stretch, so counts under-count. The
distinction matters and is reported.

**Assumption** Singles, and a court fit good enough to locate the net line.

**Failure mode** Doubles would fail it by construction, correctly - doubles is untested and
unsupported.

---

## 9 · Homography `mini_visual_court/mini_court.py`

**What** Maps a point in the video frame to a point on a top-down court, in metres.

**Why** Pixels are not distances. A player near the baseline and one at the net move very
different real distances for the same pixel displacement.

**Algorithm** A homography is the 3×3 projective transform relating two views of the same
**plane**. With 14 known correspondences it is over-determined, so `cv2.findHomography`
with RANSAC fits it while discarding outlier keypoints. Cached per unique keypoint tuple,
since consecutive frames often predict identically.

**Assumption - and this is the one that matters most.** A homography is only valid **on
the plane it was fitted to**, which here is the court floor. Everything not on the floor is
mapped wrongly, and the error grows with height.

**Failure mode** Projecting an airborne ball through the floor homography places it where
the camera ray through it *meets the ground*, which can be tens of metres away. Measured:
a serve contact 2.7 m in the air projected to a mini-court position outside the court
entirely. This is why ball positions are only trusted at floor level (stage 11) and why
3-D reconstruction exists (stage 18).

If the homography cannot be fitted at all, the code falls back to a nearest-keypoint
approximation that cannot correct perspective. That fallback is **counted and reported** -
it used to happen silently, which meant a run could quietly degrade to a materially
different algorithm and still report its numbers with full confidence.

---

## 10 · Event detection `utils/hit_bounce_classifier.py`

The critical path. Everything after it depends on getting these frames right.

### 10a · Candidate generation

**What** Proposes frames where *something* happened to the ball.

**Algorithm** Three generators feeding a union, because each is blind to a different event
shape:
- **y-reversal** - the ball changes vertical direction. Catches most contacts and bounces.
- **x-velocity** - a sharp change in horizontal velocity. Catches contacts that redirect
  the ball sideways without reversing it vertically. Alone it recalls *worse* than
  y-reversal (70.3% vs 76.0%); the **union** recalls 87.7%.
- **bounce generator** - dedicated, because x-velocity recalls only 12% of bounces.

Then `merge_nearby_candidates` collapses duplicates, since two generators firing on one
real event would otherwise double-count it.

**Failure mode** The merge window is a fixed number of frames, and it is the largest
remaining source of loss: **16.5% of contacts are lost in merging** - real events that are
genuinely closer together in time than the window. A further 15.4% are never proposed by
any generator.

### 10b · Hit versus bounce

**What** Decides whether each candidate is a racket contact or a floor bounce.

**Why** They look similar in a trajectory and mean completely different things.

**Algorithm** Logistic regression on four trajectory-shape features: ball height, |vertical
velocity change|, |horizontal velocity change|, and whether the horizontal direction
flipped. The last one carries the physics: **a bounce is a floor reflection and mostly
preserves horizontal velocity; a racket redirects it.** Measured - hits flip x-direction
71.8% of the time, bounces 2.1%.

**86.4% held-out accuracy**, split by *clip* rather than by event, so camera, lighting and
player correlations cannot leak between train and test.

**The instructive result:** the feature set was chosen on end-to-end F1, not on this
accuracy, and the two disagree.

| features | held-out accuracy | end-to-end shot F1 |
|---|---|---|
| height, vertical, horizontal | 84.1% | 0.737 |
| plus raw signed velocities | **89.3%** | **0.600** |
| plus horizontal reversal (shipped) | 86.4% | **0.824** |

The most accurate model on the benchmark is the worst in the product. The cause is a train
and serve mismatch: the dataset's velocities come from hand-annotated positions, while the
pipeline computes them from real detections with interpolated gaps.

**Failure mode** Features are in pixels per frame and are not normalised for resolution or
frame rate - the reason stage 2's gate exists.

### 10c · Rally decoding `utils/rally_decode.py`

**What** Re-labels the whole event sequence using the rules of tennis, instead of trusting
each event in isolation.

**Why** The classifier labels events independently at 86.4%, so roughly one in seven is
wrong. Independently that is respectable. In a *sequence* it is not, because the errors
compound into rallies that cannot physically happen - the self-audit found one player
hitting five times in succession.

**Algorithm** A rally is a grammar:

```
contact(near) → contact(near)   IMPOSSIBLE, the ball never crossed the net
contact(X)    → contact(Y)      legal, a volley taken before the bounce
contact(X)    → bounce          legal, the normal case
bounce        → contact(X)      legal, a groundstroke
bounce        → bounce          IMPOSSIBLE, a second bounce ends the point
```

The classifier emits a *probability*, not a hard label. Finding the most likely labelling
that obeys a grammar is exactly **Viterbi decoding** - the same technique that repairs
character errors in OCR, applied to a sport. State is (last label, which side last struck
the ball).

A third option matters: a candidate can be labelled **NOISE** and discarded. Relabelling
alone measured *worse* than not decoding at all (false positives 7 → 10), because every
repair pushes an event into the other class, and on a candidate set where 41% are not
events at all, those repairs land on noise. Letting the grammar *discard* is usually the
correct repair.

Measured: events the audit proves are missing fall **83 → 35** across 9 clips, with contact
recall on 40 labelled dataset clips **unchanged at 75.9%**.

**Assumption** Singles, one continuous rally, no missing player attribution.

**Failure mode** It cannot invent an event that was never detected, and does not try. When
it repeatedly overrules a *confident* classifier that signals an upstream problem, so those
overrides are counted and warned about rather than absorbed silently.

---

## 11 · Floor-level classification and 12 · Kalman smoothing

**What** Marks which frames are floor-valid anchors, then smooths the projected tracks.

**Why** Stage 9's assumption: the floor homography is correct only at floor level, which
means at a bounce or at a contact. In between, the ball has real height and must be
*interpolated between anchors*, not projected.

**Algorithm** A constant-velocity Kalman filter - predict position from the last position
and velocity, correct with the measurement, weighted by relative uncertainty.

**Failure mode, measured and instructive.** Forward-backward RTS smoothing was implemented
and **deliberately not enabled**: it buys complete coverage and ~1.5 points of recall for
4% worse median error. Two findings came out of it. Smoothing *across* a contact is
measurably worse, because a racket changes velocity discontinuously and a constant-velocity
smoother blends the incoming and outgoing velocities. And even per-flight it does not
improve median error, because TrackNet's error is not Gaussian - a 5.4px median against an
18.0px p90 is a heavy tail of gross mislocalizations, and a Kalman smoother *spreads* those
into neighbouring good frames instead of rejecting them.

A chi-square outlier gate in front of the smoother was worse still: median error 6.3px →
29.4px, and 207.9px at tight tuning. It diverges, because the constant-velocity prediction
is too poor to serve as a reference, so the gate rejects correct measurements and coasts on
a wrong track.

---

## 13 · Serve detection `utils/serve_detector.py`

**What** Identifies which contacts are serves, from physical evidence.

**Why** The original rule was "the first shot in a sequence is a serve." That only holds if
a clip begins exactly at the start of a point, and real clips are cut from mid-match - so
every "Serve" the pipeline ever reported was that heuristic firing, not a serve being
recognised.

**Algorithm** A serve is the only shot simultaneously **struck above the player's head**
*and* **from the baseline or behind it**. Both conditions must hold, each is independently
measurable, and a rejection reports which one failed.

**Failure mode** A high defensive lob struck from behind the baseline could satisfy both.
Rare, and the rally position usually disambiguates.

---

## 14 · Physics validation `utils/shot_physics.py`

**What** Tests Volley, Smash and Lob against physical facts, and downgrades to
"Groundstroke" when no evidence supports the label.

**Why** `ShotClassifier` decides Volley and Smash from court position alone, which has
never had ground truth and produced smashes in the middle of baseline rallies.

**Algorithm**
- **Smash** - struck above the head but *not* from a baseline (the serve test, zone
  inverted).
- **Volley** - **no bounce between it and the previous contact.** Volleying is by
  definition hitting before the bounce, so this is a fact about the event sequence rather
  than an inference from position.
- **Lob** - 3-D apex far above net height.

**What it turns out to be.** Measured across 9 clips and 81 shots: **1 positively
evidenced, 20 downgraded.** A 20-to-1 rejection ratio. On broadcast rallies almost nothing
is a volley or a smash, so this is a **validation filter**, not a classifier, and is
described as one. That is not a failure: a quarter of all shots would otherwise carry a
confident Volley or Smash label with nothing behind it.

**Failure mode** Volley detection depends on bounce recall (~80%), so a missed bounce can
fake a volley. The position clause is a guard against exactly that.

---

## 15 · Pose-based forehand / backhand `utils/pose_shot_classifier.py`

**What** Decides forehand from backhand using body geometry.

**Algorithm** MediaPipe pose inside the player's box, then a **body-relative** test:
whether the hitting arm crosses the shoulder midline horizontally. Body-relative makes it
independent of handedness, facing, and which side of the court the player is on.

**Failure mode - and this is honestly the weakest part of the system.** It scores **54%**
against ground truth, barely above chance on a two-class problem, and predicts forehand 89%
of the time. It is accurate on forehands (87-100%) and fails on backhands (0-47%).

The cause is upstream and more specific than "pose fails". MediaPipe finds the player on
**100%** of frames and then omits the landmarks of the *occluded arm* - which on a backhand
is the racket arm, missing 44-58% of the time. Both the rule and any classifier built on it
are reading a hand that is often not there.

Three approaches have been measured and rejected: the geometric rule (54%), a trained
classifier (76.3% on indoor THETIS footage, **53.6% on broadcast** - the transfer failed),
and a stronger pose model (SAM 3D Body, **66.4% against MediaPipe's 85.5% on identical
clips**). The last one is the most interesting: SAM 3D recovers the occluded arm by
inferring it from a body prior, and that inference destroys exactly the signal that decides
forehand from backhand. **MediaPipe's refusal to guess was a quality filter, not only a
loss.**

What is missing is not a better model. It is ground truth on broadcast footage.

---

## 16 · Serve speed `utils/serve_speed.py`

**What** Measures serve speed, or refuses to.

**Why** Every other speed derives from a floor projection of an airborne ball, which is
wrong by an amount depending on height and camera geometry (stage 9).

**Algorithm** Two things in a serve *are* reliably on the floor: the server's **feet at
contact**, and the ball's **first bounce**. Both project correctly. Horizontal distance
between them, divided by flight time, is a real measurement with no airborne projection
anywhere. Ignoring the vertical drop costs about 1%: a serve struck at 2.7 m landing 18 m
away travels √(18² + 2.7²) = 18.2 m.

**The gate:** the bounce must land **inside the correct service box**. The hit/bounce
classifier is ~86% accurate, so roughly one "bounce" in seven is not one, and a mid-flight
point used instead projects metres away. Measured: one such point inflated a flight to
26.6 m where a serve travels ~18 m - the whole of a 31% error against radar.

**Validated against broadcast radar**, which is third-party ground truth:

| clip | pipeline | radar |
|---|---|---|
| 1 | 213.4 km/h | 214.0 km/h |
| 2 | 164.1 km/h | 177.0 km/h |

Mean ratio 0.96, always at or below radar - which is what drag predicts, since radar reads
at contact and this is an average over the flight.

**Failure mode** Reported as **average flight speed**, never as a radar-equivalent contact
speed. When the landing is not observed inside the service box, no number is reported at
all.

---

## 17 · Rally self-audit `utils/rally_audit.py`

**What** Estimates what the detector *missed*, on the clip in front of you, with no ground
truth.

**Why** Every other number in this project comes from a labelled dataset and describes
average behaviour. A system that reports 72% recall on a benchmark and then says nothing
about the video you just uploaded has answered the wrong question.

**Algorithm** Tennis has enough structure to answer the right one. Where an *impossible*
ordering appears, an event was missed, and that can be stated without knowing the correct
sequence:
1. A player cannot hit twice in succession - the opponent's contact between them was missed.
2. A ball cannot bounce twice with play continuing.
3. Two contacts with no bounce between them is legal only near the net (a volley), so it is
   flagged softly when the hitter was deep.

**Failure mode** It is a **lower bound**. Two missed events in a row can restore a
valid-looking alternation, so the true count can be higher and never lower. It does not
guess where the missing events were and does not insert them.

---

## 18 · 3-D reconstruction `utils/trajectory_3d.py`

**What** Reconstructs the ball's flight path in 3-D between two floor-anchored events.

**Why** This is the *speed fix*, not a visualisation feature. Speeds from a floor
projection of an airborne ball are geometrically wrong, and no threshold tuning fixes a
geometry error.

**Algorithm** Between two events the ball is in free flight: horizontal motion constant,
vertical motion parabolic. If both endpoints and the flight time are known, the trajectory
is **fully determined** - a two-point boundary value problem with a closed-form solution:

```
vx  = (x1 - x0) / T
vy  = (y1 - y0) / T
vz0 = (z1 - z0 + ½gT²) / T        from  z(T) = z0 + vz0·T - ½gT²
```

No fitting, no initial guess, no convergence risk.

**The key design decision:** height is **modelled** from contact anchors, not fitted from
the image. At broadcast camera angles, raising the ball and pushing it further away move it
in almost the same image direction, so a free ballistic fit can match the picture to a pixel
while being metres wrong in space. Modelling is the honest choice and is labelled as such.

**Rejection gates:** flights longer than 1.5 s (not one flight, but several with the events
between them missed), endpoints more than 5 m outside the court, bounce-to-bounce spans (no
racket contact at either end), same-player-at-both-ends, non-net-crossing contact pairs, and
speeds above physical bounds.

**Not every segment is a shot.** Only a segment that *begins* at a racket contact is a ball
leaving a racket. One that begins at a bounce is the post-bounce leg travelling to the
receiver: real geometry, correctly reconstructed, and not a shot. On the reference clip, 25
segments reconstruct, of which 13 are shots, 11 are post-bounce legs and 1 is an outlier.

**Failure mode, ordered by measured size** - and the ordering was surprising:

| source of error | mean | median | worst |
|---|---|---|---|
| **event timing (±2.4 frames)** | **12.5%** | 12.2% | 28.0% |
| contact height (±0.20 m) | 0.7% | 0.2% | 3.5% |
| ball localization (±0.09 m) | 0.1% | 0.1% | 0.7% |

Event timing dominates by roughly 18× over contact height, and scales as 1/T - flights
under 0.5 s average 23.9% sensitivity, over 1.0 s average 6.5%. A reconstructed speed is
about as accurate as the event detector is *punctual*, and no amount of better geometry
improves it. **Drag and spin are not modelled**, and adding them would be modelling the
small terms while the large one goes unaddressed.

---

## 19 · Outputs

| File | Contents |
|---|---|
| annotated `.avi` | Video with overlays, plus a warning banner when a gate failed |
| `summary_*.json` | Clip-level verdicts: court, frame rate, players, ball, rally decoding, calibration, shot classification, 3-D speed |
| `stats_*.csv` | Per-frame running player statistics |
| `trajectory3d_*.json` | Reconstructed segments with per-segment speed status and reason |
| `*.html` | Self-contained interactive 3-D viewer, no CDN, no build step |

Every quality verdict carries a plain-language reason, so a consumer can act on it without
reading this document.

---

## The evidence and refusal system

Not a stage. It is the constraint the whole design obeys.

Valid outputs include `UNKNOWN`, `UNAVAILABLE`, `INSUFFICIENT_EVIDENCE`, `OUTLIER`,
`REJECTED` and `UNSUPPORTED`. None of them is ever replaced with a plausible-looking
number.

The reasoning is that **a silent fallback to a materially different algorithm is worse than
a crash**, because the output still looks right. A crash gets investigated; a confident
wrong number gets screenshotted and shared. Three real instances were found and fixed in
this codebase:

- an unmappable player position became the **centre of the court**, and flowed into
  distance and speed as though observed;
- a failed homography silently substituted the **nearest-keypoint approximation**, the
  method this project describes as the old and wrong one;
- requesting the SAM 3D pose backend without its weights fell through to MediaPipe in
  silence, despite the two measuring 85.5% and 66.4% on identical clips.

All three now omit, count and report. That pattern - *measure it, gate it, say so* - is
what the project is actually for.

---

## Reading order for the code

1. `main.py` - the orchestration, top to bottom.
2. `utils/hit_bounce_classifier.py` - `derive_shot_frames` is where the shot numbers come
   from.
3. `utils/rally_decode.py` - the most interesting algorithm here.
4. `utils/court_validity.py` and `utils/player_selection.py` - the two gates that decide
   whether anything is reported at all.
5. `utils/trajectory_3d.py` - the physics and its measured uncertainty.
6. `eval/` - every number in the README traces to a script here.
