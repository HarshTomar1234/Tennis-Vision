# Tennis-Vision

[![CI](https://github.com/HarshTomar1234/Tennis-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshTomar1234/Tennis-Vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Tennis match analysis from a single broadcast camera: ball tracking, court geometry,
player tracking, shot classification and 3-D trajectory reconstruction.

<div align="center">
  <img src="frame_images/tennis_analysis_quarter_frame53.png" width="820" alt="Annotated output frame">
</div>

## The one thing that makes this different

Every number this project reports carries the evidence for it, or it is not reported.

That reads like a slogan, so here is what it means in practice. The pipeline refuses to
print a serve speed when it cannot see the ball land. It flags a clip whose court fit
failed instead of computing real-world speeds from a court fitted to the crowd. It labels
a ball height "unknown" rather than dressing up a guess. And this README publishes the
numbers that make the project look worse alongside the ones that make it look better,
because the difference between them is usually the interesting part.

Two examples from this repository:

- Ball detection is usually quoted as a "detection rate". Ours is 88.6%. Measured against
  hand-labelled ground truth, only **42.5%** of visible-ball frames are located within
  5px. Both numbers are true and they measure different things.
- Contact and bounce detection scores **87.6% recall given perfect ball positions** and
  **72.0% running real detection end to end**. The second one is what you actually get.

Every figure below names the script that produced it.

## Quickstart

```bash
git clone https://github.com/HarshTomar1234/Tennis-Vision.git
cd Tennis-Vision

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e .

tennis-vision download-models       # about 140 MB
tennis-vision analyze input_videos/input_video_2.mp4 -o output/demo.avi
```

`download-models` prints one manual step. The TrackNet ball weights belong to their
original author and their licence is unstated, so they are fetched from that project
rather than rehosted here. Everything else downloads automatically.

Add `--max-frames 60` for a fast check before committing to a full run.

Outputs land in `output/`: an annotated video, a per-frame stats CSV, a run summary JSON,
and an interactive 3-D viewer as a single self-contained HTML file with no external
dependencies.

## What it does

**Ball tracking.** TrackNet, with the position taken from the largest connected heatmap
response rather than the mean of all responding pixels. Ball positions map to the court
through the floor homography only at floor level, meaning at a bounce or a racket contact.
While the ball is airborne the floor homography does not apply to it, so those frames
interpolate between floor-valid anchors instead of being projected as though the ball were
on the ground.

**Court geometry.** ResNet-50 keypoint regression, 14 points, re-detected per frame so
camera pan and tilt are handled, then a real perspective homography via
`cv2.findHomography`.

**Court validity gate.** The keypoint model is a plain regression head with no way to say
"this camera angle is outside my training distribution". On unfamiliar footage it returns a
tidy quadrilateral that is simply not on the court, and every real-world measurement
downstream is then computed from it and reported with full confidence. That failure is
worse than a crash because the output looks plausible. The gate samples along every
predicted line and asks whether those pixels are actually brighter than the surface a few
pixels to either side. A predicted line lying on the crowd fails; one lying on paint
passes.

**Player tracking.** YOLOv8x with ByteTrack, then a six-criteria selection aggregated
across the whole clip. Frame-zero selection was the original approach and it failed on real
footage: on one eval clip the true player is a track id that does not exist at frame zero.

**Event detection.** Three candidate generators feed a union, because each is blind to a
different event shape. y-reversal and x-velocity are hit-shaped by construction, and a
dedicated bounce generator covers what they miss. A trained trajectory classifier then
splits contacts from bounces.

**Serve detection and speed.** A serve is the only shot that is simultaneously struck
above the player's head and from the baseline or behind it. Both conditions must hold, each
is independently measurable, and a rejection reports which one failed. Speed comes from the
contact and the landing, both floor-valid, and is not reported at all when the landing is
never observed inside the service box.

**3-D reconstruction.** Between two known contacts the ball follows a parabola whose
curvature is gravity and whose endpoints are known heights, so there are no free
parameters. Height is modelled from those anchors rather than measured from the image,
because at broadcast camera angles raising the ball and pushing it further away move it in
almost the same image direction. A free ballistic fit can match the picture to a pixel
while being metres wrong in space.

**Shot classification.** Rule-based serve, forehand, backhand, volley and smash, with a
pose-based forehand and backhand upgrade via MediaPipe. The forehand/backhand half of this
is measured at 54% against ground truth and is documented under Limitations as unreliable.

The pose test is body-relative: whether the hitting arm crosses the shoulder midline
horizontally, which makes it independent of handedness, facing, and which side of the
court the player is on. That is the right idea and it is not sufficient, because a volley
is played with the body square to the net and the arm never crosses the midline at all.

## Optional: SAM 3D Body pose backend

MediaPipe drops the racket arm on 44-58% of backhands (see Limitations). SAM 3D Body
predicts a whole-body mesh and infers occluded joints instead of dropping them. On the
exact frames MediaPipe could not complete, it returned keypoints on **6 of 6**
(`eval/sam3d_occluded_arm_test.py`), and on a full backhand clip it completed 100% of
frames against MediaPipe's 88%.

It runs on **contact frames only**, roughly 15 per clip. At about 1.6s per frame a
per-frame pass would take half an hour on a mid-range GPU; 30 inferences takes under a
minute.

```bash
# 1. request access, then create a read token, then put HF_TOKEN in .env
python scripts/download_sam3d_body.py

# 2. the inference code is a separate repository
git clone https://github.com/facebookresearch/sam-3d-body.git
export SAM3D_BODY_CODE=/path/to/sam-3d-body

# 3. enable it
#    configs/config.yaml -> pipeline.use_sam3d_pose: true
```

Off by default. If it is switched on and the weights are absent the pipeline uses
MediaPipe and says so, rather than falling through in silence: the two backends measure
85.5% and 66.4% balanced on identical clips, so a run that quietly used the other one is
not the run that was asked for.

**On licensing.** The weights are under Meta's SAM License, not MIT. That licence grants
free use, modification and derivative works, and requires anyone *redistributing* the
materials to pass the same terms along. This project therefore does not bundle them:
doing so would have an MIT licence make a promise about Meta's weights it has no standing
to make. You fetch them under terms you accept directly, which is the same arrangement
already used for the TrackNet weights.

Verified working on a GTX 1050 Ti (4.3 GB) in float32. Do not wrap inference in
`torch.autocast`: the MHR head is a TorchScript module and raises `NotImplementedError`
inside one. Casting the weights fails in both directions as well.

## Measured results

Scripts marked `(dataset)` need a third-party dataset that is over 7 GB and not
redistributable. See `datasets/README.md` for sources. Everything else runs against what
ships in this repository plus the downloadable weights.

### Ball localization

Against the original TrackNet dataset's own hand-labelled coordinates, 16 clips.

| Metric | Result | Script |
|---|---|---|
| Detection rate (a position was output, **not** an accuracy) | 88.6% | `eval/ball_localization_accuracy.py` (dataset) |
| Localization error vs ground truth | median **5.4px**, 90th percentile 18.0px, at 360x640 | same |
| Located within 5px of the labelled centre | 46.8% of outputs, 42.5% of visible-ball frames | same |

Detection rate and accuracy are not the same measurement, and the gap here is large. The
detector reliably finds roughly where the ball is and is not pixel-precise. That is
adequate for trajectory shape, bounce timing and speed across a flight. It is marginal for
exact landing coordinates.

It is worth being precise about what this does and does not cost. It bounds the accuracy of
speeds, 3-D reconstruction and landing positions. It does **not** cost event recall: the
funnel below shows every labelled contact has a detected ball near it, so nothing is missed
for want of a detection.

### Contact and bounce event detection

Two questions with very different answers. Both are published, because the gap between
them is the honest cost of detection noise.

**Given perfect ball positions**, feeding the dataset's labelled coordinates straight into
the candidate generators. This isolates the generators and is an upper bound, not shipped
behaviour. 91 clips.

| Configuration | Recall | Precision | Script |
|---|---|---|---|
| y-reversal only | 75.8% | 88.9% | `eval/retest_union_candidates_full_pipeline.py` (dataset) |
| y-reversal + x-velocity union | 87.6% | 90.3% | same |

**Running real detection end to end**, which is what the pipeline does. 10 clips, 76
labelled contacts, trajectory classifier only.

| Configuration | Recall | Precision | F1 | Mean offset |
|---|---|---|---|---|
| mean of all heatmap pixels, chained clustering | 48.7% | 92.5% | 0.638 | 3.6 frames |
| largest connected component, chained clustering | 51.3% | 92.9% | 0.661 | 3.2 frames |
| **largest connected component, bounded clustering** (shipped) | **68.4%** | **92.9%** | **0.788** | **2.4 frames** |

Script: `eval/event_detection_on_real_detections.py` (dataset). Rows above are the same 10
clips throughout so the comparison is controlled.

On a wider 25-clip sample, 164 labelled contacts:

| Metric | Result |
|---|---|
| Recall | **72.0%** |
| Precision | **95.9%** |
| F1 | **0.822** |
| Mean offset | 2.4 frames |
| Per-clip recall | min 50%, median 71%, max 100% |
| Clips below 40% recall | **0 of 25** |

The per-clip row matters more than the aggregate. A 72% mean could hide clips that fail
completely, and it does not: the worst clip in the sample still recovers half its contacts,
and none scores zero.

Where the remaining misses go, attributed by `eval/event_recall_funnel.py` across 12 clips
and 91 labelled contacts:

| Stage | Share of all contacts |
|---|---|
| reported | 68.1% |
| ball never detected nearby | **0.0%** |
| no candidate proposed | 15.4% |
| lost in candidate merging | 16.5% |
| rejected by the classifier | **0.0%** |

Detection reaches every labelled contact and the classifier discards none. Everything still
missing is lost in candidate generation or in merging, which is the opposite of what this
project assumed before the funnel existed.

### Hit versus bounce classification

Trajectory-only logistic regression on ball height, vertical and horizontal velocity
change, and whether the ball reversed horizontally. No player position needed. Trained on
820 events, tested on 214 held out, split by clip rather than by event so camera, lighting
and player correlations cannot leak.

**86.4% held-out accuracy.** See `eval/train_hit_bounce_classifier.py`.

The feature set was chosen on end-to-end F1, not on this accuracy, and the two disagree:

| Features | Held-out accuracy | End-to-end shot F1 |
|---|---|---|
| height, vertical, horizontal | 84.1% | 0.737 |
| plus raw signed velocities | **89.3%** | **0.600** |
| plus horizontal reversal (shipped) | 86.4% | **0.824** |

The most accurate model on the benchmark is the worst in the product. The cause is a train
and serve mismatch: the dataset's velocities come from hand-annotated positions, while the
pipeline computes them from real detections with interpolated gaps. Raw signed velocities
took the largest weights and did not survive contact with real data.

### Court keypoints

Held-out validation split of the TennisCourtDetector dataset, 2,211 images.

| Metric | Base weights | Fine-tuned (shipped) | Script |
|---|---|---|---|
| Median keypoint error | 4.03px | **2.90px** | `eval/court_keypoint_accuracy.py` (dataset) |
| Images with all 14 keypoints within 25px | 96.8% | **98.3%** | same |
| Real clips passing the court-validity gate | 4/9 | **8/9** | `eval/court_validity_calibration.py` (dataset) |

Fine-tuned with geometric augmentation only: translation, scale, perspective and flip.
Per-surface error is near-identical (hard 3.90px, clay 4.58px, grass 4.65px), so surface is
not the weakness. Camera framing is. Validated on Wimbledon grass the model had never seen.

The validity gate was calibrated by measurement, and one obvious approach was discarded:
homography reprojection error is useless for this. Across 9 clips it ranged 1.40 to 1.88px
on correct fits and 2.13px on a visibly wrong one, with 14 of 14 RANSAC inliers every time.
It measures whether the 14 points are self-consistent, and a tidy quadrilateral on the
stands is perfectly self-consistent.

### Serve speed

Validated against broadcast radar, which is third-party ground truth rather than a
self-generated reference.

| Clip | Pipeline | Broadcast radar |
|---|---|---|
| 1 | 213.4 km/h | 214.0 km/h |
| 2 | 164.1 km/h | 177.0 km/h |

Mean ratio 0.96, always at or below radar, which is what aerodynamic drag predicts given
that radar reads at racket contact. Reproduce with `eval/serve_speed_accuracy.py`.

### Rally coherence

A rally has orderings that cannot happen: a player cannot hit twice in succession, and a
ball cannot bounce twice with play continuing. Where the reported sequence contains one,
an event is provably missing, and that is checkable with no ground truth at all. So this
is measured on every clip rather than only the labelled one.

The hit/bounce classifier labels each event alone, so its errors compound into impossible
rallies. `utils/rally_decode.py` re-labels the sequence as a whole, keeping the most likely
labelling the rules permit, and discarding candidates that have no legal place in it.

| Metric | Without decoding | With decoding | Script |
|---|---|---|---|
| Events the ordering proves are missing (9 clips) | 83 | **35** | `eval/rally_coherence.py` |
| Clips improved | | 10 of 10, none worse | same |
| Contact recall, 40 labelled clips | 75.9% | **75.9%** | `eval/event_detection_on_real_detections.py` |
| Shot recall on the labelled clip | 100% | 100% | `eval/shot_frame_accuracy.py` |
| Shot false positives on the labelled clip | 7 | 8 | same |

The coherence number alone would be trivially gamed by discarding every candidate, so it is
only quoted next to the recall it cost. Discarding does cost recall as it gets more
aggressive, monotonically, which is why the shipped setting is the smallest one that gets
the full coherence benefit rather than the one with the best coherence score.

What it costs: one extra false-positive shot on the reference clip. That clip has 7 labelled
shots, so a single event is inside its noise, and the dataset-scale recall it is traded
against is unchanged. The cost is listed rather than left out.

It cannot recover an event that was never detected: a missing contact stays missing, and
the audit still reports it.

### Reference clip, end to end

One clip with 7 hand-labelled shots. Listed because it is the reproducible demo, not
because 7 events settle anything. The dataset-scale numbers above are the ones to trust,
and on precision they disagree with this clip.

| Metric | Result | Script |
|---|---|---|
| Shot-frame recall | 7/7 found, mean offset 7.9 frames | `eval/shot_frame_accuracy.py` |
| Shot-frame precision | 46.7%, 8 false positives in 15 reported, F1 0.636 | same |
| Ball speed plausibility (a range check, **not** accuracy) | 21/21 within physical bounds | `eval/speed_accuracy.py` |

### Test suite

**350 unit and integration tests** (`pytest tests/`), covering ball-state classification,
Kalman and RTS smoothing including the physical speed-plausibility gate, mini-court
coordinate mapping, trajectory drawing, pose-based shot classification, the hit and bounce
classifier and its feature contract, the rally grammar and its decoder, the no-ground-truth
rally audit, TrackNet postprocessing geometry, detection-cache keying, and packaging
integrity.

The end-to-end smoke test runs genuine fresh detection and depends on no cached artefacts,
so it fails for everyone if the pipeline breaks.

## What we tried that did not work

Published because negative results are expensive to produce and cheap to reuse. Each of
these was implemented, measured, and rejected on the number.

**Homography reprojection error as a court-validity signal.** Cannot distinguish a court
fitted to the court from one fitted to the stands, for the reason given above.

**Raising the ball detector's heatmap threshold, and changing its minimum cluster size.**
Thresholds from 0 to 128 and cluster sizes from 3 to 10 all land within noise of each other.
The shipped configuration is already at its optimum for this postprocess. The remaining
error is in the network's output, not in how it is thresholded.

**A larger or newer YOLO for player detection.** Detection is already saturated: on the
reference clip YOLOv8x finds 11 to 14 people per frame and the pipeline needs 2. The hard
problem is selecting which two are the players, which is our own logic, not the detector's.

**RTS forward-backward smoothing of the ball trajectory.** Buys complete coverage and about
1.5 points of recall for 4% worse median error. Two useful findings came out of it.
Smoothing across a contact is measurably worse than smoothing between contacts, because a
racket hit changes velocity discontinuously and a constant-velocity smoother run through
one blends the incoming and outgoing velocities. And even applied per flight span it does
not improve median error, because TrackNet's error is not Gaussian: a 5.4px median against
an 18.0px 90th percentile is a heavy tail of gross mislocalizations, and a Kalman smoother
spreads those into neighbouring good frames instead of rejecting them. Shipped as a tested
utility, off by default.

**A chi-square outlier gate in front of that smoother.** Catastrophic on real data, taking
median error from 6.3px to 29.4px, and to 207.9px at tight tuning. It diverges: the
constant-velocity prediction is too poor to serve as a reference, so the gate rejects
correct measurements and coasts on a wrong track. Gating is the right idea for choosing
among several candidate detections per frame, which is a different job. Our postprocess
emits exactly one position, so a gate can only discard.

**The ratio of outgoing to incoming vertical speed as a bounce signal.** The physics is
sound, since a floor bounce can only lose vertical speed while a racket adds it, and the
medians do separate: 2.04 for hits against 0.98 for bounces. But it reaches only 65.7%
accuracy on 1,034 labelled events, because at broadcast camera angles vertical pixel speed
is substantially measuring depth rather than energy.

**Replacing MediaPipe with SAM 3D Body for forehand/backhand.** SAM 3D Body recovers the
occluded racket arm that MediaPipe drops on 44-58% of backhands, taking usable clips from
171 to 200 and from a 55/45 class skew to a perfect 100/100 balance. It also made the
classifier substantially worse:

| configuration | balanced | forehand | backhand |
|---|---|---|---|
| MediaPipe, shared clips only | **85.5%** | 83.5% | 87.4% |
| SAM 3D, shared clips only | 66.4% | 71.6% | 61.3% |

Measured on identical clips, so this is not about the extra data. The landmark mapping was
verified against MediaPipe on frames where both succeed and agrees within 1-3 pixels, so it
is not an integration error either.

The damage is confined to position, not motion:

| feature subset | MediaPipe | SAM 3D |
|---|---|---|
| wrist side only | 78.6% | **51.8%** (chance) |
| speed and reach only | 63.1% | 65.8% |

On frames where the arm is occluded, MediaPipe declines and SAM 3D infers the arm from a
body prior. That inference is anatomically plausible and it is still a guess about where
the racket is, and it destroys exactly the signal that decides forehand from backhand.

The lesson is one this project already claims to hold: MediaPipe's refusal was a quality
filter, not only a loss. A model that always answers is not better than one that knows when
to stay quiet. The optional backend remains in the tree for its coverage, meshes and camera
estimates, and is off by default.

**A trained forehand/backhand classifier on pose features.** Scores 76.3% balanced on
THETIS with subject-grouped splits and repeated cross-validation, against 54% for the
hand-crafted geometry it was built to replace. On real broadcast images it scores 53.6%,
statistically the same as the rule, with the bias flipped rather than removed. Everything
about the training was methodologically sound and it would still have been a regression in
production. Kept, measured, not wired in.

**"The first shot in a sequence is a serve."** This was the original serve rule. It only
holds if a clip begins exactly at the start of a point, and ours are cut from mid-match, so
every "Serve" the pipeline ever reported was this heuristic firing rather than a serve being
recognised.

## Limitations

**Wrong or unvalidated today:**

- **Rally and groundstroke speeds are unvalidated.** 3-D reconstruction produces 62 to
  153 km/h with a mean of 92 on the reference clip, and the physics is verified, but no
  ground truth exists for non-serve shots. Serve speed is validated; rally speed is not.

  The dominant uncertainty is not the one you would expect. `trajectory_3d.py` used to
  list drag, spin and contact height as its honest limits, and all three are smaller than
  event timing, which the list omitted. Speed is distance over flight time, flight time
  comes from event frames, and the measured mean event offset is 2.4 frames:

  | source of error | mean | median | worst |
  |---|---|---|---|
  | **event timing (±2.4 frames)** | **12.5%** | 12.2% | 28.0% |
  | contact height (±0.20 m) | 0.7% | 0.2% | 3.5% |
  | ball localization (±0.09 m) | 0.1% | 0.1% | 0.7% |

  Measured on 25 real segments by `eval/speed_timing_sensitivity.py`. Sensitivity scales
  as 1/T, so flights under 0.5 s average 23.9% and flights over 1.0 s average 6.5%. A
  reconstructed speed is about as accurate as the event detector is punctual, which is
  why adding drag or Magnus terms would be modelling the small terms first.

- **Not every reconstructed segment is a shot, and the report says which.** A rally
  alternates contact, bounce, contact, so only a segment that begins at a racket contact
  is a ball leaving a racket. On the reference clip 25 segments reconstruct, of which 13
  are shots, 11 are post-bounce legs travelling to the receiver, and 1 is flagged as an
  outlier (struck and landing on the striker's own side without crossing the net, which
  breaks the free-flight assumption). Averaging all 25 into one figure is what previously
  put an 18 km/h reading next to a 170 km/h one and called both "shot speed".
- **Roughly a quarter to a third of contacts in a rally are missed** (72.0% recall on real
  detections, 95.9% precision). Reported events are overwhelmingly real, so the shot count
  is an under-count rather than noise.
- **Forehand versus backhand is unreliable, and measured as such.** Against THETIS ground
  truth (120 clips, 8 classes, balanced by construction) the pose geometry scores **54%**,
  which is barely above chance on a two-class problem, and it predicts forehand **89%** of
  the time. It is accurate on forehands (87-100%) and fails on backhands (0-47%).

  Two separate causes, both measured with `eval/forehand_backhand_on_thetis.py`. Choosing
  which wrist is the hitting hand accounts for about 24 points. The side projection itself
  accounts for the rest: even given the correct hand it tops out at 78%, and on volleys it
  reaches only 27%, because a volley is blocked with the body square to the net and the
  wrist never crosses the shoulder midline the test depends on.

  A trained classifier was built to replace it and **did not survive the transfer test**.
  On THETIS it scores 76.3% balanced with subject-grouped splits, fixing the asymmetry
  (forehand 78.9%, backhand 73.7%). On real broadcast images it scores **53.6%**, which is
  the same as the rule, and the bias flips direction rather than merely weakening
  (forehand 39.1%, backhand 68.0%). See `eval/validate_on_broadcast_images.py`.

  That test is handicapped: 8 of the 20 features describe motion and a still image has
  none, so it is a lower bound rather than a like-for-like comparison. But it is the only
  broadcast evidence that exists, and it does not support shipping the classifier. It is
  trained, measured and committed, and deliberately not wired into the pipeline.

  The common cause of both failures is upstream, and it is more specific than "pose
  fails". MediaPipe finds the player on **100%** of frames and then omits the landmarks of
  the occluded arm. On a backhand that is the racket arm:

  | class | frames | no pose | most-missing landmarks |
  |---|---|---|---|
  | backhand_volley | 117 | 0 | right elbow 68%, right wrist **58%** |
  | backhand | 142 | 0 | right elbow 47%, right wrist **44%** |
  | forehand_volley | 120 | 0 | left elbow 9%, left wrist 5% |
  | forehand_flat | 146 | 0 | left elbow 25%, left wrist 23% |

  Forehand versus backhand is decided by where that arm is, so both classifiers are
  reading a hand that is often not there. That is why a stronger pose model is the fix
  rather than a better classifier on the same landmarks, and it is measured by
  `eval/pose_availability_at_contacts.py`.

- **Volley and smash labels come from position rules with no ground truth.** Serve is now
  detected from physical evidence. Those two are not.
- **The learned temporal shot classifier is not wired into the pipeline.** It scores 73.4%
  on unseen subjects across 6 classes, but it is trained on THETIS indoor demonstration
  footage and its transfer to broadcast video is unmeasured.
- **Player detection has no ground-truth eval.**
- **Ball height is modelled, not measured**, and cannot be otherwise from this camera
  geometry. It is anchored at known contact heights and interpolated by gravity, so it
  degrades whenever a contact is missed.

**Out of scope right now:**

- **Ground-level cameras fail.** Validated on broadcast and elevated fixed-camera footage
  only. The validity gate flags these rather than reporting wrong numbers.
- **Doubles and amateur footage are untested.** Every evaluation clip is broadcast singles.
- **Frame rates outside 23 to 31 fps are not supported.** See below.

## Supported inputs

Every threshold in the event-detection path is counted in **frames**, and the two largest
weights in the hit/bounce classifier are velocities in **pixels per frame**. None of it is
normalised by frame rate, so the same tennis sampled at a different rate produces a
different event set. Every evaluation clip here runs between 23.57 and 29.82 fps and both
input videos are 30.0, so the entire measured record sits inside one narrow band.

| Frame rate | Status | What to expect |
|---|---|---|
| 23 to 31 fps | **Supported** | The range every number on this page was measured on |
| 18 to 23, 31 to 50 fps | **Partially supported** | Runs, and event counts are wrong in a known direction |
| below 18 or above 50 fps | **Unsupported** | Event counts should not be treated as measurements |

Measured by resampling the reference clip's ball track and running the real generators and
the real classifier over it. Events per second is the comparison that means something,
since the rally contains the same contacts however fast it was sampled:

| fps | events/s | vs 30 fps | contact:bounce |
|---|---|---|---|
| 15 | 0.84 | -43% | 10:6 |
| 18 | 1.05 | -29% | 9:11 |
| 24 | 1.37 | -7% | 15:11 |
| **30** | **1.47** | **baseline** | **14:14** |
| 36 | 1.79 | +21% | 14:20 |
| 50 | 2.21 | +50% | 15:27 |
| 60 | 2.42 | +64% | 11:35 |

Two failures in opposite directions. Too slow and real events are never proposed. Too fast
and the generators fire more often while the per-frame velocities shrink, so the classifier
calls almost everything a bounce: 11 contacts to 35 bounces on a rally with roughly 14 of
each. **60 fps is ordinary footage and it is genuinely not supported today.**

One caveat on the method, in the direction that matters: below 30 fps the resampling
discards information, which is what a slower camera does. Above 30 fps it invents
intermediate points a real fast camera would have measured independently, and cannot model
sharper motion or reduced blur. The high-rate rows are a **lower bound** on the disruption,
not an estimate of it.

The pipeline still analyses an out-of-band clip rather than refusing it, and stamps the
rendered video, the log and `summary.json` with what it cannot vouch for. Normalising the
event path to seconds and metres and retraining is the real fix; it is post-launch work,
because it would invalidate every number above in the process.

Also unsupported: doubles, ground-level cameras, and any clip whose court fit fails the
validity gate. All three are reported rather than guessed at.

## Roadmap

Ordered by measured value, not by interest.

1. **Labelled broadcast video for shot types, before any more modelling.** Three
   approaches have now been measured on forehand/backhand and none is trustworthy: the
   geometric rule (54%), a trained classifier (76.3% indoors, 53.6% on broadcast), and
   swapping in a stronger pose model (66.4%, worse than MediaPipe on identical clips). Each
   was chosen on reasoning and rejected on measurement. What is missing is not a better
   model, it is ground truth on the footage this actually runs on, which
   `tools/label_shots.py` produces.

2. **A stronger pose model used as a supplement, not a replacement.** SAM 3D Body gives
   100% landmark coverage, body meshes and camera parameters. Used naively it is worse than
   MediaPipe (above), but its camera estimate is an independent check on the homography,
   which is currently validated only by image evidence.
3. **A smarter merge decision.** A fixed frame window is the wrong instrument: it still
   loses 16.5% of contacts, which are real events genuinely closer together than the
   window. Two candidates should merge because the trajectory says they describe one
   physical event, not because they are near each other in time. This is the largest
   remaining bucket.
4. **Candidate generation.** A further 15.4% of contacts are never proposed by any of the
   three generators, so they are blind to some event shape. Finding out which is a
   labelling exercise, not a modelling one.
5. **Ball localization.** Median 5.4px, 18.0px tail. Demoted from first place, because the
   funnel shows it costs zero recall: every labelled contact has a detected ball nearby.
   It still bounds the accuracy of speeds, 3-D reconstruction and landing positions, which
   is why it stays on the list.
6. **Audio impact detection.** A racket strike and a floor bounce are sharp broadband
   transients that a broadcast mix carries clearly. Audio cannot say where the ball is, but
   it says precisely when it was struck, including while the ball is hidden behind a player
   or the net. It is the most promising route to the contacts no generator proposes.
7. **Player-height-normalised contact distance.** The current threshold is a raw pixel
   constant, which is wrong at different resolutions and at different depths within a single
   frame. Dividing by the player's own pixel height converts pixels to metres at that
   player's depth without needing to know the ball's height.
8. **Geometric court detection.** The four cross-court lines have a projective-invariant
   cross-ratio that is identical under any camera view, so a court can be found by searching
   for that signature rather than by a learned model. This would remove the per-surface
   fine-tuning dependency entirely.
9. **Broadcast ground truth for shot types**, so the temporal classifier can be validated
   and wired in, or dropped.

## Reproducing the numbers

```bash
pip install -e ".[dev]"
pytest tests/                                       # 350 tests, needs the weights
pytest tests/ -m "not slow"                         # 349, what CI runs, no weights

python eval/shot_frame_accuracy.py                  # reference clip, ships with repo
python eval/speed_accuracy.py                       # reference clip, ships with repo
python eval/rally_coherence.py                      # any clips, needs no ground truth
python eval/speed_timing_sensitivity.py             # reads a run's own 3-D output

python eval/ball_localization_accuracy.py --clips 16          # needs dataset
python eval/event_detection_on_real_detections.py --compare   # needs dataset
python eval/train_hit_bounce_classifier.py                    # needs dataset
python eval/court_keypoint_accuracy.py                        # needs dataset
```

## Repository layout

```
configs/              config.yaml, every tunable parameter
constants/            court dimensions, physical plausibility bounds
court_line_detector/  ResNet-50 court keypoint regression
eval/                 every number in this README traces to a script here
mini_visual_court/    mini-court mapping and trajectory drawing
models/               small trained weights (committed); large weights fetched by script
notes/                CV concept write-ups
scripts/              download_models.py, build_clip_suite.py
tests/                350 unit and integration tests
tools/                label_shots.py, keyboard-driven contact and bounce labelling
trackers/             tracknet_ball_tracker.py, player_tracker.py
training/             court keypoint and shot classifier training
utils/                ball_state, court_validity, hit_bounce_classifier, kalman_smoother,
                      serve_detector, serve_landing, trajectory_3d, viewer_3d, and more
main.py               pipeline entry point
cli.py                tennis-vision command
```

## Notes on the CV concepts

Written while building, in `notes/`:

- `01_homography_basics.md`, court coordinate transformation
- `02_kalman_filter.md`, ball trajectory smoothing
- `03_temporal_smoothing.md`, keypoint jitter reduction
- `04_sort_tracker.md`, multi-object tracking
- `05_deepsort_reid.md`, re-identification
- `06_shot_detection.md`, shot classification methodology

## Credits

- **TrackNet** ([yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)), ball
  detection weights and the labelled dataset behind every event-detection number here
- **TennisCourtDetector**, the court keypoint dataset behind the fine-tuned model
- **Ultralytics YOLOv8**, player detection
- **MediaPipe**, pose estimation
- **THETIS**, shot type dataset
- PyTorch, OpenCV, NumPy, pandas

The court keypoint model published at
[Coddieharsh/tennis-court-keypoints](https://huggingface.co/Coddieharsh/tennis-court-keypoints)
is a derivative fine-tune, with a model card recording provenance and per-surface accuracy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule specific to this project: a new number
needs a script in `eval/` that produces it, and that script goes in the same commit.

## License

MIT. See [LICENSE](LICENSE).
