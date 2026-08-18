# Changelog

All notable changes to Tennis-Vision are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with one addition:
a **Measured and rejected** section per release. Approaches that were built, measured and
turned down are recorded with the number that turned them down, because a negative result
is expensive to produce and cheap to reuse.

Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-08-18

147 commits. Not a feature release: V1 already produced numbers, and this release is about
whether those numbers were true. Several were not, and the corrections are below.

Every figure names the script that produced it. Where something is unmeasured, it says so.

### Added

**Measurement infrastructure**

- `eval/ball_localization_accuracy.py` : ball position against hand-labelled ground truth,
  reporting detection rate and localization error as separate quantities. Sweeps the
  heatmap threshold and cluster size, and compares postprocessing modes.
- `eval/event_detection_on_real_detections.py` : contact and bounce detection running real
  TrackNet inference end to end, rather than on the dataset's labelled ball coordinates.
- `eval/event_recall_funnel.py` : attributes every missed contact to the pipeline stage
  that lost it, so the next piece of work is chosen by size rather than by guess. Also
  sweeps the candidate merge window under both linkage rules.
- `eval/serve_false_positive_check.py` : does the pipeline claim a serve on clips cut from
  mid-rally, which is most footage a user brings.
- `eval/pose_availability_at_contacts.py` : how often pose is usable at the moment a shot
  is struck, at the exact frame and within a small window.
- `eval/forehand_backhand_on_thetis.py` : the geometric rule against ground truth, with an
  oracle mode separating a wrong hand choice from a projection that cannot express the
  answer.
- `eval/extract_thetis_pose_features.py` and `eval/train_forehand_backhand.py` : named pose
  features and a classifier trained with subject-grouped splits, repeated cross-validation,
  balanced accuracy and feature ablations.
- `eval/validate_on_broadcast_images.py` : transfer test from indoor training data to real
  broadcast footage.
- `eval/swing_candidate_recall.py` and `eval/sam3d_occluded_arm_test.py`.

**Pipeline capability**

- 3-D free-flight reconstruction between detected contacts, and an interactive viewer
  written as a single self-contained HTML file with no external dependencies.
- Court validity gate based on line support: samples along every predicted court line and
  asks whether those pixels are brighter than the surface beside them.
- Physical serve detection: ball above the player's head **and** hitter at or behind a
  baseline, with a minimum separation between serves in seconds rather than frames.
- Physics-based volley, smash and lob classification from the event sequence.
- Trained hit versus bounce classifier on four named trajectory features.
- Optional SAM 3D Body pose backend, off by default, interface-compatible with the
  MediaPipe estimator.
- `scripts/download_thetis.py` and `scripts/download_sam3d_body.py`.
- `tools/LABELLING_GUIDE.md` : procedure for producing shot-type ground truth.

**Packaging**

- `pip install -e .` with a `tennis-vision` CLI, layered configuration, and regression tests
  asserting the built-in defaults match the shipped config exactly.

### Changed

- **Ball position is now the centroid of the largest connected heatmap response**, not the
  mean of every responding pixel. Median localization error 5.8px to 5.4px, 90th percentile
  20.2px to 18.0px. The tail improves more than the median, which is the signature of
  removing blended two-response frames rather than of general smoothing.
- **Candidate clustering is bounded, not transitive.** Recall 51.3% to 68.4% at unchanged
  precision.
- **Detection caches are keyed per video** and off by default.
- **Pose is requested at the contact frame and then its neighbours**, since the contact
  frame is the worst moment to ask: the player is fully extended, often side-on, occluded
  by their own racket arm and motion-blurred. Pose reaches 77% of contacts at the exact
  frame and 91% within four frames either side.
- **Shots labelled "Groundstroke" are now eligible for the pose upgrade.** That label means
  "a ground stroke, side unknown", which is exactly what pose resolves, and excluding it
  withheld 4 of 13 shots on the reference clip. Shots receiving a pose-based label went
  from 54% to 85%.
- **The 3-D viewer draws players**, fits the canvas instead of using a fifth of it, and
  separates arcs by weight so the one under examination is legible among seventeen.
- Evals now grade the pipeline's shipped output rather than an earlier stage.

### Fixed

- **Detection caches were keyed to nothing.** A single shared file with no record of which
  video wrote it, so analysing one clip and then another silently gave the second the
  first's ball positions. The file on disk held 40 frames from a smoke run and was being
  used to grade a 570-frame video.
- **Candidate clusters grew without bound.** Frames 0, 10, 20, 30 and 40 collapsed into one
  event at frame 20. The merge window had been set to match "the tolerance used throughout
  this sprint's eval scripts", which is a category error: an eval tolerance answers how
  close a detection must be to count as a match, a merge window answers how close two
  generator firings can be and still describe one physical event.
- **`pip install` produced a silently degraded pipeline.** Wheel users got the superseded
  court model, YOLO instead of TrackNet, no pose shots, missing classifier weights, caching
  enabled against documented behaviour, and two undeclared runtime dependencies. Everything
  ran and nothing errored.
- **Classifier weights were resolved against the working directory**, so they loaded only
  when the process happened to start from the repository root. Elsewhere the classifier
  returned None for every event and the pipeline fell back to a weaker heuristic silently.
- **`PlayerTracker` crashed on a cache miss** instead of detecting fresh.
- **Two serves were reported 18 frames apart**, which is 0.6s at 30fps and physically
  impossible.
- **A features/weights mismatch raised a bare `KeyError`** from deep inside classification.
  It now names the drift, and never defaults a missing feature to zero, which would return
  confident probabilities from an input the model never saw.
- **`save_video` was hardcoded to 24fps**, so every 30fps clip was written 25% slow and its
  clock drifted against the 3-D viewer's timeline.
- **The 3-D viewer's first scale computation threw on a null trig basis** at page init,
  leaving the canvas blank with no console error because the listener attaches later.

### Corrected numbers

Figures previously published that described something other than what a reader would
assume.

| Claim | Previously | Actually |
|---|---|---|
| Ball detection | "82.5% detection rate", read as accuracy | 88.6% of frames get a position; **42.5%** of visible-ball frames are within 5px |
| Event recall | "87.6%, production config" | 87.6% **given perfect ball positions**; **72.0%** running real detection |
| Shot-frame accuracy | "mean offset 4.9 frames, EXCELLENT" | 7/7 recall, mean offset 7.4 frames, precision measured separately |
| Court validity clips | 8/9 | unchanged, but an eval reported 5/9 until it was fixed to use per-frame keypoints |

### Measured and rejected

Each was implemented, measured and turned down. Full detail in the README.

| Approach | Result |
|---|---|
| Homography reprojection error as a court-validity signal | 1.40-1.88px on correct fits, 2.13px on a wrong one, 14/14 inliers every time |
| Heatmap threshold and cluster-size sweeps | every setting within noise of the shipped one |
| A larger or newer YOLO for player detection | detection saturated: 11-14 people found per frame where 2 are needed |
| RTS forward-backward smoothing | complete coverage and 1.5 points of recall for 4% worse median error |
| A chi-square outlier gate | median error 6.3px to 29.4px, and to 207.9px at tight tuning |
| A body-based swing generator | 58.5% of ordinary frames score at or above the median contact |
| The energy-ratio bounce signal | 65.7% on 1,034 labelled events |
| A trained forehand/backhand classifier | 76.3% indoors, **53.6% on broadcast** |
| Replacing MediaPipe with SAM 3D Body | **19 points worse** on identical clips |

The last one is worth reading in full. SAM 3D Body recovers the occluded racket arm that
MediaPipe drops on 44-58% of backhands, taking usable clips from 171 to 200 and class
balance from 55/45 to a perfect 100/100, and it made the classifier substantially worse.
The damage is confined to position and leaves motion untouched: the wrist-side feature
falls from 78.6% to chance. On occluded frames MediaPipe declines and SAM 3D infers the arm
from a body prior, which is anatomically plausible and still a guess about where the racket
is. MediaPipe's refusal was a quality filter, not only a loss.

### Known limitations

- Forehand versus backhand is unreliable: 54% on balanced ground truth. Three replacements
  were measured and none is better.
- Rally and groundstroke speeds are unvalidated. Serve speed is radar-validated at a 0.96
  mean ratio; rally speed has no ground truth.
- Roughly a quarter of contacts in a rally are missed, at 95.9% precision, so the shot
  count is an under-count rather than noise.
- Volley and smash have never been checked against labels.
- Ball height is modelled from contact anchors and gravity, not measured, and cannot be
  otherwise from broadcast camera geometry.
- Ground-level cameras fail; the validity gate flags them rather than reporting wrong
  numbers.
- Doubles and amateur footage are untested.

### Tests

83 to 213.

---

## [1.0.0] - 2025-08-25

First working version, preserved at the `v1.0.0` tag and the `v1-stable` branch.

### Added

- YOLOv8 player and ball detection with ByteTrack
- ResNet-50 court keypoint regression, 14 points
- Mini-court coordinate mapping and bird's-eye visualisation
- Rule-based shot classification
- Player movement and shot speed statistics
- Annotated video output
- Training scripts, contributing guidelines, MIT licence

### Superseded by 2.0.0

V1's reported figures were not wrong so much as differently defined from how they read.
Its ball detection rate was taken as an accuracy, and its event recall described a pipeline
with perfect ball positions rather than the real one. 2.0.0 publishes both numbers in each
case.

---

[2.0.0]: https://github.com/HarshTomar1234/Tennis-Vision/releases/tag/v2.0.0
[1.0.0]: https://github.com/HarshTomar1234/Tennis-Vision/releases/tag/v1.0.0
