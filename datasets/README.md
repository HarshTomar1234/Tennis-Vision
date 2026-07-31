# Datasets

```
datasets/
  external/     Third-party pulled datasets (gitignored — large, not project source)
  labels/       Our own hand-labeled CSVs (tracked — small, valuable, see tools/label_shots.py)
```

## external/tracknet_original/

Original TrackNet tennis dataset (Huang et al. 2019, arXiv:1907.03698). Pulled
2026-07-31 — see `docs/specs/phase_3_labeled_dataset.md` for why, and
`docs/journal/0009`/`0010` for the full research and re-test trail.

- **Source:** https://github.com/yastrebksv/TrackNet (unofficial reimplementation repo,
  links to the original Google Drive dataset release)
- **Contents:** 10 broadcast games, **95 clips**, 19,835 labeled frames, 1280×720 @ 30fps.
  Each clip has a `Label.csv`: `file name, visibility, x-coordinate, y-coordinate, status`.
- **Bonus:** `ctb_regr_bounce.cbm` — a pretrained CatBoost bounce-regression model that
  came in the same drive folder.
- **Same lineage as our model:** `model_best.pt` here has the identical Google Drive file
  ID already referenced in `trackers/tracknet_ball_tracker.py`'s docstring — this is the
  training data behind the TrackNet weights we already run, not a new foreign source.
- **License:** not explicitly stated by the original authors; released for research
  reproduction. Treat as research-use-only.

### Label encoding — confirmed against the paper text (arXiv:1907.03698, Section III),
### cross-checked against `yastrebksv/TrackNet/bounce_train.py:34`. Not inferred.

**`status`** ("Trajectory Pattern"):
| value | meaning |
|---|---|
| `0` | flying — normal in-flight motion, no event |
| `1` | **hit** — player contact |
| `2` | **bounce** — ball touches the court |
| *(blank)* | undocumented — do not assume a meaning; observed near some clip boundaries |

**`visibility`** ("Visibility Class"):
| value | meaning |
|---|---|
| `0` | ball not in frame |
| `1` | clearly visible |
| `2` | in frame but not easily identifiable (blends with background etc.) |
| `3` | occluded by another object (e.g. a player) |

### Layout after extraction
```
external/tracknet_original/
  Dataset.zip              (as downloaded)
  ctb_regr_bounce.cbm
  model_best.pt
  Dataset/
    game1/Clip1/  0000.jpg ... Label.csv
    game1/Clip2/  ...
    ...
    game10/Clip9/
```
95 clips across 10 games (13, 8, 9, 7, 15, 4, 9, 9, 9, 12).

## labels/

Our own labels, produced with `tools/label_shots.py`. See
`docs/specs/phase_3_labeled_dataset.md` for the schema.
