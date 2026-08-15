"""
eval/explore_hit_bounce_features.py
────────────────────────────────────
Feature exploration for hit-vs-bounce classification, using the real 1,030-event
TrackNet ground truth (see docs/journal/0010, 0011).

Physical hypotheses to test before committing to any model (ladder-thinking - check
what actually separates the classes before reaching for anything more than the simplest
feature that works):

  1. Height (already measured: hit mean 275px, bounce mean 315px - real but overlapping).
  2. Vertical velocity magnitude change at the event (both cause a y-reversal by
     definition, so this may not discriminate much - testing anyway).
  3. Horizontal (x) velocity change - a bounce is a court reflection, which mostly
     preserves x-velocity (no strong sideways force from the ground). A hit is a player
     redirecting the ball - can reverse or sharply change x-direction (cross-court shots,
     returns). If this holds, it should be a much stronger signal than height alone.

Split by CLIP, not by event - events from the same clip share camera, lighting, and
players, so an event-level split would leak information and overstate accuracy.
"""
import csv
import io
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"
WINDOW = 4   # frames before/after the event to measure velocity


def extract_features(zf: zipfile.ZipFile, label_path: str) -> list[dict]:
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))

    # Build a clean (x, y) or None per frame.
    positions: list[tuple[float, float] | None] = []
    for r in rows:
        if r["visibility"] != "0" and r["x-coordinate"]:
            positions.append((float(r["x-coordinate"]), float(r["y-coordinate"])))
        else:
            positions.append(None)

    events = []
    n = len(rows)
    for i, r in enumerate(rows):
        if r["status"] not in ("1", "2"):
            continue
        if positions[i] is None:
            continue

        before = [p for p in positions[max(0, i - WINDOW):i] if p is not None]
        after  = [p for p in positions[i + 1:i + 1 + WINDOW] if p is not None]
        if len(before) < 2 or len(after) < 2:
            continue   # not enough context to compute velocity

        # Average velocity just before / just after (px/frame).
        vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
        vy_before = (before[-1][1] - before[0][1]) / (len(before) - 1)
        vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
        vy_after  = (after[-1][1] - after[0][1]) / (len(after) - 1)

        events.append({
            "clip": label_path,
            "frame": i,
            "label": "hit" if r["status"] == "1" else "bounce",
            "height_y": positions[i][1],
            "vy_change_mag": abs(vy_after - vy_before),
            "vx_before": vx_before,
            "vx_after": vx_after,
            "vx_change_mag": abs(vx_after - vx_before),
            "vx_sign_flip": int((vx_before > 0) != (vx_after > 0)) if abs(vx_before) > 0.5 and abs(vx_after) > 0.5 else None,
        })
    return events


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    label_paths = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))

    all_events = []
    for lp in label_paths:
        all_events.extend(extract_features(zf, lp))

    hits    = [e for e in all_events if e["label"] == "hit"]
    bounces = [e for e in all_events if e["label"] == "bounce"]
    print(f"Usable events (enough context on both sides): {len(hits)} hits, {len(bounces)} bounces")
    print()

    def stats(name, key, subset):
        vals = [e[key] for e in subset if e[key] is not None]
        if not vals:
            return
        mean = sum(vals) / len(vals)
        var  = sum((v - mean) ** 2 for v in vals) / len(vals)
        print(f"  {name:<16} n={len(vals):<5} mean={mean:8.2f}  std={var**0.5:7.2f}")

    for key, label in [
        ("height_y", "height (y-px)"),
        ("vy_change_mag", "|vy change|"),
        ("vx_change_mag", "|vx change|"),
    ]:
        print(f"--- {label} ---")
        stats("hit", key, hits)
        stats("bounce", key, bounces)
        print()

    flip_hit = [e["vx_sign_flip"] for e in hits if e["vx_sign_flip"] is not None]
    flip_bounce = [e["vx_sign_flip"] for e in bounces if e["vx_sign_flip"] is not None]
    print(f"--- x-direction sign flip rate (only where |vx| > 0.5px/frame both sides) ---")
    print(f"  hit:    {sum(flip_hit)}/{len(flip_hit)} = {100*sum(flip_hit)/len(flip_hit):.1f}% flipped direction")
    print(f"  bounce: {sum(flip_bounce)}/{len(flip_bounce)} = {100*sum(flip_bounce)/len(flip_bounce):.1f}% flipped direction")

    # Save for the next step (classifier training) - clip-level split done there.
    import json
    Path("datasets/external/hit_bounce_features.json").write_text(json.dumps(all_events))
    print(f"\nSaved {len(all_events)} events -> datasets/external/hit_bounce_features.json")


if __name__ == "__main__":
    main()
