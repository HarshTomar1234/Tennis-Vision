"""
eval/explore_slice_features.py
─────────────────────────────────
Feature exploration for slice-vs-flat/topspin classification, mirroring the
methodology that worked for the hit/bounce classifier in Phase 3
(eval/explore_hit_bounce_features.py): compute several physically-motivated candidate
features first, measure which ones actually separate the classes on real data, THEN
pick a model -- not the other way around.

Reads datasets/external/thetis_slice_features.json (from
eval/extract_thetis_slice_features.py).
"""
import json
import statistics
from pathlib import Path


def shoulder_width(frame_lm):
    ls, rs = frame_lm.get("LEFT_SHOULDER"), frame_lm.get("RIGHT_SHOULDER")
    if not ls or not rs:
        return None
    return ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5


def hitting_wrist_trace(sequence):
    """
    Pick whichever wrist moves more across the clip (proxy for "the swinging arm" --
    THETIS clips have no ball/contact marker to identify it directly), and return its
    (offset_x, offset_y) from the shoulder midpoint per frame, normalized by shoulder
    width so subject size/distance-from-camera doesn't leak into the features.
    """
    valid = [f for f in sequence if f]
    widths = [shoulder_width(f) for f in valid]
    widths = [w for w in widths if w]
    if len(widths) < 5:
        return None, None
    scale = statistics.median(widths)
    if scale < 1e-3:
        return None, None

    def wrist_offsets(name):
        pts = []
        for f in sequence:
            if not f or name not in f or "LEFT_SHOULDER" not in f or "RIGHT_SHOULDER" not in f:
                pts.append(None)
                continue
            ls, rs = f["LEFT_SHOULDER"], f["RIGHT_SHOULDER"]
            cx, cy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
            w = f[name]
            pts.append(((w[0] - cx) / scale, (w[1] - cy) / scale))
        return pts

    left = wrist_offsets("LEFT_WRIST")
    right = wrist_offsets("RIGHT_WRIST")

    def y_range(pts):
        ys = [p[1] for p in pts if p]
        return (max(ys) - min(ys)) if len(ys) >= 5 else 0.0

    return (left, right) if y_range(left) >= y_range(right) else (right, left)


def linreg_slope(ys):
    xs = list(range(len(ys)))
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-9 else 0.0


def compute_features(record):
    trace, _ = hitting_wrist_trace(record["sequence"])
    if trace is None:
        return None
    pts = [(i, p) for i, p in enumerate(trace) if p]
    if len(pts) < 10:
        return None

    ys = [p[1] for _, p in pts]
    xs = [p[0] for _, p in pts]
    peak_idx = max(range(len(pts)), key=lambda i: ys[i])
    peak_frac = pts[peak_idx][0] / max(1, len(trace) - 1)

    n = len(pts)
    last_third = ys[int(n * 2 / 3):]
    first_third = ys[:int(n / 3)]

    return {
        "y_range": max(ys) - min(ys),
        "x_range": max(xs) - min(xs),
        "net_decline_after_peak": ys[peak_idx] - ys[-1],   # positive = wrist rises back up after peak
        "overall_slope": linreg_slope(ys),
        "last_third_slope": linreg_slope(last_third) if len(last_third) >= 2 else 0.0,
        "first_third_slope": linreg_slope(first_third) if len(first_third) >= 2 else 0.0,
        "peak_fraction": peak_frac,
        "coverage": len(pts) / len(trace),
    }


def main():
    path = Path("datasets/external/thetis_slice_features.json")
    with open(path) as f:
        records = json.load(f)

    rows = []
    skipped = 0
    for r in records:
        feats = compute_features(r)
        if feats is None:
            skipped += 1
            continue
        feats["is_slice"] = r["is_slice"]
        feats["subject"] = r["subject"]
        feats["clip"] = r["clip"]
        rows.append(feats)

    print(f"{len(rows)} clips with usable features ({skipped} skipped -- too little pose coverage)")
    slice_rows = [r for r in rows if r["is_slice"]]
    flat_rows  = [r for r in rows if not r["is_slice"]]
    print(f"slice: {len(slice_rows)}, not-slice: {len(flat_rows)}")
    print()

    feature_names = [k for k in rows[0] if k not in ("is_slice", "subject", "clip")]
    print(f"{'feature':<22} {'slice mean':>12} {'slice std':>10} {'flat mean':>12} {'flat std':>10} {'separation':>11}")
    for name in feature_names:
        s = [r[name] for r in slice_rows]
        fl = [r[name] for r in flat_rows]
        s_mean, s_std = statistics.mean(s), statistics.stdev(s) if len(s) > 1 else 0
        f_mean, f_std = statistics.mean(fl), statistics.stdev(fl) if len(fl) > 1 else 0
        pooled_std = ((s_std ** 2 + f_std ** 2) / 2) ** 0.5
        cohens_d = abs(s_mean - f_mean) / pooled_std if pooled_std > 1e-9 else 0
        print(f"{name:<22} {s_mean:>12.3f} {s_std:>10.3f} {f_mean:>12.3f} {f_std:>10.3f} {cohens_d:>11.3f}")

    out_path = Path("datasets/external/thetis_slice_summary_features.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} feature rows -> {out_path}")


if __name__ == "__main__":
    main()
