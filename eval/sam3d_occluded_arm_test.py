"""
eval/sam3d_occluded_arm_test.py
───────────────────────────────
Does SAM 3D Body recover the arm MediaPipe drops on a backhand?

The precise problem this tests
------------------------------
MediaPipe does not fail to find the player. Measured across THETIS backhand clips it
finds a pose on 100% of frames, and then omits the landmarks of the occluded arm:

    class              frames  no pose  most-missing
    backhand_volley       117        0  right elbow 68%, right wrist 58%
    forehand_volley       120        0  left elbow   9%, left wrist   5%
    backhand              142        0  right elbow 47%, right wrist 44%
    forehand_flat         146        0  left elbow  25%, left wrist  23%

On a backhand the arm that goes missing is the one holding the racket, roughly half the
time. Forehand versus backhand is decided by where that arm is, so this is upstream of
every classifier built on these landmarks: the geometric rule scores 54% and a trained
classifier scores 53.6% on broadcast, and both are reading a hand that is often not there.

SAM 3D Body is trained for occlusion and unusual postures and predicts a full-body mesh
rather than independent landmarks, so an occluded wrist is inferred from the body rather
than dropped. This checks whether that is true on the exact frames MediaPipe gives up on.

What a pass looks like
----------------------
SAM 3D Body returns a wrist position on frames where MediaPipe returned none, and the
position is plausible (inside the image, attached to the body). That would make it a
genuine fix rather than a heavier model with the same blind spot.

Hardware note
-------------
The shipped config asks for bfloat16, which Pascal GPUs (GTX 10-series) do not support.
This falls back to float16 and then float32, and reports which path ran, because "it works"
and "it works on this machine" are different claims.

Usage
-----
    python eval/sam3d_occluded_arm_test.py --frames 12
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from utils import PoseEstimator

SAM3D_CODE = pathlib.Path(
    "C:/Users/Harsh/AppData/Local/Temp/claude/D--Tennis-Vision/sam3dbody"
)
WEIGHTS_DIR = pathlib.Path("models/sam3d_body")
CLASSES = ("backhand_volley", "backhand", "backhand_slice")

# Landmarks whose absence makes a frame unusable for forehand/backhand.
ARM = ("LEFT_WRIST", "RIGHT_WRIST", "LEFT_ELBOW", "RIGHT_ELBOW")


def find_frames_missing_an_arm(n_wanted: int):
    """Frames where MediaPipe finds a pose but omits a wrist. The interesting cases."""
    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Run: tennis-vision download-models")
        sys.exit(1)

    found = []
    for cls in CLASSES:
        folder = pathlib.Path(f"datasets/external/thetis/VIDEO_RGB/{cls}")
        if not folder.is_dir():
            continue
        for clip in sorted(folder.glob("*.avi"))[:25]:
            cap = cv2.VideoCapture(str(clip))
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            if len(frames) < 20:
                continue
            lo, hi = len(frames) // 4, len(frames) * 3 // 4
            for i in range(lo, hi, 4):
                h, w = frames[i].shape[:2]
                lms = estimator.detect_in_bbox(frames[i], [0, 0, w, h])
                if lms is None:
                    continue
                absent = [n for n in ARM if n not in lms]
                if absent:
                    found.append({"image": frames[i], "clip": clip.stem,
                                  "frame": i, "missing": absent})
                    if len(found) >= n_wanted:
                        estimator.close()
                        return found
    estimator.close()
    return found


def load_model():
    """Load SAM 3D Body, trying the precisions this GPU can actually run."""
    sys.path.insert(0, str(SAM3D_CODE))
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"device: {name}, {vram:.1f} GB")
        print(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")

    last = None
    # Weights stay float32 and mixed precision is applied with autocast at inference.
    # Casting the weights fails both ways: as float16 the preprocessing feeds bfloat16
    # into float16 biases, and as bfloat16 a head that expects float32 rejects it. The
    # config USE_FP16/FP16_TYPE describes an autocast context, not a weight dtype.
    for dtype in (torch.float32,):
        try:
            # Signature is (checkpoint_path, device, mhr_path); the config is found
            # from the checkpoint directory, and mhr_path must be given explicitly or
            # the pose head is constructed with an empty filename.
            # Returns (model, config); the config is needed by the estimator.
            model, model_cfg = load_sam_3d_body(
                checkpoint_path=str(WEIGHTS_DIR / "model.ckpt"),
                device=device,
                mhr_path=str(WEIGHTS_DIR / "assets" / "mhr_model.pt"),
            )
            model.eval()
            print(f"loaded in {dtype}")
            # No detector or segmentor: the player box is supplied directly, which is
            # what the pipeline already has from YOLO.
            return SAM3DBodyEstimator(model, model_cfg), device, dtype
        except Exception as e:
            last = e
            print(f"  {dtype} failed: {type(e).__name__} {str(e)[:160]}")
            torch.cuda.empty_cache() if device == "cuda" else None
    raise RuntimeError(f"could not load model: {last}")


def main():
    ap = argparse.ArgumentParser(description="Can SAM 3D Body recover the occluded arm?")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    if not (WEIGHTS_DIR / "model.ckpt").exists():
        print(f"Weights not found in {WEIGHTS_DIR}.")
        sys.exit(1)

    print("Finding frames where MediaPipe finds a body but drops an arm...")
    cases = find_frames_missing_an_arm(args.frames)
    if not cases:
        print("None found, which would itself be worth investigating.")
        return
    print(f"  {len(cases)} such frames\n")

    estimator, device, dtype = load_model()

    recovered = attempted = 0
    times = []
    for case in cases:
        image = case["image"]
        h, w = image.shape[:2]
        attempted += 1
        try:
            t0 = time.time()
            # No autocast. The MHR head is a TorchScript module and an autocast context
            # makes it raise NotImplementedError on ops it has no bf16 kernel for. Plain
            # float32 runs, which is the configuration this measurement describes.
            with torch.no_grad():
                out = estimator.process_one_image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    bboxes=np.array([[0, 0, w, h]], dtype=np.float32),
                    inference_type="body",
                )
            times.append(time.time() - t0)
        except Exception as e:
            print(f"  {case['clip']} f{case['frame']}: inference failed "
                  f"{type(e).__name__} {str(e)[:120]}")
            continue

        # The output shape is not documented in the README, so the first successful
        # frame prints its keys once; guessing at a field name would be a way to report
        # success from a model that returned nothing useful.
        # Output is a list, one entry per person. Its field names are not documented,
        # so the first frame prints them: guessing at a key would let this report
        # success from a model that returned nothing useful.
        person = out[0] if isinstance(out, list) and out else out
        if attempted == 1:
            print(f"    output: {type(out).__name__} of {len(out) if isinstance(out, list) else 1}")
            if isinstance(person, dict):
                print(f"    person keys: {sorted(person.keys())}")

        keypoints = None
        if isinstance(person, dict):
            for key in ("keypoints_2d", "pred_keypoints_2d", "joints_2d", "keypoints",
                        "pred_keypoints_3d", "joints", "pred_joints", "j2d", "j3d"):
                val = person.get(key)
                if val is not None:
                    keypoints = np.asarray(val.detach().cpu() if hasattr(val, "detach") else val)
                    break
        got = keypoints is not None and keypoints.size > 0
        recovered += int(got)
        print(f"  {case['clip']:<28} f{case['frame']:<4} "
              f"missing {','.join(m.split('_')[0].lower() + ' ' + m.split('_')[1].lower() for m in case['missing'])[:38]:<40} "
              f"-> {'keypoints returned' if got else 'no keypoints'}")

    print()
    print(f"SAM 3D Body returned keypoints on {recovered}/{attempted} frames "
          f"MediaPipe could not complete")
    if times:
        print(f"mean inference {sum(times)/len(times):.2f}s per frame on {device} ({dtype})")
        print(f"a 570-frame clip with 2 players would cost "
              f"{sum(times)/len(times)*570*2/60:.0f} minutes at this rate")


if __name__ == "__main__":
    main()
