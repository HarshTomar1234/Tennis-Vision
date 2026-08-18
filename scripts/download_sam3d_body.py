"""
scripts/download_sam3d_body.py
──────────────────────────────
Fetches the optional SAM 3D Body pose weights into models/sam3d_body.

What it is for
--------------
MediaPipe finds the player on every frame and then omits the occluded arm. Measured
across THETIS backhand clips, the missing landmark is the racket arm on 44-58% of frames:

    class              frames  no pose  most-missing
    backhand_volley       117        0  right elbow 68%, right wrist 58%
    forehand_volley       120        0  left elbow   9%, left wrist   5%
    backhand              142        0  right elbow 47%, right wrist 44%
    forehand_flat         146        0  left elbow  25%, left wrist  23%

Forehand versus backhand is decided by where that arm is, so the label is often read from
a hand that is not there. SAM 3D Body predicts a whole-body mesh and infers occluded
joints instead of dropping them: on the exact frames MediaPipe could not complete, it
returned keypoints on 6 of 6 (eval/sam3d_occluded_arm_test.py).

It is used on contact frames only, roughly 15 per clip. At about 1.6s per frame a
per-frame pass would take half an hour on a mid-range GPU; 30 inferences takes under a
minute.

Why this is not part of download-models
----------------------------------------
Two reasons, and both are about honesty rather than convenience.

The weights are GATED: you must request access on Hugging Face and accept Meta's terms
yourself, and then authenticate. Nobody can do that on your behalf.

The weights are under Meta's SAM License, not MIT. That licence grants free use,
modification and derivative works, and requires that anyone redistributing the materials
passes the same terms along with them. This project is MIT, so bundling the weights would
have our licence make a promise about Meta's weights that we have no standing to make.
Fetching them yourself, under terms you accepted directly, avoids that entirely. It is the
same arrangement used for the TrackNet weights.

Before running
--------------
1. Request access at https://huggingface.co/facebook/sam-3d-body-vith
2. Create a read token at https://huggingface.co/settings/tokens
3. Put it in the environment or in .env as HF_TOKEN=hf_...

Then enable it in configs/config.yaml:

    pipeline:
      use_sam3d_pose: true

The inference code is a separate repository and is not vendored here:

    git clone https://github.com/facebookresearch/sam-3d-body.git
    export SAM3D_BODY_CODE=/path/to/sam-3d-body

Usage
-----
    python scripts/download_sam3d_body.py
    python scripts/download_sam3d_body.py --model dinov3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOS = {
    "vith": "facebook/sam-3d-body-vith",
    "dinov3": "facebook/sam-3d-body-dinov3",
}
TARGET = Path("models/sam3d_body")
FILES = ["model_config.yaml", "assets/mhr_model.pt", "model.ckpt"]


def find_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("HF_TOKEN", "HUGGINGFACE_TOKEN")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Download optional SAM 3D Body weights")
    ap.add_argument("--model", choices=sorted(REPOS), default="vith",
                    help="vith is 2.4 GB, dinov3 is 2.8 GB and slightly stronger")
    args = ap.parse_args()

    token = find_token()
    if not token:
        print("No Hugging Face token found in the environment or .env.")
        print("These weights are gated, so a token is required even after access is")
        print("granted. See the header of this file for the three steps.")
        return 1

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub")
        return 1

    repo = REPOS[args.model]
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching {repo} into {TARGET}")
    print("About 2.4 GB. The licence is Meta's SAM License, not MIT.\n")

    for name in FILES:
        target = TARGET / name
        if target.exists():
            print(f"  [skip]     {name}")
            continue
        print(f"  [download] {name}")
        try:
            hf_hub_download(repo_id=repo, filename=name, token=token,
                            local_dir=str(TARGET))
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if token:
                message = message.replace(token, "<token>")
            print(f"  [FAILED]   {name}: {type(exc).__name__} {message[:200]}")
            if "GatedRepo" in message or "401" in message or "403" in message:
                print()
                print("  Access has not been granted yet, or the token lacks permission.")
                print(f"  Request access at https://huggingface.co/{repo}")
                print("  Approval is manual and can take time.")
            return 1

    print("\nDone. Enable it in configs/config.yaml:")
    print("    pipeline:")
    print("      use_sam3d_pose: true")
    print("\nThe inference code is a separate repository:")
    print("    git clone https://github.com/facebookresearch/sam-3d-body.git")
    print("    export SAM3D_BODY_CODE=/path/to/sam-3d-body")
    return 0


if __name__ == "__main__":
    sys.exit(main())
