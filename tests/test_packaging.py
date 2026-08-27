"""
tests/test_packaging.py
───────────────────────
Guards the two ways this project can ship a silently degraded pipeline.

Neither failure mode raises an error at runtime. Both produce a pipeline that runs to
completion and reports numbers worse than the README's measured ones, with nothing in
the output admitting it. That makes them worse than a crash, so they get tests.

1. `main._DEFAULTS` drifting from `configs/config.yaml`. The defaults are not a minimal
   fallback, they are what actually runs for anyone who installed the wheel instead of
   cloning, since the wheel ships no YAML. When they drifted, wheel users got the
   superseded court model (median 4.03px against 2.90px, 4 of 9 clips passing against 8),
   the YOLO ball detector instead of TrackNet, and no pose-based shot classification.

2. The hit/bounce weights path resolving against the working directory. As a bare
   relative path it only worked when the process was started from the repo root; from
   anywhere else the weights were "not found", every event classified as None, and the
   pipeline fell back to the player-proximity heuristic without saying so.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "configs" / "config.yaml"


def test_defaults_match_config_yaml():
    """
    Every key configs/config.yaml sets must have the same value in main._DEFAULTS.

    config.yaml is the documented, measured configuration. A wheel user never sees it, so
    any key where the two disagree is a setting where cloning and pip-installing produce
    different results.
    """
    from main import _DEFAULTS

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    mismatches = []
    for section, values in config.items():
        for key, expected in (values or {}).items():
            actual = _DEFAULTS.get(section, {}).get(key, "<<MISSING>>")
            if actual != expected:
                mismatches.append(
                    f"{section}.{key}: config.yaml={expected!r} _DEFAULTS={actual!r}"
                )

    assert not mismatches, (
        "main._DEFAULTS has drifted from configs/config.yaml. Wheel users run the "
        "defaults, so these settings differ between a clone and a pip install:\n  "
        + "\n  ".join(mismatches)
    )


def test_classifier_weights_resolve_from_any_working_directory():
    """
    The weights must load with the process started somewhere other than the repo root.

    Run in a subprocess because the path is resolved at import time, so changing the
    working directory inside this process would not exercise it.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from utils.hit_bounce_classifier import _load_weights\n"
        "w = _load_weights()\n"
        "assert w is not None, 'weights did not load'\n"
        "print(','.join(w['feature_names']))\n" % REPO
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.expanduser("~"),   # deliberately not the repo root
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"weights failed to load from a foreign working directory:\n{result.stderr}"
    )
    assert "height_y" in result.stdout


def test_shipped_weights_match_the_trained_feature_set():
    """
    The committed weights file must carry the feature names the code computes.

    Catches a retrain that changes the feature set without the inference path following,
    which surfaces as confident predictions from features the model never saw.
    """
    from utils.hit_bounce_classifier import DEFAULT_WEIGHTS_PATH, compute_event_features

    weights = json.loads(Path(DEFAULT_WEIGHTS_PATH).read_text(encoding="utf-8"))

    # A trajectory with enough clean context either side for every feature to compute.
    positions = [(float(x), float(100 + abs(x - 10) * 5)) for x in range(20)]
    features = compute_event_features(positions, event_frame=10)
    assert features is not None

    missing = [n for n in weights["feature_names"] if n not in features]
    assert not missing, f"weights expect features the code does not compute: {missing}"


@pytest.mark.parametrize("package", ["supervision", "lapx"])
def test_runtime_dependencies_are_declared(package):
    """
    Dependencies needed at runtime must be in pyproject.toml, not only requirements.txt.

    ultralytics' tracking mode (which PlayerTracker uses) needs both. They were in
    requirements.txt but absent from pyproject, so `pip install tennis-vision` produced
    an install that crashed on the first tracked frame while the requirements.txt path
    worked fine.
    """
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert package in pyproject, f"{package} is required at runtime but not declared"


def test_version_is_consistent_across_the_project():
    """
    pyproject, the CLI and the CHANGELOG must agree.

    They did not: both pyproject and cli.py said 0.1.0 while the CHANGELOG and the git
    tags were on 2.x, so `tennis-vision version` reported a number that matched no
    release. A user cannot report a bug against a version string that does not exist.
    """
    import re

    # Regex rather than tomllib: tomllib is stdlib only from Python 3.11 and this project
    # declares requires-python >= 3.10, so importing it here would break the floor the
    # CI matrix exists to verify. It did, on the first run of this test.
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    packaged = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)

    cli_source = (REPO / "cli.py").read_text(encoding="utf-8")
    cli_version = re.search(r'__version__ = "([^"]+)"', cli_source).group(1)

    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    latest_release = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.M).group(1)

    assert packaged == cli_version, (
        f"pyproject says {packaged}, cli.py says {cli_version}"
    )
    assert packaged == latest_release, (
        f"pyproject says {packaged}, newest CHANGELOG entry is {latest_release}"
    )
