"""tools/lib/bench_periodicity_null.py: detect_periodicity false-positive table."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_periodicity_null as B  # noqa: E402

from fuzzer_tool.core.synthetic_series import ar1  # noqa: E402

ENTRY = Path(__file__).resolve().parent.parent / "tools" / "benchmark.py"
ALPHA = 0.05
SLACK = 0.05  # binomial sd at p=0.05, 200 reps is ~0.015; 3x+ headroom
SEED = 20261030


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRY), "periodicity_null", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _white(rng, n):
    return rng.normal_array(n)


def test_output_shape():
    """Header plus one row per (null, n); each row ends in two rates."""
    out = _run("--reps", "3", "--n", "64", "96", "--seed", "1")
    assert out.returncode == 0, out.stderr

    lines = out.stdout.strip().splitlines()
    rows = [ln for ln in lines if ln.split()[-1].replace(".", "").isdigit()]
    assert len(rows) >= 2 * len(B.NULLS)
    for row in rows:
        raw, pw = (float(x) for x in row.split()[-2:])
        assert 0.0 <= raw <= 1.0 and 0.0 <= pw <= 1.0


def test_white_noise_stays_near_alpha():
    raw, pw = B.null_rates(_white, 256, reps=200, seed=SEED, alpha=ALPHA)
    assert raw < ALPHA + SLACK, raw
    assert pw < ALPHA + SLACK, pw


def test_falsify_benchmark_sees_raw_inflation():
    """The measurement can fail: raw AR(1) phi=0.9 must read far above alpha."""
    raw, pw = B.null_rates(lambda r, n: ar1(r, n, 0.9), 256, reps=60, seed=SEED, alpha=ALPHA)
    assert raw > 0.5, raw
    assert pw < raw


def test_control_same_seed_is_reproducible():
    """Rule 46 control: a run against itself must agree exactly."""
    a = B.null_rates(_white, 128, reps=30, seed=SEED, alpha=ALPHA)
    b = B.null_rates(_white, 128, reps=30, seed=SEED, alpha=ALPHA)
    assert a == b


def test_adversarial_bad_args_rejected():
    """n below the detector's min_samples would read 0.000 silently."""
    for args in (("--reps", "0"), ("--n", str(B.MIN_SAMPLES - 1))):
        out = _run(*args)
        assert out.returncode == 2, args
