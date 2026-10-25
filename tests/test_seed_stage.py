"""``--stage-execs K``: pick a seed once, run up to K mutations of it.

The main loop picked and scored a seed on every exec (``weighted_pick_seed``
plus mutate ~60% of a LITE profile). AFL runs a stage of mutations per pick,
sized by the seed's perf score. Default K=1 keeps the old loop exactly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.services.seed_stage import PERF_NEUTRAL, SeedStage

K = 4
A, B = b"seed-a", b"seed-b"
LIVE = {A: {}, B: {}}


def _drain(stage: SeedStage, live) -> int:
    """Execs a begun stage serves after the picking exec."""
    n = 0
    while stage.resume(live) is not None:
        n += 1
    return n


def test_default_never_resumes():
    """Falsification: K=1 is the old loop, one pick per exec."""
    stage = SeedStage(1)
    stage.begin(A, PERF_NEUTRAL)
    assert stage.resume(LIVE) is None


def test_neutral_score_runs_full_stage():
    stage = SeedStage(K)
    stage.begin(A, PERF_NEUTRAL)
    assert stage.resume(LIVE) == A
    assert 1 + 1 + _drain(stage, LIVE) == K


@pytest.mark.parametrize(
    ("score", "execs"),
    [(PERF_NEUTRAL * 2, K), (PERF_NEUTRAL / 2, K // 2), (1.0, 1), (0.0, 1)],
)
def test_budget_scales_with_score(score, execs):
    """Stage length is K * score / neutral, clamped to [1, K]."""
    stage = SeedStage(K)
    stage.begin(A, score)
    assert 1 + _drain(stage, LIVE) == execs


def test_evicted_seed_ends_stage():
    """Adversarial: a seed pruned mid-stage is not fuzzed again."""
    stage = SeedStage(K)
    stage.begin(A, PERF_NEUTRAL)
    assert stage.resume({B: {}}) is None
    assert stage.resume(LIVE) is None


def test_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        SeedStage(0)


# ── Wiring: the real loop picks once per stage ─────────────────────────

_REPO = Path(__file__).resolve().parent.parent
_TARGET_SRC = _REPO / "targets" / "test_target.c"
_ITERS = 24
_HARNESS = """
import json, sys
from pathlib import Path
from fuzzer_tool.services.fuzzer import Fuzzer

target, root, k = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
f = Fuzzer(target=target, corpus_dir=str(root / "corpus"), crashes_dir=str(root / "crashes"),
           max_len=64, use_coverage=False, seed=7, stage_execs=k)
f._seed_scorer = None  # neutral score: stage length is exactly k
picks = [0]
real = f._pick_seed
def counting():
    picks[0] += 1
    return real()
f._pick_seed = counting
f.run(iterations=ITERS)
print(json.dumps(picks[0]))
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory) -> Path:
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("clang not available")
    exe = tmp_path_factory.mktemp("stage") / "test_target"
    built = subprocess.run([clang, "-O1", str(_TARGET_SRC), "-o", str(exe)], capture_output=True)
    if built.returncode != 0:
        pytest.skip("test_target did not build")
    return exe


def _picks(target: Path, root: Path, k: int) -> int:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "s0").write_bytes(b"hello world")
    script = _HARNESS.replace("ITERS", str(_ITERS))
    cmd = [sys.executable, "-c", script, str(target), str(root), str(k)]
    done = subprocess.run(cmd, cwd=root, capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]
    return json.loads(done.stdout.decode().strip().splitlines()[-1])


def test_loop_picks_once_per_stage(target, tmp_path):
    assert _picks(target, tmp_path / "k1", 1) == _ITERS
    assert _picks(target, tmp_path / "k4", K) == _ITERS // K


@pytest.mark.parametrize("bad", ["0", "-3", "many"])
def test_cli_rejects_bad_k(bad, tmp_path):
    """Adversarial: K < 1 or non-integer is an argparse error, not a traceback."""
    done = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "t", "--stage-execs", bad],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 2
    assert "--stage-execs" in done.stderr
