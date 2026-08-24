"""Regression tests for the seed-baseline calibration pass.

`Fuzzer._calibrate_seed_baselines()` executes every corpus seed verbatim
before the fuzz loop. Without it, `fuzz_one()` runs ONLY mutated input
(`_dedup_mutate`), so a structured-format campaign starts blind: mutated
seeds fail validation, per-exec coverage collapses to wrapper-only
edges (~6 on png_read.so), and Good-Turing reads the starved universe
as fully saturated. See docs/TODO.md "Seed baseline calibration".
"""

import os
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

_TARGET = Path("targets/png_read.so")
_CRASH_TARGET = Path("targets/test_target.so")

needs_png_target = pytest.mark.skipif(not _TARGET.exists(), reason="targets/png_read.so not built")
needs_crash_target = pytest.mark.skipif(
    not _CRASH_TARGET.exists(), reason="targets/test_target.so not built"
)


def _valid_png(width: int = 8) -> bytes:
    """Minimal well-formed grayscale PNG (decodes through libpng).

    Width varies the inflate/filter workload so distinct seeds produce
    genuinely distinct (overlapping but not identical) edge sets —
    trailing bytes after IEND would just make libpng reject the file.
    """
    import struct
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, 8, 8, 0, 0, 0, 0))
    row = bytes(((x * 7) & 0xFF) for x in range(width))
    raw = b"".join(b"\x00" + row for _ in range(8))
    idat = chunk(b"IDAT", zlib.compress(raw))
    return sig + ihdr + idat + chunk(b"IEND", b"")


def _make_fuzzer(corpus_dir: Path) -> Fuzzer:
    return Fuzzer(
        target=str(_TARGET),
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        max_len=4096,
        timeout=5,
        use_coverage=True,
        cmplog=False,
    )


@pytest.fixture(autouse=True)
def _cwd_repo_root():
    """Targets are resolved relative to the repo root."""
    if not Path("src/fuzzer_tool").exists():
        pytest.skip("must run from the repo root")


def _run_isolated(script: str) -> str:
    """Run a calibration scenario in a fresh interpreter.

    dlopen() caches the target .so per process, so a SECOND Fuzzer in the
    same interpreter reuses the first one's shim mapping — whose
    __afl_area still points at the previous test's (cleaned-up) segment.
    Real campaigns never reload a target mid-process, so the honest way
    to assert end-to-end numbers here is one scenario per process.
    """
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=os.getcwd(),
    )
    assert r.returncode == 0, f"isolated scenario failed:\n{r.stderr[-1500:]}"
    return r.stdout


class TestSeedBaselineCalibration:
    @needs_png_target
    def test_baseline_records_seed_coverage(self, tmp_path):
        """Falsification: seeds' own coverage must enter the tracker.

        Before the fix, a campaign over valid PNGs sat at single-digit
        edges because only mutants were ever executed.
        """
        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        for i in range(3):
            (seeds / f"s{i}.png").write_bytes(_valid_png(width=8 + 8 * i))
        out = _run_isolated(
            "from pathlib import Path\n"
            f"corpus = Path({str(tmp_path)!r})\n"
            "import sys; sys.path.insert(0, 'tests')\n"
            "from test_regression_seed_baseline import _make_fuzzer\n"
            "f = _make_fuzzer(corpus)\n"
            "f._load_corpus()\n"
            "assert len(f.corpus) == 3, len(f.corpus)\n"
            "f._calibrate_seed_baselines()\n"
            "assert f.shm_cov.cumulative_edges > 50, f.shm_cov.cumulative_edges\n"
            "print('ok')\n"
        )
        assert "ok" in out

    @needs_png_target
    def test_calibration_is_idempotent_across_calls(self, tmp_path):
        """A second pass must add (almost) nothing.

        Not exact equality: direct_lite never resets between execs, so
        the shim's carried-over __afl_prev_loc makes even an identical
        input claim a handful of fresh slots on the next pass. The
        invariant worth pinning is that re-calibration converges — the
        growth is a rounding error against the first pass.
        """
        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        (seeds / "s0.png").write_bytes(_valid_png())
        out = _run_isolated(
            "from pathlib import Path\n"
            f"corpus = Path({str(tmp_path)!r})\n"
            "import sys; sys.path.insert(0, 'tests')\n"
            "from test_regression_seed_baseline import _make_fuzzer\n"
            "f = _make_fuzzer(corpus)\n"
            "f._load_corpus()\n"
            "f._calibrate_seed_baselines()\n"
            "first = f.shm_cov.cumulative_edges\n"
            "f._calibrate_seed_baselines()\n"
            "second = f.shm_cov.cumulative_edges\n"
            f"assert first > 50, first\n"
            "assert second - first <= max(8, first // 20), (first, second)\n"
            "print('ok')\n"
        )
        assert "ok" in out

    @needs_crash_target
    def test_crashing_seed_does_not_kill_startup(self, tmp_path):
        """Adversarial: a hostile seed must be counted, not fatal."""
        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        (seeds / "boom").write_bytes(b"CRASHS")  # test_target's crash magic
        f = Fuzzer(
            target=str(_CRASH_TARGET),
            corpus_dir=str(tmp_path),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=64,
            timeout=5,
            use_coverage=True,
            cmplog=False,
        )
        f._load_corpus()
        f._calibrate_seed_baselines()  # must not raise
        assert f.crash_count >= 1, "crashing seed was not accounted for"

    @needs_png_target
    def test_noop_without_coverage_or_corpus(self, tmp_path):
        """Empty corpus / no SHM paths must early-return cleanly."""
        (tmp_path / "seeds").mkdir(parents=True)
        f = _make_fuzzer(tmp_path)
        f._load_corpus()
        f.shm_cov.cleanup()
        f.shm_cov = None
        f._calibrate_seed_baselines()  # no shm — must return silently
        f2 = _make_fuzzer(tmp_path)
        f2.corpus = []
        f2._calibrate_seed_baselines()  # empty corpus — same

    @needs_png_target
    def test_baseline_feeds_good_turing(self, tmp_path):
        """The estimator must see a real universe after calibration.

        Pre-fix symptom: saturation pinned at 100% over a handful of
        edges because N2 < 10 damping collapsed estimated_undiscovered.
        """
        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        for i in range(4):
            (seeds / f"s{i}.png").write_bytes(_valid_png(width=8 + 8 * i))
        out = _run_isolated(
            "from pathlib import Path\n"
            f"corpus = Path({str(tmp_path)!r})\n"
            "import sys; sys.path.insert(0, 'tests')\n"
            "from test_regression_seed_baseline import _make_fuzzer\n"
            "f = _make_fuzzer(corpus)\n"
            "f._load_corpus()\n"
            "f._calibrate_seed_baselines()\n"
            "gt = f._edge_tracker.good_turing_estimate()\n"
            "assert gt['n'] > 50, gt\n"
            "assert len(f.shm_cov._seen_edge_ids) > 50, len(f.shm_cov._seen_edge_ids)\n"
            "print('ok')\n"
        )
        assert "ok" in out
