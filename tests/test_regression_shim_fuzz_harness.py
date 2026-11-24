"""tools/shim_fuzz: the AFL++ differential harness for afl_shim.c's cmplog layer.

The harness runs the shim's libc interposers and sanitizer callbacks against
operands flush with a PROT_NONE page, and (records-only mode) checks that a
non-equal compare really lands in the $_CMPLOG_RECORDS_SHM segment.  These
tests run it standalone -- no AFL needed, one seed per process -- so a
regression in the shim shows up in the normal suite, and pin the harness
itself with two controls built from older shims:

  - a55cdd4d^ (before the bounded string scans): the unterminated strcmp seed
    must fault.  If it stops faulting, the harness no longer reaches the
    overread and every "clean" result is meaningless.
  - 93cca99d (before the record-gate fix): the records-only oracle must trip
    on strcmp/strcasecmp/wide compares that were silently dropped.

Controls skip when the revision is not in the local history.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHIM_REL = "src/fuzzer_tool/adapters/afl_shim.c"
TOOL = ROOT / "tools" / "shim_fuzz"
SEEDS = sorted((TOOL / "seeds").iterdir()) if (TOOL / "seeds").is_dir() else []
CC = shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(CC is None or not SEEDS, reason="needs a C compiler and the seeds")

PRE_BOUNDED_SCANS = "a55cdd4d^"
PRE_RECORD_GATES = "93cca99d"


def _cc(args: list[str]) -> None:
    r = subprocess.run([CC, "-O1", "-w", *args], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-800:]


def _shim(tmp: Path, name: str, src: Path) -> Path:
    out = tmp / f"{name}.so"
    _cc(
        [
            "-fPIC",
            "-shared",
            "-fno-omit-frame-pointer",
            "-D__AFL_PRELOAD_ONLY",
            str(src),
            "-o",
            str(out),
            "-ldl",
            "-lpthread",
        ]
    )
    return out


def _shim_at(tmp: Path, rev: str) -> Path:
    r = subprocess.run(
        ["git", "show", f"{rev}:{SHIM_REL}"], cwd=ROOT, capture_output=True, timeout=60
    )
    if r.returncode != 0:
        pytest.skip(f"{rev} not in local history")
    src = tmp / f"shim_{rev.replace('^', '_')}.c"
    src.write_bytes(r.stdout)
    return _shim(tmp, src.stem, src)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shim_fuzz")
    harness = tmp / "harness"
    _cc(["-fno-builtin", str(TOOL / "harness.c"), "-o", str(harness), "-ldl"])
    return tmp, harness, _shim(tmp, "current", ROOT / SHIM_REL)


def _run(harness: Path, shim: Path, seed: Path, env_extra: dict[str, str]) -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("_CMPLOG", "__AFL", "AFL_"))}
    env.update(env_extra, LD_PRELOAD=str(shim))
    with open(seed, "rb") as fh:
        return subprocess.run(
            [str(harness)], stdin=fh, env=env, capture_output=True, timeout=60
        ).returncode


def _rec_segment(tmp: Path):
    exe = tmp / "mkrec"
    if not exe.exists():
        _cc([str(TOOL / "mkrec.c"), "-o", str(exe)])
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("SysV shared memory unavailable")
    return r.stdout.strip()


def _drop(segid: str) -> None:
    subprocess.run(["ipcrm", "-m", segid], capture_output=True)


def test_regression_harness_seeds_clean_text_mode(built):
    tmp, harness, shim = built
    bad = [
        s.name for s in SEEDS if _run(harness, shim, s, {"_CMPLOG_OUT": str(tmp / "cmp.out")}) != 0
    ]
    assert not bad, bad


def test_regression_harness_seeds_clean_records_mode(built):
    tmp, harness, shim = built
    seg = _rec_segment(tmp)
    try:
        bad = [
            s.name
            for s in SEEDS
            if s.name.startswith("rec_")
            and _run(harness, shim, s, {"_CMPLOG_RECORDS_SHM": seg}) != 0
        ]
    finally:
        _drop(seg)
    assert not bad, bad


def test_regression_harness_control_unterminated_strcmp_faults(built):
    tmp, harness, _ = built
    old = _shim_at(tmp, PRE_BOUNDED_SCANS)
    rc = _run(
        harness,
        old,
        TOOL / "seeds" / "ctrl_unterminated_strcmp",
        {"_CMPLOG_OUT": str(tmp / "cmp.out")},
    )
    assert rc == -signal.SIGSEGV, rc


def test_regression_harness_control_dropped_records_trip_oracle(built):
    tmp, harness, _ = built
    old = _shim_at(tmp, PRE_RECORD_GATES)
    seg = _rec_segment(tmp)
    try:
        rcs = {
            n: _run(harness, old, TOOL / "seeds" / n, {"_CMPLOG_RECORDS_SHM": seg})
            for n in ("rec_ne5", "rec_ne7", "rec_ne20", "rec_ne22", "rec_ne19")
        }
    finally:
        _drop(seg)
    assert all(rc == -signal.SIGILL for rc in rcs.values()), rcs
