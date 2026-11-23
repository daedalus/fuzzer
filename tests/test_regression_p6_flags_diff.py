"""``targets/p6_flags_diff.c``: P6 micro-op flag/alias model vs native x86.

The model follows nand2mario's P6 post (2026-11-22): per-register flags, ``U2``
publish masks, ``MOVEMERGEFLGS`` for ADC, branchless ``MERGE``/``CMOV`` CMPXCHG.
The reference is the host CPU, driven through a JIT'd instruction.

Hard Rule 46: the control (``exh`` on the unmodified model) must pass before a
planted bug (``-DP6_BUG=n``) is allowed to prove the comparison can fail.
"""

from __future__ import annotations

import random
import re
import shutil
import signal
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "targets" / "p6_flags_diff.c"
_CC = shutil.which("clang") or shutil.which("gcc")

REC = 12
MAX_RECS = 64
OP_ADC, OP_INC, OP_CMPXCHG, OP_MINC = 2, 8, 19, 21
W32 = 2

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not _CC, reason="needs linux x86-64 and a C compiler"
)


def _build(out: Path, *extra: str) -> Path:
    cmd = [_CC, "-O1", "-g", "-fno-sanitize=all", *extra, "-o", str(out), str(_SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"build failed: {r.stderr[:800]}")
    return out


@pytest.fixture(scope="module")
def good(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("p6") / "good")


def _bad(tmp_path_factory, bug: int) -> Path:
    return _build(tmp_path_factory.mktemp(f"p6bug{bug}") / "bad", f"-DP6_BUG={bug}")


def _rec(op, width=W32, d=0, s=1, flags=0, disp=0, a=0, b=0) -> bytes:
    ctl = width | d << 2 | s << 4
    return struct.pack("<BBBBII", op, ctl, flags, disp, a, b)


def _run(exe: Path, *args: str, data: bytes | None = None):
    return subprocess.run(
        [str(exe), *args], input=data, capture_output=True, timeout=120
    )


def test_control_model_matches_hardware(good):
    r = _run(good, "exh")

    assert r.returncode == 0, r.stderr.decode()
    assert b"mismatches=0" in r.stdout


def test_exh_covers_every_op(good):
    r = _run(good, "exh")
    names = re.findall(rb"^op (\w+) ", r.stdout, re.M)

    assert len(names) == int(_run(good, "list").stdout.split()[0])


@pytest.mark.parametrize("bug", [1, 2, 3, 4])
def test_planted_bug_fails_exh(tmp_path_factory, bug):
    r = _run(_bad(tmp_path_factory, bug), "exh")

    assert r.returncode != 0
    assert b"mismatches=0" not in r.stdout


# (bug, record) pairs that must abort in fuzz mode: carry-in set where it matters.
_TRIGGERS = [
    (1, _rec(OP_INC, flags=0x01, a=5)),
    (2, _rec(OP_ADC, flags=0x01, a=1, b=1)),
    (3, _rec(OP_CMPXCHG, d=1, s=2, a=7, b=9)),
    (4, _rec(OP_MINC, flags=0, a=0xFFFFFFFF, b=0x7FFFFFF0)),
]


@pytest.mark.parametrize(("bug", "rec"), _TRIGGERS)
def test_planted_bug_aborts_in_fuzz_mode(tmp_path_factory, tmp_path, bug, rec):
    f = tmp_path / "in.bin"
    f.write_bytes(rec)

    r = _run(_bad(tmp_path_factory, bug), "fuzz", str(f))

    assert r.returncode == -signal.SIGABRT
    assert b"MISMATCH" in r.stderr


@pytest.mark.parametrize(("bug", "rec"), _TRIGGERS)
def test_same_record_passes_on_control(good, tmp_path, bug, rec):
    f = tmp_path / "in.bin"
    f.write_bytes(rec)

    assert _run(good, "fuzz", str(f)).returncode == 0


# ---- adversarial: no input may crash the control ----


def test_empty_input_is_noop(good, tmp_path):
    f = tmp_path / "e.bin"
    f.write_bytes(b"")

    assert _run(good, "fuzz", str(f)).returncode == 0


def test_truncated_record_is_padded(good, tmp_path):
    f = tmp_path / "t.bin"
    f.write_bytes(b"\xff" * (REC - 1))

    assert _run(good, "fuzz", str(f)).returncode == 0


def test_all_ones_max_records(good, tmp_path):
    f = tmp_path / "ff.bin"
    f.write_bytes(b"\xff" * REC * MAX_RECS)

    assert _run(good, "fuzz", str(f)).returncode == 0


def test_oversized_input_is_capped(good, tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"\x00" * REC * MAX_RECS * 50)

    assert _run(good, "fuzz", str(f)).returncode == 0


@pytest.mark.parametrize("seed", range(40))
def test_random_streams_agree(good, tmp_path, seed):
    rng = random.Random(seed)
    f = tmp_path / "r.bin"
    f.write_bytes(rng.randbytes(rng.randrange(1, REC * MAX_RECS)))

    r = _run(good, "fuzz", str(f))

    assert r.returncode == 0, r.stderr.decode()


def test_stdin_entry(good):
    assert _run(good, data=_rec(OP_INC, flags=1, a=0xFFFFFFFF)).returncode == 0
