"""gcc float/double compares (``__sanitizer_cov_trace_cmpf`` / ``cmpd``).

gcc ``-fsanitize-coverage=trace-cmp`` instruments float and double compares
too; clang never does. Before the shim defined both callbacks, a gcc
trace-cmp target with a floating compare failed to link.

Both log the operands' IEEE bit patterns as 4/8-byte CMP records: the bytes
the input carries, so input-to-state can find and replace them.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector
from tests.test_gcc_trace_pc import requires_gcc12

AFL_SHIM = Path("src/fuzzer_tool/adapters/afl_shim.c")

# Operands come from the input bytes (memcpy), so no constant folding and no
# text-to-float conversion stands between the input and the compare.
_TARGET_C = r"""
#include <stdio.h>
#include <string.h>
#include <unistd.h>
volatile int sink;
int main(void) {
    unsigned char in[12] = {0};
    ssize_t n = read(0, in, sizeof in);
    (void)n;
    float f;
    double d;
    memcpy(&f, in, 4);
    memcpy(&d, in + 4, 8);
    if (f == 3.25f) sink = 1;
    if (d < 1234.5) sink = 2;
    printf("%d\n", sink);
    return 0;
}
"""

_FLOAT_CONST = 3.25
_DOUBLE_CONST = 1234.5


@pytest.fixture(scope="module")
def float_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("trace_cmp_float")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "gcc",
            "-O1",
            "-fsanitize-coverage=trace-pc,trace-cmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    return exe


def _pairs(exe: Path, tmp_path: Path, data: bytes) -> set[tuple[bytes, bytes]]:
    c = CmplogCollector()
    c.log_path = str(tmp_path / "cmplog.txt")
    Path(c.log_path).touch()
    env = dict(os.environ, _CMPLOG_OUT=c.log_path)
    env.pop("_CMPLOG_COUNTS", None)
    env.pop("_CMPLOG_SITE_COUNTS", None)
    env.pop("LD_PRELOAD", None)
    subprocess.run([str(exe)], input=data, capture_output=True, env=env, timeout=60, check=True)
    c.collect_tokens()
    return set(c.pairs)


def _has_pair(pairs, a: bytes, b: bytes) -> bool:
    return (a, b) in pairs or (b, a) in pairs


@requires_gcc12
class TestFloatCompares:
    def test_float_bits_reach_the_pair_pool(self, tmp_path, float_target):
        f_in = struct.pack("<f", 7.5)
        pairs = _pairs(float_target, tmp_path, f_in + struct.pack("<d", 9.0))
        assert _has_pair(pairs, f_in, struct.pack("<f", _FLOAT_CONST))

    def test_double_bits_reach_the_pair_pool(self, tmp_path, float_target):
        d_in = struct.pack("<d", 9.0)
        pairs = _pairs(float_target, tmp_path, struct.pack("<f", 7.5) + d_in)
        assert _has_pair(pairs, d_in, struct.pack("<d", _DOUBLE_CONST))

    def test_falsify_not_integer_converted(self, tmp_path, float_target):
        """Records carry IEEE bits, not the value cast to an integer (7, 3)."""
        pairs = _pairs(float_target, tmp_path, struct.pack("<f", 7.5) + struct.pack("<d", 9.0))
        assert not _has_pair(pairs, struct.pack("<I", 7), struct.pack("<I", 3))
        assert not _has_pair(pairs, struct.pack("<Q", 9), struct.pack("<Q", 1234))

    def test_adversarial_nan_and_negative_zero(self, tmp_path, float_target):
        """NaN (unordered) and -0.0 (numerically equal, bitwise not) still log
        exact bit patterns and do not crash the target."""
        nan = struct.pack("<I", 0x7FC00001)
        neg_zero = struct.pack("<d", -0.0)
        pairs = _pairs(float_target, tmp_path, nan + neg_zero)
        assert _has_pair(pairs, nan, struct.pack("<f", _FLOAT_CONST))
        assert _has_pair(pairs, neg_zero, struct.pack("<d", _DOUBLE_CONST))
