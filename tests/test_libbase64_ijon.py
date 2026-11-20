"""libbase64 target IJON slots (--ijon), driven through a fake libbase64.

The SIMD codecs validate a whole block branch-free, so edge coverage cannot
see where in a block the first bad char sits, nor whether an input of a
given tail residue was long enough to enter every main loop. The target
exposes both as IJON_MAX slots:

    slot = BAD_BAND[mode] + (pos & 63)     first bad char, decode modes
    slot = LEN_BAND + mode*64 + (n & 63)   payload length, every mode
    value = min(x, CAP) + 1                bounded; 0 means "never set"

The fake library keeps the test independent of the vendored tree: plain
decode accepts [A-Za-z0-9+/=] with len % 4 == 0; every SIMD codec reports
"not built" (-1), so only the plain reference runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import IjonMaxShm, ShmCoverage
from fuzzer_tool.core.ijon import IJON_MAP_SLOTS
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

TARGET = "targets/libbase64_read.c"
MAP_ENTRIES = 8192
IJON_ENV = "__AFL_IJON_SHM_ID"

# Slot layout spec (independent of the C constants).
LANES = 64
BAD_BAND = {0: 0, 2: LANES}
LEN_BAND = 2 * LANES
CAP = 256
DECODE, ENCODE, STREAM_DECODE = 0, 1, 2

FAKE_LIB = r"""
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "libbase64.h"

static int fake_chars(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '='))
            return 0;
    }
    return 1;
}
static int fake_ok(const char *s, size_t n) { return n % 4 == 0 && fake_chars(s, n); }
int base64_decode(const char *s, size_t n, char *o, size_t *ol, int f) {
    (void)f; *ol = n / 4 * 3; memset(o, 0, *ol); return fake_ok(s, n);
}
void base64_encode(const char *s, size_t n, char *o, size_t *ol, int f) {
    (void)s; (void)f; *ol = 4 * ((n + 2) / 3); memset(o, 'A', *ol);
}
void base64_stream_decode_init(struct base64_state *st, int f) {
    memset(st, 0, sizeof *st); st->flags = f;
}
/* libbase64 contract: 1 ok, 0 bad input, -1 codec not built. A partial
 * quantum carries over in st->bytes, as in the real stream decoder. */
int base64_stream_decode(struct base64_state *st, const char *s, size_t n,
                         char *o, size_t *ol) {
    if (st->flags != BASE64_FORCE_PLAIN) return -1;
    size_t total = (size_t)st->bytes + n;
    *ol = total / 4 * 3; memset(o, 0, *ol);
    st->bytes = (int)(total % 4);
    return fake_chars(s, n);
}
void base64_stream_encode_init(struct base64_state *st, int f) { (void)st; (void)f; abort(); }
void base64_stream_encode(struct base64_state *st, const char *s, size_t n,
                          char *o, size_t *ol) { (void)st; (void)s; (void)n; (void)o; (void)ol; abort(); }
void base64_stream_encode_final(struct base64_state *st, char *o, size_t *ol) {
    (void)st; (void)o; (void)ol; abort();
}
"""

FAKE_HDR = r"""
#pragma once
#include <stddef.h>
#define BASE64_FORCE_AVX2   (1 << 0)
#define BASE64_FORCE_NEON32 (1 << 1)
#define BASE64_FORCE_NEON64 (1 << 2)
#define BASE64_FORCE_PLAIN  (1 << 3)
#define BASE64_FORCE_SSSE3  (1 << 4)
#define BASE64_FORCE_SSE41  (1 << 5)
#define BASE64_FORCE_SSE42  (1 << 6)
#define BASE64_FORCE_AVX    (1 << 7)
#define BASE64_FORCE_AVX512 (1 << 8)
struct base64_state { int eof; int bytes; int flags; unsigned char carry; };
int base64_decode(const char *, size_t, char *, size_t *, int);
void base64_encode(const char *, size_t, char *, size_t *, int);
void base64_stream_decode_init(struct base64_state *, int);
int base64_stream_decode(struct base64_state *, const char *, size_t, char *, size_t *);
void base64_stream_encode_init(struct base64_state *, int);
void base64_stream_encode(struct base64_state *, const char *, size_t, char *, size_t *);
void base64_stream_encode_final(struct base64_state *, char *, size_t *);
"""

HARNESS = r"""
#include <stdio.h>
int main(int argc, char **argv) {
    static unsigned char buf[1 << 16];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    return fuzz_shm_run(buf, n);
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    from pathlib import Path

    tmp = tmp_path_factory.mktemp("b64ijon")
    (tmp / "libbase64.h").write_text(FAKE_HDR)
    target = Path(__file__).resolve().parents[1] / TARGET
    src = f'{FAKE_LIB}\n#include "{target}"\n{HARNESS}'
    return _build(tmp, src, "-fsanitize-coverage=trace-pc-guard", f"-I{tmp}")


def _slots(exe, tmp_path, data: bytes, ijon: bool = True) -> np.ndarray | None:
    import subprocess

    inp = tmp_path / "in.bin"
    inp.write_bytes(data)
    shm = ShmCoverage(size=MAP_ENTRIES)
    seg = IjonMaxShm(slots=IJON_MAP_SLOTS) if ijon else None
    try:
        env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        if seg is not None:
            env[IJON_ENV] = seg.env_id
        rc = subprocess.run([str(exe), str(inp)], env=env, timeout=30).returncode
        assert rc == 0
        if seg is None:
            return None
        out = np.zeros(seg.slots, dtype=np.uint64)
        seg.read_into(out)
        return out
    finally:
        shm.cleanup()
        if seg is not None:
            seg.cleanup()


def _val(x: int) -> int:
    return min(x, CAP) + 1


def _len_slot(mode: int, n: int) -> int:
    return LEN_BAND + mode * LANES + (n % LANES)


@requires_clang
def test_bad_char_lane_and_length(exe, tmp_path):
    payload = b"QUJD" * 5 + b"!" + b"QUJ"  # bad char at 20, len 24
    v = _slots(exe, tmp_path, bytes([DECODE]) + payload)

    bad = BAD_BAND[DECODE] + 20
    ln = _len_slot(DECODE, len(payload))
    assert int(v[bad]) == _val(20)
    assert int(v[ln]) == _val(len(payload))
    assert int(np.count_nonzero(v)) == 2


@requires_clang
def test_falsify_valid_decode_sets_no_bad_slot(exe, tmp_path):
    payload = b"QUJD" * 8
    v = _slots(exe, tmp_path, bytes([DECODE]) + payload)

    assert int(v[_len_slot(DECODE, len(payload))]) == _val(len(payload))
    assert int(np.count_nonzero(v[:LEN_BAND])) == 0, "valid input must not touch bad-char band"


@requires_clang
def test_falsify_no_segment_is_inert(exe, tmp_path):
    assert _slots(exe, tmp_path, b"\x00QU!D", ijon=False) is None


@requires_clang
def test_adversarial_deep_bad_char_and_length_capped(exe, tmp_path):
    pos = 3 * CAP + 7
    payload = b"A" * pos + b"\xff" + b"A" * 3
    v = _slots(exe, tmp_path, bytes([DECODE]) + payload)

    assert int(v[BAD_BAND[DECODE] + pos % LANES]) == _val(pos) == CAP + 1
    assert int(v[_len_slot(DECODE, len(payload))]) == CAP + 1


@requires_clang
def test_adversarial_bad_char_at_zero_is_nonzero(exe, tmp_path):
    """pos 0 must still register: value 0 means "never set"."""
    v = _slots(exe, tmp_path, b"\x00!QUJ")
    assert int(v[BAD_BAND[DECODE]]) == 1


@requires_clang
def test_adversarial_stream_mode_uses_own_band(exe, tmp_path):
    """Stream decode skips its seed byte; bands must not alias mode 0."""
    payload = b"QU!D"
    v = _slots(exe, tmp_path, bytes([STREAM_DECODE, 0x07]) + payload)

    assert int(v[BAD_BAND[STREAM_DECODE] + 2]) == _val(2)
    assert int(v[BAD_BAND[DECODE] + 2]) == 0
    assert int(v[_len_slot(STREAM_DECODE, len(payload))]) == _val(len(payload))


@requires_clang
def test_adversarial_empty_payload_length_zero(exe, tmp_path):
    v = _slots(exe, tmp_path, bytes([DECODE]))
    assert int(v[_len_slot(DECODE, 0)]) == 1
