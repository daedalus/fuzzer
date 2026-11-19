"""Smith-Waterman-Gotoh crash localization (core/local_align.py).

The numpy DP is checked against an independent scalar Gotoh written here,
and every traceback is replayed byte by byte against both inputs.
"""

import re

import pytest

from fuzzer_tool.core.local_align import (
    GAP_EXTEND,
    GAP_OPEN,
    MATCH,
    MAX_CELLS,
    MISMATCH,
    Fit,
    local_align,
)

_NEG = -(1 << 40)
_CIGAR = re.compile(r"(\d+)([MXID])")


def _ref_score(a: bytes, b: bytes) -> int:
    """Scalar Gotoh local score: H (any), E (gap in a), F (gap in b)."""
    m = len(b)
    h_prev = [0] * (m + 1)
    f_prev = [_NEG] * (m + 1)
    best = 0
    for i in range(1, len(a) + 1):
        h = [0] * (m + 1)
        e = _NEG
        f = [_NEG] * (m + 1)
        for j in range(1, m + 1):
            e = max(h[j - 1] + GAP_OPEN, e + GAP_EXTEND)
            f[j] = max(h_prev[j] + GAP_OPEN, f_prev[j] + GAP_EXTEND)
            sub = MATCH if a[i - 1] == b[j - 1] else MISMATCH
            h[j] = max(0, h_prev[j - 1] + sub, e, f[j])
            best = max(best, h[j])
        h_prev, f_prev = h, f
    return best


def _lcg(seed: int, n: int, alphabet: int) -> bytes:
    """Deterministic byte stream (no global RNG in tests)."""
    out = bytearray()
    x = seed
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append((x >> 33) % alphabet)
    return bytes(out)


def _replay(parent: bytes, crash: bytes, al) -> int:
    """Walk the CIGAR over both inputs; return its recomputed score."""
    i, j = al.parent_span[0], al.crash_span[0]
    score, last = 0, ""
    for count, op in _CIGAR.findall(al.cigar):
        for _ in range(int(count)):
            if op in "MX":
                assert (parent[i] == crash[j]) == (op == "M")
                score += MATCH if op == "M" else MISMATCH
                i, j = i + 1, j + 1
            else:
                score += GAP_EXTEND if op == last else GAP_OPEN
                i, j = (i + 1, j) if op == "D" else (i, j + 1)
            last = op
    assert (i, j) == (al.parent_span[1], al.crash_span[1])
    return score


def _distinct_ends(a: bytes, b: bytes) -> tuple[bytes, bytes]:
    """Force no common prefix/suffix, so the score is the pure SW score."""
    return b"\xfe" + a + b"\xfd", b"\xfc" + b + b"\xfb"


def test_matches_scalar_reference():
    # Control first (Hard Rule 46): the reference agrees with itself.
    for seed in range(40):
        a = _lcg(seed, 3 + seed % 29, 4)
        b = _lcg(seed + 1000, 2 + (seed * 7) % 31, 4)
        a, b = _distinct_ends(a, b)
        assert _ref_score(a, b) == _ref_score(a, b)

        al = local_align(a, b)
        assert al.fit is Fit.EXACT
        assert al.score == _ref_score(a, b)
        assert _replay(a, b, al) == al.score


def test_identical_no_edits():
    data = _lcg(1, 300, 256)
    al = local_align(data, data)
    assert al.edits == () and al.deletions == ()


def test_substitution_span():
    parent = _lcg(2, 400, 256)
    crash = bytearray(parent)
    crash[123] ^= 0xFF
    al = local_align(parent, bytes(crash))
    assert al.edits == ((123, 124),)
    assert al.deletions == ()


def test_insertion_robust_to_shift():
    """An 8-byte insert is one span; the shifted tail is not flagged."""
    parent = _lcg(3, 500, 256)
    crash = parent[:200] + b"\x00" * 8 + parent[200:]
    al = local_align(parent, crash)
    assert al.edits == ((200, 208),)


def test_deletion_reported_as_crash_offset():
    parent = _lcg(4, 500, 256)
    crash = parent[:300] + parent[305:]
    al = local_align(parent, crash)
    assert al.edits == ()
    assert al.deletions == (300,)


def test_indel_plus_substitutions_inside_core():
    """Edits inside one middle: the DP (not affix trimming) separates them."""
    parent = _lcg(5, 600, 256)
    crash = bytearray(parent[:100] + b"INSERTED" + parent[100:400] + parent[410:])
    crash[250] ^= 0x55
    al = local_align(parent, bytes(crash))
    assert al.fit is Fit.EXACT
    assert al.edits == ((100, 108), (250, 251))
    assert al.deletions == (408,)
    assert _replay(parent, bytes(crash), al) <= al.score


def test_appended_tail_is_edit():
    parent = _lcg(6, 100, 256)
    crash = parent + b"TAIL"
    assert local_align(parent, crash).edits == ((100, 104),)


def test_falsification_hamming_would_flag_tail():
    """Sanity of the premise: positional diff flags the whole shifted tail."""
    parent = _lcg(7, 200, 256)
    crash = parent[:50] + b"Z" + parent[50:]
    positional = sum(x != y for x, y in zip(parent, crash, strict=False))
    assert positional > 100
    assert local_align(parent, crash).edits == ((50, 51),)


@pytest.mark.parametrize(
    "parent,crash",
    [(b"", b""), (b"", b"abc"), (b"abc", b""), (b"a", b"b"), (b"x" * 50, b"y" * 60)],
)
def test_adversarial_degenerate(parent, crash):
    al = local_align(parent, crash)
    covered = sum(e - s for s, e in al.edits)
    assert covered <= len(crash)
    if parent != crash and crash and not set(parent) & set(crash):
        assert al.edits == ((0, len(crash)),)


def test_adversarial_repeats_ambiguous_insert():
    """Homopolymer insert: any 1-byte span is valid, exactly one is reported."""
    al = local_align(b"A" * 1000, b"A" * 1001)
    assert len(al.edits) == 1
    assert al.edits[0][1] - al.edits[0][0] == 1


def test_adversarial_over_cap_falls_back_coarse():
    side = int(MAX_CELLS**0.5) + 64
    parent = b"P" + _lcg(8, side, 256) + b"Q"
    crash = b"C" + _lcg(9, side, 256) + b"D"
    al = local_align(parent, crash)
    assert al.fit is Fit.COARSE
    assert al.edits == ((0, len(crash)),)


def test_large_input_small_diff_stays_exact():
    """Affix trimming keeps a 1 MiB input with one edit inside the cap."""
    parent = _lcg(10, 1 << 20, 256)
    crash = parent[:777_777] + b"!!" + parent[777_777:]
    al = local_align(parent, crash)
    assert al.fit is Fit.EXACT
    assert al.edits == ((777_777, 777_779),)


def test_root_cause_reports_mutated_region(tmp_path):
    """Wiring: root_cause attaches indel-robust edit spans to its result."""
    from unittest.mock import patch

    from fuzzer_tool.services.root_cause import root_cause

    parent = _lcg(11, 64, 200)
    crash_input = parent[:20] + b"\xffBUG" + parent[20:]
    crash = tmp_path / "crash.bin"
    crash.write_bytes(crash_input)
    base = tmp_path / "base.bin"
    base.write_bytes(parent)

    def fake_run(target, data, timeout, env=None):
        return (-11, "", 1) if b"\xffBUG" in data else (0, "", 1)

    with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
        result = root_cause("/bin/true", str(crash), baseline_file=str(base))

    assert result is not None
    assert result["mutated_spans"] == ((20, 24),)
    assert "Mutated region: edits 0x14-0x18" in result["report"]
