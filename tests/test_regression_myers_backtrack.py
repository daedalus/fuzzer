"""The opt-in Myers path must return a script that actually rebuilds the target.

The first version of ``_myers_backtrack`` unwound snakes greedily instead of
bounding them by the previous endpoint, so it consumed match runs belonging to
lower D levels and then fell out of the loop with ``(x, y)`` short of the
origin. The script it returned was silently truncated: it round-tripped when
``D == 1`` and was wrong for every larger edit distance, while still looking
plausible (right op types, monotone positions, no exception).

``root_cause`` replays these scripts positionally during delta debugging, so a
truncated script is not a cosmetic defect -- it reconstructs a candidate that
is not the crash.
"""

import random

import pytest

from fuzzer_tool.core.similarity import configure_diff_myers, levenshtein_align


def apply_script(base: bytes, ops: list[tuple[str, int, bytes]]) -> bytes:
    """Rebuild the target from ``base`` by replaying an edit script."""
    out = bytearray()
    for op, pos, data in ops:
        if op == "match":
            out.append(base[pos])
        elif op in ("replace", "insert"):
            out += data
        elif op == "delete":
            continue
        else:  # pragma: no cover - guards a typo in a future op name
            raise AssertionError(f"unknown op {op!r}")
    return bytes(out)


def non_match(ops) -> int:
    return sum(1 for op, _pos, _data in ops if op != "match")


@pytest.fixture
def myers_on():
    configure_diff_myers(True)
    yield
    configure_diff_myers(False)


# Whether the greedy unwind over-consumed depended on where match runs happened
# to fall, so most hand-picked inputs round-tripped even on the broken version.
# This pair is the smallest found (70 bytes, 3 edits) that does not, pinned as
# literals so the test keeps falsifying regardless of RNG behaviour.
_PINNED_A = bytes.fromhex(
    "8f8549b0d729bc8ec82dd4bdf057fad7a1bd1f0f9782b0408b8e646eb61c81ff"
    "c49c0f8ef65069c08e65b71100322a9ff761f5be0be4a3481f7184a5f11d48c3"
    "759b1506b03e"
)
_PINNED_B = bytes.fromhex(
    "8f8549b029bc8ec82dd4bdf057fad7a1bde00f9782b0408b8e646eb61c81ffc4"
    "9c0f8ef669c08e65b71100322a9ff761f5be0be4a3481f7184a5f11d48c3759b"
    "1506b0af3e"
)


def test_pinned_case_rebuilds_target(myers_on):
    """The minimal known input that the greedy-unwind version got wrong."""
    ops = levenshtein_align(_PINNED_A, _PINNED_B)
    assert apply_script(_PINNED_A, ops) == _PINNED_B


def test_pinned_case_default_path(_unused=None):
    """Control: the DP path must pass the same oracle, or the oracle is wrong."""
    configure_diff_myers(False)
    ops = levenshtein_align(_PINNED_A, _PINNED_B)
    assert apply_script(_PINNED_A, ops) == _PINNED_B


def test_mixed_edits_round_trip(myers_on):
    """Insertions and deletions, not just substitutions."""
    failures = []
    for t in range(120):
        rng = random.Random(t)
        a = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
        b = bytearray(a)
        for _ in range(rng.randrange(0, 12)):
            if not b:
                b = bytearray(rng.randbytes(3))
                continue
            p = rng.randrange(len(b))
            roll = rng.random()
            if roll < 0.34:
                b[p] ^= 0xFF
            elif roll < 0.67:
                del b[p]
            else:
                b.insert(p, rng.randrange(256))
        b = bytes(b)
        if apply_script(a, levenshtein_align(a, b)) != b:
            failures.append(t)
    assert not failures, f"scripts failed to rebuild the target for seeds {failures}"


def test_empty_and_degenerate_inputs(myers_on):
    for a, b in ((b"", b""), (b"", b"abc"), (b"abc", b""), (b"a", b"a")):
        assert apply_script(a, levenshtein_align(a, b)) == b


def test_myers_never_beats_levenshtein_and_is_close(myers_on):
    """Myers minimises indel distance; the DP minimises Levenshtein.

    Substitution costs 2 in Myers' model and 1 in the DP's, so after folding
    adjacent delete/insert pairs back into ``replace`` the Myers script can
    still carry a few extra ops. It must never carry *fewer* -- that would mean
    it had found a script shorter than the true Levenshtein optimum, i.e. an
    invalid one -- and it must stay tight rather than drifting.
    """
    worse = 0
    total = 0
    for t in range(120):
        rng = random.Random(t)
        a = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
        b = bytearray(a)
        for _ in range(rng.randrange(0, 12)):
            if not b:
                b = bytearray(rng.randbytes(3))
                continue
            p = rng.randrange(len(b))
            roll = rng.random()
            if roll < 0.34:
                b[p] ^= 0xFF
            elif roll < 0.67:
                del b[p]
            else:
                b.insert(p, rng.randrange(256))
        b = bytes(b)

        configure_diff_myers(False)
        dp = non_match(levenshtein_align(a, b))
        configure_diff_myers(True)
        my = non_match(levenshtein_align(a, b))

        assert my >= dp, f"seed {t}: myers={my} < dp={dp}, script cannot be valid"
        total += 1
        if my > dp:
            worse += 1
    # Empirically 12/400 on this generator; keep a loose ceiling so a real
    # regression in the coalescing shows up but noise does not.
    assert worse <= total // 5, f"{worse}/{total} scripts lost substitution folding"


def test_substitutions_fold_into_replace(myers_on):
    """A clean one-byte substitution is one op, not a delete plus an insert."""
    a = bytes(random.Random(0).randrange(256) for _ in range(256))
    b = bytearray(a)
    b[100] ^= 0xFF
    ops = levenshtein_align(a, bytes(b))
    assert apply_script(a, ops) == bytes(b)
    assert [op for op, _p, _d in ops if op != "match"] == ["replace"]
