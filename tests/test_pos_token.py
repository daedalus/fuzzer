"""PositionTokenScheduler: land mutations inside dictionary-token occurrences.

Covers core/schedulers/pos_token.py.
"""

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_token import (
    EPSILON,
    MAX_MATCHES,
    MAX_SEEDS,
    MIN_TOKEN_LEN,
    REBUILD_EVERY,
    PositionTokenScheduler,
)

SEED = b"....IHDR........IDAT....IEND...."
NO_ESCAPE = 0.99  # random() draw above EPSILON


class ScriptedRng:
    """Scripted random()/randint(); randint asserts and logs its bounds."""

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)
        self.bounds = []

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        self.bounds.append((a, b))
        v = self._ints.pop(0) if self._ints else a
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v


def _sched(rng=None, tokens=(b"IHDR", b"IDAT", b"IEND")):
    box = {"t": list(tokens)}
    s = PositionTokenScheduler(rng or ScriptedRng(), tokens_of=lambda: box["t"])
    return s, box


def _starts(data, tokens):
    """Independent oracle: every occurrence start, sorted."""
    out = []
    for t in tokens:
        i = data.find(t)
        while i >= 0:
            out.append((i, len(t)))
            i = data.find(t, i + 1)
    return sorted(out)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched()[0], PositionScheduler)

    def test_record_is_a_noop(self):
        s, _ = _sched(ScriptedRng(ints=[0, 0]))
        s.record(SEED, [5], Outcome.GAIN, 1.0)
        assert s.propose(SEED, len(SEED)) == SEED.find(b"IHDR")

    def test_active_needs_a_token(self):
        assert _sched()[0].active()
        assert not _sched(tokens=())[0].active()


class TestDeclines:
    def test_no_tokens(self):
        assert _sched(tokens=())[0].propose(SEED, len(SEED)) is None

    def test_no_occurrence(self):
        assert _sched(tokens=(b"zzzz",))[0].propose(SEED, len(SEED)) is None

    def test_empty_data_or_buffer(self):
        s, _ = _sched()
        assert s.propose(b"", 5) is None
        assert s.propose(SEED, 0) is None

    def test_epsilon_escape(self):
        s, _ = _sched(ScriptedRng(randoms=[EPSILON / 2]))
        assert s.propose(SEED, len(SEED)) is None

    def test_every_match_past_a_shrunk_buffer(self):
        s, _ = _sched(tokens=(b"IEND",))
        assert s.propose(SEED, SEED.find(b"IEND")) is None

    def test_adversarial_short_tokens_are_ignored(self):
        # One-byte tokens match everywhere: they carry no position signal.
        short = bytes([SEED[0]]) * (MIN_TOKEN_LEN - 1)
        assert _sched(tokens=(short,))[0].propose(SEED, len(SEED)) is None


class TestPicks:
    def test_lands_inside_the_scripted_occurrence(self):
        occ = _starts(SEED, (b"IHDR", b"IDAT", b"IEND"))
        for i, (start, width) in enumerate(occ):
            s, _ = _sched(ScriptedRng(ints=[i, width - 1]))
            assert s.propose(SEED, len(SEED)) == start + width - 1

    def test_falsification_only_token_bytes(self):
        occ = _starts(SEED, (b"IHDR", b"IDAT", b"IEND"))
        inside = {s + k for s, w in occ for k in range(w)}
        picks = set()
        for i, (_, width) in enumerate(occ):
            for k in range(width):
                s, _ = _sched(ScriptedRng(ints=[i, k]))
                picks.add(s.propose(SEED, len(SEED)))
        assert picks == inside

    def test_adversarial_token_straddling_a_shrunk_buffer(self):
        # IDAT starts inside buf_len but ends past it: land in range only.
        cut = SEED.find(b"IDAT") + 2
        s, _ = _sched(ScriptedRng(ints=[1, 1]))
        assert s.propose(SEED, cut) == cut - 1

    def test_overlapping_occurrences(self):
        data = b"AAAAA"
        occ = _starts(data, (b"AA",))
        s, _ = _sched(ScriptedRng(ints=[len(occ) - 1, 1]), tokens=(b"AA",))
        assert s.propose(data, len(data)) == occ[-1][0] + 1


class TestCaches:
    def test_matches_capped_and_spread(self):
        data = b"ab" * (MAX_MATCHES * 3)
        s, _ = _sched(ScriptedRng(ints=[0, 0]), tokens=(b"ab",))
        s.propose(data, len(data))
        starts = s.match_starts(data)
        assert len(starts) == MAX_MATCHES
        assert starts[-1] > len(data) // 2  # not just the head of the seed

    def test_seed_cache_bounded(self):
        s, _ = _sched()
        for i in range(MAX_SEEDS + 3):
            s.propose(SEED + bytes([i % 256, i // 256]), len(SEED))
        assert s.cached_seeds() == MAX_SEEDS

    def test_new_tokens_picked_up_after_rebuild_window(self):
        s, box = _sched(tokens=(b"IHDR",))
        assert s.propose(SEED, len(SEED)) == SEED.find(b"IHDR")
        box["t"] = [b"IEND"]
        for _ in range(REBUILD_EVERY):
            s.propose(SEED, len(SEED))
        assert s.propose(SEED, len(SEED)) == SEED.find(b"IEND")

    def test_regression_growing_token_list_rebuilds_once_per_window(self):
        s, box = _sched(tokens=(b"IHDR",))
        s.propose(SEED, len(SEED))
        builds = s.builds
        for i in range(REBUILD_EVERY - 1):
            box["t"].append(bytes([65 + i % 26]) * 3)
            s.propose(SEED, len(SEED))
        assert s.builds == builds
