"""available() reuses the format mask across inputs with one sniff result.

The format pass walked every format slot per call (~1/3 of available()
in-run on fuzzgoat). It depends on the input only through the sniff result,
which most seeds share, so the mask is memoized per fuzzer on (plan, sniff
result, live-format set). Ops and RNG draws must be unchanged.
"""

from fuzzer_tool.core.operator_registry import REGISTRY, OperatorSpec
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.operator_env import make_minimal_fuzzer as make_fuzzer

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
_ZIP = b"PK\x03\x04" + b"a" * 30
_PLAIN = b"plain text"


class _NoTrickle:
    """random() -> 0.99: the bootstrap trickle offers no unseen format."""

    def random(self):
        return 0.99


def _sequence(memo: bool, seed: int) -> list:
    """available() over a seed sequence with repeats; memo cleared when off."""
    f = make_fuzzer(seed=seed)
    f._rng = RandPool(seed)
    inputs = [_PLAIN] * 5 + [_PNG] * 4 + [_PLAIN] * 3 + [_ZIP] * 4 + [_PNG] * 2
    out = []
    for data in inputs:
        if not memo:
            f._format_memo = None
        out.append(REGISTRY.available(f, data))
    out.append(sorted(f._live_formats))
    return out


def test_memo_matches_uncached():
    """Falsification: same ops, same order, same RNG stream as without memo."""
    for seed in range(4):
        want = _sequence(False, seed)
        assert _sequence(False, seed) == want  # control (Rule 46)
        assert _sequence(True, seed) == want


def test_memo_hits_on_same_sniff(monkeypatch):
    """Falsification: inputs sharing a sniff result skip the format pass."""
    calls = {"n": 0}
    real = REGISTRY._sniff_formats

    def counting(*a):
        calls["n"] += 1
        return real(*a)

    monkeypatch.setattr(REGISTRY, "_sniff_formats", counting)
    f = make_fuzzer(seed=1)
    REGISTRY.available(f, _PLAIN)  # creates the live set and memo table
    warm = calls["n"]
    for i in range(5):
        REGISTRY.available(f, _PLAIN + bytes([i]))  # new content, same (empty) sniff
    assert calls["n"] == warm + 1

    REGISTRY.available(f, _PNG)  # different sniff result: recompute
    assert calls["n"] == warm + 2


def test_memo_bounded(monkeypatch):
    """Adversarial: many distinct sniff results cannot grow the table unbounded."""
    from fuzzer_tool.core import operator_registry as reg

    results = iter(frozenset({-1 - i}) for i in range(10**6))
    monkeypatch.setattr(REGISTRY, "_matched", lambda _plan, _data: next(results))
    f = make_fuzzer(seed=6)
    f._rng = _NoTrickle()
    for _ in range(reg._FORMAT_MEMO_MAX * 3):
        REGISTRY.available(f, _PLAIN)
    assert 0 < len(f._format_memo) <= reg._FORMAT_MEMO_MAX


def test_mutable_input_not_memoized():
    """Adversarial: a bytearray edited in place must be re-sniffed."""
    f, g = make_fuzzer(seed=2), make_fuzzer(seed=2)
    f._rng = g._rng = _NoTrickle()
    buf = bytearray(_PLAIN)
    REGISTRY.available(f, buf)
    buf[:] = _PNG
    assert REGISTRY.available(f, buf) == REGISTRY.available(g, bytes(_PNG))
    assert "png_chunk_mutate" in f._live_formats


def test_live_format_change_invalidates():
    """Adversarial: a format made live elsewhere must show up at once."""
    op = "png_chunk_mutate"
    f = make_fuzzer(seed=3)
    f._rng = _NoTrickle()
    assert op not in REGISTRY.available(f, _PLAIN)

    f._live_formats.add(op)  # grown in place: size changes
    assert op in REGISTRY.available(f, _PLAIN)

    f._live_formats = set()  # replaced set: identity changes
    assert op not in REGISTRY.available(f, _PLAIN)


def test_plan_change_invalidates():
    """Adversarial: a registration between calls is offered on the next call."""
    name = "format_memo_probe_op"
    f = make_fuzzer(seed=4)
    REGISTRY.available(f, _PLAIN)
    REGISTRY.register(OperatorSpec(name=name, category="bit", handler_name="_op_x"))
    try:
        assert name in REGISTRY.available(f, _PLAIN)
    finally:
        REGISTRY._ops.pop(name, None)
    assert name not in REGISTRY.available(f, _PLAIN)


def test_foreign_memo_attribute_ignored():
    """Adversarial: a non-memo value in the slot (stub fuzzers) is a miss."""
    f, g = make_fuzzer(seed=5), make_fuzzer(seed=5)
    f._rng = g._rng = None
    f._format_memo = object()
    assert REGISTRY.available(f, _PLAIN) == REGISTRY.available(g, _PLAIN)
