"""REGISTRY.available runs each shared gate once, input-only gates once per input.

Per call it ran 47 predicates (8 dictionary ops, 6 cmplog-pair ops and 5
weizz ops each re-ran the same check) and 22 mutator ``is_available``
calls, 13 of which only look at the input bytes: ~33 us per exec. Fast
path: one call per distinct gate, fanned out to its slots; input-only
gates (``Availability.INPUT``) are cached per input content beside the
format sniff. Context gates stay per call (slow path). Results and RNG
draws must equal the per-op evaluation.
"""

from itertools import compress
from types import SimpleNamespace

import pytest
import xxhash

from fuzzer_tool.core import operator_registry as reg
from fuzzer_tool.core.mutator_interface import Availability, MutationContext, MutatorBase
from fuzzer_tool.core.operator_registry import REGISTRY, OperatorRegistry, OperatorSpec
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.operator_env import make_minimal_fuzzer


def _ref_plan(ops):
    """Oracle plan: one (slot, fn) entry per op, no grouping."""
    mask, preds, formats, mutators = [], [], [], []
    for i, spec in enumerate(ops.values()):
        mask.append(spec.available is None)
        fmt = getattr(spec.available, "format_name", None)
        if spec.available is None:
            continue
        if spec.mutator is not None:
            mutators.append((i, spec.mutator))
        elif fmt is not None:
            formats.append((i, spec.name, reg._FORMAT_SNIFFERS[fmt]))
        else:
            preds.append((i, spec.available))
    return mask, preds, formats, mutators


def _ref_available(ops, sniffed, fuzzer, data):
    """Oracle: the pre-change per-op evaluation, verbatim modulo plan layout."""
    mask, preds, formats, mutators = _ref_plan(ops)
    for i, fn in preds:
        mask[i] = bool(fn(fuzzer, data))

    key = xxhash.xxh3_64_intdigest(data) if data else 0
    matched = sniffed.get(key)
    if matched is None:
        matched = frozenset(i for i, _n, sniff in formats if data and sniff(data))
        sniffed[key] = matched
    live = reg._live_formats(fuzzer)
    pending = []
    for i, name, _s in formats:
        if name in live:
            mask[i] = True
        elif i in matched:
            live.add(name)
            mask[i] = True
        else:
            pending.append(i)

    if mutators:
        ctx = MutationContext.from_fuzzer(fuzzer)
        for i, m in mutators:
            mask[i] = bool(m.is_available(ctx, data))
    if pending:
        keep = reg._trickle(getattr(fuzzer, "_rng", None), len(pending))
        for i, ok in zip(pending, keep, strict=True):
            mask[i] = ok
    return list(compress(tuple(ops), mask))


_INPUTS = [
    b"\x89PNG\r\n\x1a\n" + bytes(64),
    b"\x1f\x8b\x08\x00" + bytes(40),
    b"PK\x03\x04" + bytes(60),
    b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(40),
    b"{[(a)]}" * 80,  # delimiters, >= 512 bytes
    b"",
    bytes(range(256)),
]


def _states():
    """Fuzzer states flipping every config gate kind on and off."""
    yield {}
    yield {"dictionary": [b"tok"], "grammar": object(), "markov_trained": True}
    yield {
        "_cmplog": SimpleNamespace(pairs=[(b"ab", b"cd")], _pair_cmp={1: 2}),
        "_path_solver": object(),
        "weizz_tags": True,
        "enable_regex_bomb": True,
        "op_afl_det": True,
    }


def _fresh(state, seed):
    f = make_minimal_fuzzer(pool=RandPool(seed))
    for k, v in state.items():
        setattr(f, k, v)
    return f


def _run(available, state, seed, rounds=3):
    f = _fresh(state, seed)
    out = [available(f, d) for _ in range(rounds) for d in _INPUTS]
    return out, sorted(reg._live_formats(f)), f._rng.random()


def _ref(state, seed):
    sniffed = {}
    return _run(lambda f, d: _ref_available(REGISTRY._ops, sniffed, f, d), state, seed)


def test_control_oracle_matches_itself():
    """Rule 46: the oracle agrees with a second run of itself."""
    for state in _states():
        assert _ref(state, 7) == _ref(state, 7)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("state_idx", [0, 1, 2])
def test_matches_per_op_evaluation(seed, state_idx):
    """Falsification: same op lists, live formats and RNG position, call
    for call, across config states and inputs (repeated, so cached)."""
    state = list(_states())[state_idx]
    REGISTRY._plan_key = None  # fresh content cache, like the oracle's
    assert _run(REGISTRY.available, state, seed) == _ref(state, seed)


class _CountingMutator(MutatorBase):
    category = "adaptive"

    def __init__(self, name, availability):
        self.name = name
        self.availability = availability
        self.calls = 0

    def is_available(self, context, data):
        self.calls += 1
        return data.startswith(b"Y")

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        return None


def _counting_registry():
    calls = {"shared": 0, "pure": 0}

    def shared(_f, _d):
        calls["shared"] += 1
        return True

    def pure(_f, d):
        calls["pure"] += 1
        return d.startswith(b"Y")

    pure.availability = Availability.INPUT
    r = OperatorRegistry()
    for i in range(4):
        r.register(OperatorSpec(f"s{i}", "byte", "", available=shared))
    r.register(OperatorSpec("p0", "byte", "", available=pure))
    r.register(OperatorSpec("p1", "byte", "", available=pure))
    inp = _CountingMutator("m_in", Availability.INPUT)
    ctx = _CountingMutator("m_ctx", Availability.CONTEXT)
    r.register_mutator(inp)
    r.register_mutator(ctx)
    return r, calls, inp, ctx


def test_shared_gate_runs_once_per_call():
    """Falsification: four ops on one gate cost one call, not four."""
    r, calls, _, _ = _counting_registry()
    f = make_minimal_fuzzer(pool=RandPool(1))
    r.available(f, b"Yes")
    r.available(f, b"No")
    assert calls["shared"] == 2


def test_input_gates_cached_per_content_context_not():
    """Falsification: input-only gates run once per distinct input; context
    gates every call (slow path kept)."""
    r, calls, inp, ctx = _counting_registry()
    f = make_minimal_fuzzer(pool=RandPool(1))
    got = [r.available(f, d) for d in (b"Yes", b"No", b"Yes", b"Yes")]
    assert calls["pure"] == 2 and inp.calls == 2
    assert ctx.calls == 4
    assert got[0] == ["s0", "s1", "s2", "s3", "p0", "p1", "m_in", "m_ctx"]
    assert got[1] == ["s0", "s1", "s2", "s3"]


def test_input_cache_bounded(monkeypatch):
    """Adversarial: unbounded distinct inputs never grow the cache past
    _SNIFF_CACHE_MAX, and results stay correct across the reset."""
    monkeypatch.setattr(reg, "_SNIFF_CACHE_MAX", 3)
    r, _, _, _ = _counting_registry()
    f = make_minimal_fuzzer(pool=RandPool(1))
    for i in range(20):
        d = (b"Y" if i % 2 else b"N") + bytes([i])
        assert ("p0" in r.available(f, d)) is d.startswith(b"Y")
    assert len(r._current_plan().sniffed) <= 3


def test_registry_change_rebuilds_groups():
    """Adversarial: a registration after first use is grouped too."""
    r, calls, _, _ = _counting_registry()
    f = make_minimal_fuzzer(pool=RandPool(1))
    r.available(f, b"Y")
    r.register(OperatorSpec("late", "byte", "", available=lambda _f, d: d == b"Y"))
    assert "late" in r.available(f, b"Y")
