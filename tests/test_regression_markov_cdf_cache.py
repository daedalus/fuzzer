"""MarkovChain samples from cached per-context tables, byte-identically.

generate/sample_byte/codelength re-summed every context's counts per byte
(and generate's fallback ran most_common(1) per byte). The tables are cached
and rebuilt when the chain changes; outputs must match the old code exactly.
"""

import bisect
import math

import pytest

from fuzzer_tool.core.markov import MarkovChain
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 7
_CORPUS = [b"<a href='x'>1</a>", b"<b>22</b><i>3</i>", b"GET / HTTP/1.1\r\n\r\n"]


# ── Oracle: the pre-cache code, verbatim modulo self -> chain ──────────


def _ref_sample(chain, ctx, rng):
    counts = chain.transitions.get(ctx)
    if counts is None or not counts:
        if chain._global_freq:
            return chain._global_freq.most_common(1)[0][0]
        return rng.randint(0, 255)
    total = sum(counts.values()) + chain.smoothing * 256
    r = rng.random() * total
    cumulative = 0.0
    for byte_val, count in counts.items():
        cumulative += count + chain.smoothing
        if r <= cumulative:
            return byte_val
    return rng.randint(0, 255)


def _ref_generate(chain, length, rng):
    result = bytearray()
    ctx = b"\x00" * chain.order
    for _ in range(length):
        result.append(_ref_sample(chain, ctx, rng))
        ctx = bytes(result[max(0, len(result) - chain.order) :])
    return bytes(result)


def _ref_codelength(chain, data):
    if not chain.transitions or not data:
        return len(data) * 8.0
    bits = 0.0
    ctx = b"\x00" * chain.order
    for i, b in enumerate(data):
        counts = chain.transitions.get(ctx)
        if counts is None or not counts:
            bits += 8.0
        else:
            total = sum(counts.values()) + chain.smoothing * 256
            bits += -math.log2((counts.get(b, 0) + chain.smoothing) / total)
        ctx = data[max(0, i + 1 - chain.order) : i + 1]
    return bits


def _chain(order=1, smoothing=0.01):
    chain = MarkovChain(order=order, smoothing=smoothing, rng=RandPool(_SEED))
    for d in _CORPUS:
        chain.train(d)
    return chain


def _gen_pair(chain, length):
    """(cached output, oracle output) from identically seeded pools."""
    chain._rng = RandPool(_SEED)
    got = chain.generate(length)
    return got, _ref_generate(chain, length, RandPool(_SEED))


def test_control_oracle_matches_itself():
    """Rule 46: two oracle runs on one seed agree, else the oracle is broken."""
    chain = _chain()
    assert _ref_generate(chain, 300, RandPool(_SEED)) == _ref_generate(chain, 300, RandPool(_SEED))


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("smoothing", [0.01, 5.0])
def test_generate_matches_oracle(order, smoothing):
    """Falsification. smoothing=5 pushes r past the last cumulative often,
    exercising the randint fallback inside a known context."""
    got, want = _gen_pair(_chain(order, smoothing), 400)
    assert got == want


def test_sample_byte_matches_oracle():
    chain = _chain(order=2)
    ctxs = [*list(chain.transitions)[:10], b"\xff\xfe"]  # last: unseen context
    chain._rng = RandPool(_SEED)
    got = [chain.sample_byte(c) for c in ctxs * 20]
    rng = RandPool(_SEED)
    assert got == [_ref_sample(chain, c, rng) for c in ctxs * 20]


def test_codelength_matches_oracle():
    chain = _chain(order=2)
    data = b"<a href='y'>9</a>\x00\xff unseen"
    assert chain.codelength(data) == pytest.approx(_ref_codelength(chain, data), rel=1e-12)


def test_train_after_sampling_invalidates():
    """Adversarial: a byte learned after the first sample must be reachable."""
    chain = _chain()
    chain.generate(50)
    chain.train(b"\x00Z" * 200)  # context b"\x00" now strongly predicts Z
    got, want = _gen_pair(chain, 200)
    assert got == want


def test_smoothing_change_invalidates():
    """Adversarial: refit_alpha rewrites smoothing in place."""
    chain = _chain()
    chain.generate(50)
    chain.smoothing = 3.0
    got, want = _gen_pair(chain, 200)
    assert got == want


def test_from_dict_replacement_invalidates():
    """Adversarial: loaders swap the transitions object wholesale."""
    chain = _chain()
    chain.generate(50)
    other = MarkovChain(order=1)
    other.train(b"zzzzyyyy" * 30)
    chain.from_dict(other.to_dict())
    got, want = _gen_pair(chain, 200)
    assert got == want


def test_untrained_falls_back_to_rng():
    """Adversarial: no transitions, no global freq -> uniform per byte."""
    chain = MarkovChain(order=1)
    got, want = _gen_pair(chain, 64)
    assert got == want


def test_cdf_lookup_is_bisect_left():
    """The cumulative search must pick the first cum >= r, as the loop did."""
    cum = [1.0, 2.0, 3.0]
    assert [bisect.bisect_left(cum, r) for r in (0.5, 1.0, 1.5, 3.0, 3.1)] == [0, 0, 1, 2, 3]
