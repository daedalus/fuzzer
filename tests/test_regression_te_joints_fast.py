"""Transfer entropy rebuilt both joints in a Python loop per surrogate.

16 builds per pair (1 + 15 surrogates), 90 pairs per causal-sector refresh:
~1 s per refresh under --hail-mary. The target-only joint never changes
across surrogates, and the counts are now taken with Counter(zip(...)),
keeping keys and insertion order so the entropies are identical.
"""

from collections import defaultdict

import pytest

from fuzzer_tool.core.analyzers.analyzer_transfer_entropy import TransferEntropy
from fuzzer_tool.core.rand_pool import RandPool

LEN = 300


def _old_joints(k, source, target):
    """The pre-change _build_joints loop, verbatim."""
    n = min(len(source), len(target))
    jt, jb = defaultdict(int), defaultdict(int)
    ct = cb = 0
    for t in range(k, n - 1):
        y_future = target[t + 1]
        y_hist = tuple(target[t - k + 1 : t + 1])
        jt[(y_future, y_hist)] += 1
        ct += 1
        jb[(y_future, y_hist, source[t])] += 1
        cb += 1
    return jt, jb, ct, cb


class _OldTE(TransferEntropy):
    def _build_joints(self, source, target):
        return _old_joints(self.k, source, target)


def _series(rng, alphabet):
    return [rng.randint(0, alphabet - 1) for _ in range(LEN)]


@pytest.mark.parametrize(("k", "alphabet"), [(1, 2), (2, 2), (1, 256)])
def test_regression_te_joints_fast(k, alphabet):
    rng = RandPool(k * 1000 + alphabet)
    src, tgt = _series(rng, alphabet), _series(rng, alphabet)
    src = [s if i % 3 else tgt[i] for i, s in enumerate(src)]  # some real coupling
    old, new = _OldTE(history_length=k), TransferEntropy(history_length=k)
    assert new.transfer_entropy(src, tgt) == old.transfer_entropy(src, tgt)


def test_old_matches_itself():
    """Control (Hard Rule 46): the oracle is deterministic."""
    rng = RandPool(5)
    src, tgt = _series(rng, 2), _series(rng, 2)
    assert _OldTE().transfer_entropy(src, tgt) == _OldTE().transfer_entropy(src, tgt)


def test_short_series_is_zero():
    """Adversarial: below k + 2 samples there is nothing to estimate."""
    assert TransferEntropy(history_length=2).transfer_entropy([1, 0, 1], [0, 1, 0]) == 0.0


def test_joints_match_old_loop():
    rng = RandPool(9)
    src, tgt = _series(rng, 4), _series(rng, 4)
    new = TransferEntropy(history_length=2)._build_joints(src, tgt)
    old = _old_joints(2, src, tgt)
    assert [list(new[0].items()), list(new[1].items()), new[2], new[3]] == [
        list(old[0].items()),
        list(old[1].items()),
        old[2],
        old[3],
    ]
