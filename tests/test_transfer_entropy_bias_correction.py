"""transfer_entropy()'s shuffle-based bias correction was checked against
the analytic Panzeri-Treves / Miller-Madow closed-form alternative (see
the docstring on TransferEntropy.transfer_entropy) and the closed form was
rejected: it's a first-order approximation only valid when the sample size
comfortably exceeds the joint-context alphabet size, and this module's data
is deep in the opposite regime (near-one distinct context per sample).

These tests pin down both halves of that finding so a future "simplify
this to a closed form" pass doesn't silently reintroduce the false
positives it produces: the shuffle correction must report ~0 for
genuinely independent streams, and the analytic alternative -- kept here
only as a reference implementation, not for production use -- must
demonstrably fail that same check.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from fuzzer_tool.core.transfer_entropy import TransferEntropy


def _analytic_panzeri_treves_te(te: TransferEntropy, source: list[int], target: list[int]) -> float:
    """The rejected alternative: first-order analytic bias correction.

    bias[H(X|Y)] ~= (sum_y m_y - 1) / (2N ln 2), Panzeri & Treves (1996).
    Applied to both conditional entropies TE is a difference of, then
    subtracted the same way the shuffle-based correction is.
    """
    joint_target, joint_both, count_target, count_both = te._build_joints(source, target)
    h_target = te._conditional_entropy_target(joint_target, count_target)
    h_both = te._conditional_entropy_both(joint_both, count_both)
    te_raw = h_target - h_both

    hist_groups: dict = defaultdict(set)
    for y_future, y_hist in joint_target:
        hist_groups[y_hist].add(y_future)
    s1 = sum(len(v) for v in hist_groups.values())

    ctx_groups: dict = defaultdict(set)
    for y_future, y_hist, x_present in joint_both:
        ctx_groups[(y_hist, x_present)].add(y_future)
    s2 = sum(len(v) for v in ctx_groups.values())

    correction = (s2 - s1) / (2 * count_target * math.log(2))
    return max(0.0, te_raw - correction)


def test_shuffle_correction_reports_near_zero_for_independent_streams():
    rng = random.Random(0)
    te = TransferEntropy(history_length=1, n_bins=256)
    n, alphabet = 500, 16
    source = [rng.randrange(alphabet) for _ in range(n)]
    target = [rng.randrange(alphabet) for _ in range(n)]

    result = te.transfer_entropy(source, target, n_surrogates=15)
    assert result < 0.2, f"expected near-zero TE for independent streams, got {result}"


def test_analytic_alternative_undercorrects_on_the_same_independent_streams():
    """Documents the rejection: the closed-form alternative leaves
    substantial spurious TE on data where the true value is 0, which is
    exactly why it isn't used in transfer_entropy()."""
    rng = random.Random(0)
    te = TransferEntropy(history_length=1, n_bins=256)
    n, alphabet = 500, 16
    source = [rng.randrange(alphabet) for _ in range(n)]
    target = [rng.randrange(alphabet) for _ in range(n)]

    result = _analytic_panzeri_treves_te(te, source, target)
    assert result > 1.0, (
        f"expected the analytic alternative to badly undercorrect "
        f"(that's why it was rejected), got {result}"
    )


def test_joint_context_cardinality_is_near_one_per_sample():
    """The reason the analytic formula doesn't apply here: this data sits
    in the extreme-sparsity regime, not the N >> alphabet regime the
    first-order correction assumes."""
    rng = random.Random(0)
    te = TransferEntropy(history_length=1, n_bins=256)
    n, alphabet = 500, 16
    source = [rng.randrange(alphabet) for _ in range(n)]
    target = [rng.randrange(alphabet) for _ in range(n)]

    _, joint_both, _, count_both = te._build_joints(source, target)
    distinct_contexts = len({(h, x) for _, h, x in joint_both})
    assert distinct_contexts / count_both > 0.3
