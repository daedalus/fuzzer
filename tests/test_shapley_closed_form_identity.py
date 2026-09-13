"""The per-edge frequency-weighted Shapley game has a closed form.

``shapley_values`` used to estimate the Shapley value by averaging marginal
contributions over ``n_samples`` random permutations of the operators,
costing O(n_samples * n_ops * edges). But every edge's credit split is
fixed ahead of time (``_edge_op_count``), and an operator only earns an
edge's credit in a given permutation if it is the first of that edge's
co-occurring operators to appear. By symmetry, each of the ``k`` co-occurring
operators is equally likely to be first in a uniformly random permutation,
so the expected — and therefore exact — contribution from that edge is
``credit(e, op) / k``, independent of everything else in the permutation.

This test keeps the original permutation sampler around as a reference
implementation and checks that it converges to the closed form as
``n_samples`` grows, and that the closed form is reproducible (calling it
twice gives bit-identical results, unlike the old sampler).
"""

import random

from fuzzer_tool.core.shapley import ShapleyAttribution


def _monte_carlo_reference(sa: ShapleyAttribution, operators: list[str], n_samples: int) -> dict:
    """The permutation-sampling algorithm this replaced, written out verbatim."""

    def op_credit(edge: int, op: str) -> float:
        op_counts = sa._edge_op_count.get(edge)
        if not op_counts:
            return 0.0
        count = op_counts.get(op)
        if not count:
            return 0.0
        total = sum(op_counts.values())
        return count / total if total else 0.0

    n_ops = len(operators)
    shapley = {op: 0.0 for op in operators}
    for _ in range(n_samples):
        perm = operators[:]
        random.shuffle(perm)
        prefix_edges: set[int] = set()
        for op in perm:
            marginal = 0.0
            for edge in sa._operator_edges.get(op, set()):
                if edge not in prefix_edges:
                    marginal += op_credit(edge, op)
            shapley[op] += marginal
            prefix_edges.update(sa._operator_edges.get(op, set()))
    total = sum(shapley.values())
    if total > 0:
        return {op: v / total for op, v in shapley.items()}
    return {op: 1.0 / n_ops for op in operators}


def _build_synthetic(seed: int) -> tuple[ShapleyAttribution, list[str]]:
    rng = random.Random(seed)
    sa = ShapleyAttribution(window_size=5000)
    pool = [f"op{i}" for i in range(8)]
    for _i in range(1500):
        k = rng.randint(1, 4)
        ops = set(rng.sample(pool, k))
        n_new = rng.randint(0, 3)
        edges = set(rng.sample(range(300), n_new)) if n_new else None
        sa.record(ops, n_new, edges)
    operators = sorted({op for ops, _ in sa._outcomes for op in ops})
    return sa, operators


def test_monte_carlo_converges_to_closed_form():
    sa, operators = _build_synthetic(seed=42)
    closed_form = sa.shapley_values(operators)

    random.seed(0)
    diffs = []
    for n_samples in (100, 1_000, 20_000):
        mc = _monte_carlo_reference(sa, operators, n_samples)
        diffs.append(max(abs(mc[op] - closed_form[op]) for op in operators))

    # Each successive 10x/20x increase in samples should shrink the gap
    # to the closed form (Monte-Carlo error ~ 1/sqrt(n_samples)).
    assert diffs[0] > diffs[1] > diffs[2]
    assert diffs[2] < 0.01


def test_closed_form_is_deterministic():
    sa, operators = _build_synthetic(seed=7)
    first = sa.shapley_values(operators)
    second = sa.shapley_values(operators)
    assert first == second


def test_closed_form_matches_symmetric_case_exactly():
    """Two operators, always co-occurring on identical edges, split credit exactly."""
    sa = ShapleyAttribution(window_size=100)
    for i in range(20):
        sa.record({"a", "b"}, new_edges=1, edge_indices={i})
    sv = sa.shapley_values(["a", "b"])
    assert abs(sv["a"] - sv["b"]) < 1e-12
    assert abs(sv["a"] - 0.5) < 1e-12
