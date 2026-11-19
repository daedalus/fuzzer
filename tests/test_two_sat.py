"""2-SAT solver, bit-vector clause encoders, and the path-negation fast path.

Brute force is the oracle: every formula here has at most ``MAX_BRUTE_VARS``
variables, so all ``2**n`` assignments are enumerated independently of the
implication-graph code under test.
"""

import itertools

import pytest

from fuzzer_tool.core.path_constraints import BranchRecord, PathConstraintSolver
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.two_sat import (
    eq_clauses,
    ge_clauses,
    gt_clauses,
    le_clauses,
    lt_clauses,
    ne_clause,
    solve_2sat,
    solve_one_wide,
)

MAX_BRUTE_VARS = 8
N_FORMULAS = 400
SEED = 20261119
ENC_BITS = 5  # exhaustive encoder check: 32 constants x 32 values


# ── oracle ──────────────────────────────────────────────────────────────


def _holds(assign, clauses):
    """True when ``assign`` (index v-1) satisfies every clause."""
    return all(any(assign[abs(lit) - 1] == (lit > 0) for lit in c) for c in clauses)


def _brute(n, clauses):
    """First satisfying assignment in lexicographic order, or None."""
    for bits in itertools.product((False, True), repeat=n):
        if _holds(bits, clauses):
            return list(bits)
    return None


def _random_formula(rng, wide=False):
    n = rng.randint(1, MAX_BRUTE_VARS)
    m = rng.randint(1, 3 * n)
    clauses = []
    for _ in range(m):
        width = rng.randint(1, 2)
        clauses.append(tuple(rng.randint(1, n) * rng.choice((1, -1)) for _ in range(width)))
    if not wide:
        return n, clauses, None
    k = rng.randint(3, n + 3)
    return n, clauses, tuple(rng.randint(1, n) * rng.choice((1, -1)) for _ in range(k))


def _disagreements(solver, formulas):
    """Formulas where ``solver`` and brute force disagree on sat/model validity."""
    bad = []
    for n, clauses in formulas:
        got = solver(n, clauses)
        want = _brute(n, clauses)
        if (got is None) != (want is None):
            bad.append((n, clauses))
            continue
        if got is not None and not _holds(got, clauses):
            bad.append((n, clauses))
    return bad


def _formulas():
    rng = RandPool(seed=SEED)
    return [_random_formula(rng)[:2] for _ in range(N_FORMULAS)]


# ── solve_2sat ──────────────────────────────────────────────────────────


def test_unique_model_is_found():
    # (x1|x2)(~x1|x2)(x1|~x2): only x1 = x2 = True satisfies all three.
    assert solve_2sat(2, [(1, 2), (-1, 2), (1, -2)]) == [True, True]


def test_all_four_sign_pairs_are_unsat():
    assert solve_2sat(2, [(1, 2), (1, -2), (-1, 2), (-1, -2)]) is None


def test_unit_conflict_is_unsat():
    assert solve_2sat(1, [(1,), (-1,)]) is None


def test_empty_clause_is_unsat():
    assert solve_2sat(3, [(1, 2), ()]) is None


def test_no_clauses_is_sat():
    assert solve_2sat(3, []) is not None


def test_matches_brute_force():
    formulas = _formulas()
    sat = sum(_brute(n, c) is not None for n, c in formulas)
    # Both outcomes must be exercised or the comparison proves nothing.
    assert 0 < sat < len(formulas)
    assert _disagreements(solve_2sat, formulas) == []


def test_control_oracle_against_itself():
    """Hard Rule 46: the harness passes brute vs brute, and flags a wrong solver."""
    formulas = _formulas()
    assert _disagreements(_brute, formulas) == []

    def all_true(n, _clauses):
        return [True] * n

    assert _disagreements(all_true, formulas) != []


def test_falsify_odd_cycle_through_long_chain():
    # x1 -> x2 -> ... -> xn -> ~x1 and ~x1 -> ... forces a contradiction only
    # through the whole chain: x1 must be False, then (x1 | xn) forces xn,
    # and xn -> ~x1 is fine, but (x1 | ~xn) closes the cycle.
    n = 50
    chain = [(-i, i + 1) for i in range(1, n)]
    assert solve_2sat(n, chain + [(-n, -1), (1, n), (1, -n)]) is None
    assert solve_2sat(n, chain + [(-n, -1), (1, n)]) is not None


def test_adversarial_deep_chain_no_recursion():
    """A 200k-long implication chain would overflow a recursive Tarjan."""
    n = 200_000
    clauses = [(-i, i + 1) for i in range(1, n)] + [(1,)]
    assert solve_2sat(n, clauses) == [True] * n


@pytest.mark.parametrize(
    "clauses",
    [[(0, 1)], [(1, 4)], [(-4,)], [(1, 2, 3)]],
)
def test_adversarial_bad_input_rejected(clauses):
    with pytest.raises(ValueError):
        solve_2sat(3, clauses)


# ── solve_one_wide ──────────────────────────────────────────────────────


def test_one_wide_matches_brute_force():
    rng = RandPool(seed=SEED + 1)
    triples = [_random_formula(rng, wide=True) for _ in range(N_FORMULAS)]
    formulas = [(n, clauses + [wide]) for n, clauses, wide in triples]
    sat = sum(_brute(n, c) is not None for n, c in formulas)
    assert 0 < sat < len(formulas)

    def via_split(n, all_clauses):
        return solve_one_wide(n, all_clauses[:-1], all_clauses[-1])

    assert _disagreements(via_split, formulas) == []


def test_one_wide_needs_a_split():
    # Narrow part alone prefers nothing; wide clause (x1|x2|x3) with ~x1, ~x2
    # leaves x3 as the only way out.
    got = solve_one_wide(3, [(-1,), (-2,)], (1, 2, 3))
    assert got == [False, False, True]


def test_one_wide_empty_is_unsat():
    assert solve_one_wide(2, [(1, 2)], ()) is None


# ── bit-vector encoders ─────────────────────────────────────────────────

_BITS = list(range(1, ENC_BITS + 1))  # var ids, LSB first
_LIMIT = 1 << ENC_BITS


def _value_assign(x):
    return [bool(x >> i & 1) for i in range(ENC_BITS)]


@pytest.mark.parametrize(
    ("encode", "relation"),
    [
        (ge_clauses, lambda x, c: x >= c),
        (le_clauses, lambda x, c: x <= c),
        (gt_clauses, lambda x, c: x > c),
        (lt_clauses, lambda x, c: x < c),
        (eq_clauses, lambda x, c: x == c),
        (lambda b, c: [ne_clause(b, c)], lambda x, c: x != c),
    ],
)
def test_encoder_exhaustive(encode, relation):
    for c in range(_LIMIT):
        clauses = encode(_BITS, c)
        for x in range(_LIMIT):
            assert _holds(_value_assign(x), clauses) == relation(x, c), (c, x)


def test_ge_high_bit_is_a_unit():
    # x >= 0b10000 on 5 bits: only the top bit matters.
    assert ge_clauses(_BITS, 1 << (ENC_BITS - 1)) == [(_BITS[-1],)]


def test_encoder_out_of_range_constants():
    assert gt_clauses(_BITS, _LIMIT - 1) == [()]  # nothing exceeds the max
    assert lt_clauses(_BITS, 0) == [()]  # nothing is below zero
    assert ge_clauses(_BITS, _LIMIT) == [()]
    assert le_clauses(_BITS, _LIMIT - 1) == []  # always true


# ── path-negation fast path ─────────────────────────────────────────────

_BYTE = 0x50  # 0x3f < _BYTE < 0x80


def _rec1(op_b, result, pc):
    return BranchRecord(bytes([_BYTE]), bytes([op_b]), result, 1, pc)


def _data():
    return b"HEAD" + bytes([_BYTE]) + b"TAIL"


def test_two_cnf_overlap_skips_z3():
    # Flip x < 0x80 -> x >= 0x80 (unit x7); keep x > 0x3f -> (x6 | x7).
    flip, keep = _rec1(0x80, -1, 1), _rec1(0x3F, 1, 2)
    solver = PathConstraintSolver()
    out = solver.negate(flip, _data(), others=[flip, keep])

    assert out is not None
    assert out[4] >= 0x80 and out[4] > 0x3F
    assert out[:4] == b"HEAD" and out[5:] == b"TAIL"
    assert solver.stats()["two_sat_solves"] == 1
    assert solver.stats()["z3_solves"] == 0


def test_two_cnf_contradiction_is_unsat():
    # Same comparison at two sites: flipping one breaks the other.
    flip, keep = _rec1(0x80, -1, 1), _rec1(0x80, -1, 2)
    solver = PathConstraintSolver()
    assert solver.negate(flip, _data(), others=[flip, keep]) is None
    assert solver.stats()["unsat"] == 1


def test_falsify_wide_clause_falls_through():
    # Keep x > 0x08 -> (x3 | x4 | x5 | x6 | x7): not 2-CNF, 2-SAT must not answer.
    flip, keep = _rec1(0x80, -1, 1), _rec1(0x08, 1, 2)
    solver = PathConstraintSolver()
    solver.negate(flip, _data(), others=[flip, keep])
    assert solver.stats()["two_sat_solves"] == 0


_EDGE_CONSTS = (0x00, 0x01, 0x3F, 0x40, 0x7F, 0x80, 0xBF, 0xC0, 0xFE, 0xFF)
N_QUERIES = 300


def _queries():
    """Random one-byte flip + keep queries over constants with 2-CNF encodings."""
    rng = RandPool(seed=SEED + 2)
    out = []
    for _ in range(N_QUERIES):
        byte = rng.randint(0, 0xFF)
        recs = []
        for pc in range(rng.randint(2, 4)):
            c = rng.choice(_EDGE_CONSTS)
            result = (byte > c) - (byte < c)
            recs.append(BranchRecord(bytes([byte]), bytes([c]), result, 1, pc))
        out.append((bytes([byte]), recs))  # alone: find() cannot land elsewhere
    return out


def _outcomes(make_solver):
    """Per query: None when unsat, else True (any model is fine)."""
    res = []
    for data, recs in _queries():
        out = make_solver().negate(recs[0], data, others=recs)
        res.append(out is not None)
    return res


def _z3_only():
    solver = PathConstraintSolver()
    solver._bit_cnf = lambda *_args: None
    return solver


def test_fast_path_agrees_with_z3():
    pytest.importorskip("z3")
    # Hard Rule 46 control: z3 against itself first.
    assert _outcomes(_z3_only) == _outcomes(_z3_only)

    fired = 0
    for data, recs in _queries():
        solver = PathConstraintSolver()
        got = solver.negate(recs[0], data, others=recs)
        fired += solver.stats()["two_sat_solves"]
        want = _z3_only().negate(recs[0], data, others=recs)
        assert (got is None) == (want is None)
        if got is not None:
            assert _flips_and_keeps(got[0], data[0], recs)
    assert fired > 0


def _flips_and_keeps(x, original, recs):
    """x inverts recs[0], preserves the rest, and differs from original."""
    first, *rest = recs
    c0 = first.op_b[0]
    flipped = {-1: x >= c0, 1: x <= c0, 0: x != c0}[first.result]
    kept = all((x > r.op_b[0]) - (x < r.op_b[0]) == r.result for r in rest if r.key != first.key)
    return flipped and kept and x != original
