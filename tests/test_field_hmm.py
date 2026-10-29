"""Viterbi field segmentation model (core/field_hmm.py).

Every expectation derives from an independent oracle (brute-force path
enumeration, hand-set parameters) or arithmetic; no RNG in the model.
"""

import itertools
import math

import numpy as np
import pytest

from fuzzer_tool.core.field_hmm import (
    DECODE_MAX,
    FieldHMM,
    FieldKind,
    _viterbi,
    _viterbi_one,
    _viterbi_py,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import StateStore

ZEROS_THEN_FF = b"\x00" * 40 + b"\xff" * 40


def _log_dirichlet(rng: RandPool, rows: int, cols: int) -> np.ndarray:
    return np.log(np.array([rng.dirichlet([1.0] * cols) for _ in range(rows)]))


def _score(path, obs, log_pi, log_A, log_B) -> float:
    s = log_pi[path[0]] + log_B[path[0], obs[0]]
    for t in range(1, len(obs)):
        s += log_A[path[t - 1], path[t]] + log_B[path[t], obs[t]]
    return float(s)


def _brute(obs, log_pi, log_A, log_B) -> float:
    # Oracle: best score over all K^n paths (ties make the path itself ambiguous).
    paths = itertools.product(range(log_pi.size), repeat=len(obs))
    return max(_score(p, obs, log_pi, log_A, log_B) for p in paths)


def _two_state() -> FieldHMM:
    # Sticky chain; state 0 emits 0x00, state 1 emits 0xFF.
    log_B = np.full((2, 256), math.log(1e-6))
    log_B[0, 0x00] = 0.0
    log_B[1, 0xFF] = 0.0
    hmm = FieldHMM(n_states=2)
    hmm.load_state(
        {
            "log_pi": np.log([0.5, 0.5]),
            "log_A": np.log([[0.99, 0.01], [0.01, 0.99]]),
            "log_B": log_B,
            "kinds": [FieldKind.UNKNOWN.value] * 2,
        }
    )
    return hmm


def _random_params(seed: int, k: int, m: int):
    rng = RandPool(seed=seed)
    log_pi = np.log(np.array(rng.dirichlet([1.0] * k)))
    return log_pi, _log_dirichlet(rng, k, k), _log_dirichlet(rng, k, m)


def _decode_one(obs, log_pi, log_A, log_B) -> tuple[int, ...]:
    arr = np.array([obs], np.uint8)
    path = _viterbi(arr, np.array([len(obs)]), log_pi, log_A, log_B)
    return tuple(int(s) for s in path[0])


# -- falsification ----------------------------------------------------------


def test_control_brute_force_agrees_with_itself():
    """Hard Rule 46: the oracle must match a second run of itself first."""
    rng = RandPool(seed=11)
    for case in range(20):
        log_pi, log_A, log_B = _random_params(case, 3, 4)
        obs = rng.randint_list(0, 3, 6)
        assert _brute(obs, log_pi, log_A, log_B) == _brute(obs, log_pi, log_A, log_B)


def test_falsify_viterbi_matches_brute_force():
    rng = RandPool(seed=7)
    for case in range(100):
        log_pi, log_A, log_B = _random_params(case, 3, 4)
        obs = rng.randint_list(0, 3, 6)
        path = _decode_one(obs, log_pi, log_A, log_B)
        got = _score(path, obs, log_pi, log_A, log_B)
        assert got == pytest.approx(_brute(obs, log_pi, log_A, log_B), abs=1e-12)


def test_falsify_two_segment_boundary_exact():
    hmm = _two_state()
    assert hmm.spans(ZEROS_THEN_FF) == [(0, 40, 0), (40, 80, 1)]


def test_batch_padding_matches_single_decode():
    """Rows of different length decode as if alone (padding is inert)."""
    log_pi, log_A, log_B = _random_params(3, 4, 256)
    rows = [bytes(range(i, i + n)) for i, n in ((0, 9), (50, 1), (100, 30))]
    width = max(len(r) for r in rows)
    obs = np.zeros((len(rows), width), np.uint8)
    for b, r in enumerate(rows):
        obs[b, : len(r)] = np.frombuffer(r, np.uint8)
    lens = np.array([len(r) for r in rows])

    paths = _viterbi(obs, lens, log_pi, log_A, log_B)

    for b, r in enumerate(rows):
        got = tuple(int(s) for s in paths[b, : len(r)])
        assert got == _decode_one(list(r), log_pi, log_A, log_B)


def test_single_row_decoders_match_batched():
    """Python (small K) and numpy single-row decoders tie-break like _viterbi."""
    rng = RandPool(seed=5)
    for k in (1, 2, 4, 5, 8):
        log_pi, log_A, log_B = _random_params(k, k, 256)
        log_A[0, -1] = -math.inf
        obs = np.array(rng.randint_list(0, 255, 200), np.uint8)
        want = _viterbi(obs[None, :], np.array([obs.size]), log_pi, log_A, log_B)[0]

        assert np.array_equal(_viterbi_one(obs, log_pi, log_A, log_B), want)
        assert np.array_equal(_viterbi_py(obs, log_pi, log_A, log_B), want)


# -- training ---------------------------------------------------------------


def test_fit_objective_non_decreasing():
    seeds = [ZEROS_THEN_FF, b"ABCD" * 20 + bytes(range(64)), b"\x00\x01" * 50]
    history = FieldHMM(n_states=4).fit(seeds, iters=6)
    assert len(history) >= 2
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(history))


def test_fit_separates_two_regions():
    hmm = FieldHMM(n_states=2)
    hmm.fit([ZEROS_THEN_FF] * 4)
    path = hmm.decode(ZEROS_THEN_FF)
    assert len(set(path[:40].tolist())) == 1
    assert len(set(path[40:].tolist())) == 1
    assert path[0] != path[40]


def test_fit_is_deterministic():
    seeds = [ZEROS_THEN_FF, b"hello world" * 7]
    a, b = FieldHMM(n_states=3), FieldHMM(n_states=3)
    assert a.fit(seeds) == b.fit(seeds)
    assert a.spans(seeds[1]) == b.spans(seeds[1])


def test_fit_empty_is_noop():
    hmm = FieldHMM(n_states=3)
    before = hmm.to_state()
    assert hmm.fit([]) == []
    assert hmm.fit([b""]) == []
    after = hmm.to_state()
    assert all(np.array_equal(before[k], after[k]) for k in ("log_pi", "log_A", "log_B"))


# -- adversarial ------------------------------------------------------------


def test_adversarial_empty_and_one_byte():
    hmm = _two_state()
    assert hmm.decode(b"").size == 0
    assert hmm.spans(b"") == []
    assert hmm.spans(b"\xff") == [(0, 1, 1)]


def test_adversarial_input_beyond_cap_truncated():
    hmm = _two_state()
    spans = hmm.spans(b"\x00" * (DECODE_MAX + 100))
    assert spans == [(0, DECODE_MAX, 0)]


def test_adversarial_forbidden_transition_never_taken():
    # 0 -> 1 forbidden: the 0x00..0xFF input must stay in one state.
    hmm = _two_state()
    state = hmm.to_state()
    state["log_A"] = np.array([[0.0, -math.inf], [math.log(0.5), math.log(0.5)]])
    state["log_pi"] = np.array([0.0, -math.inf])
    hmm.load_state(state)
    path = hmm.decode(ZEROS_THEN_FF)
    assert set(path.tolist()) == {0}


def test_adversarial_state_count_bounds():
    one = FieldHMM(n_states=1)
    one.fit([ZEROS_THEN_FF])
    assert one.spans(ZEROS_THEN_FF) == [(0, 80, 0)]

    wide = FieldHMM(n_states=256)
    wide.fit([bytes(range(256))])
    assert wide.decode(bytes(range(256))).max() <= 255

    for bad in (0, 257):
        with pytest.raises(ValueError):
            FieldHMM(n_states=bad)


def test_adversarial_constant_input():
    hmm = FieldHMM(n_states=4)
    hmm.fit([b"\x41" * 100])
    assert len(hmm.spans(b"\x41" * 100)) == 1


def test_spans_are_run_length_of_decode():
    hmm = FieldHMM(n_states=4)
    data = b"\x00" * 10 + b"ABC" * 10 + b"\xff" * 5 + b"\x00" * 7
    hmm.fit([data])
    path = hmm.decode(data).tolist()

    # Independent RLE.
    expect, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            expect.append((start, i, path[start]))
            start = i
    assert hmm.spans(data) == expect


# -- naming -----------------------------------------------------------------


def test_name_states_majority_vote():
    hmm = _two_state()
    kinds = np.zeros(80, np.uint8)
    kinds[:30] = FieldKind.MAGIC.value
    kinds[40:45] = FieldKind.LENGTH.value
    hmm.name_states([(ZEROS_THEN_FF, kinds)])

    assert hmm.kind(0) is FieldKind.MAGIC
    # 5 votes: below the minimum, stays unnamed.
    assert hmm.kind(1) is FieldKind.UNKNOWN


def test_name_states_split_vote_stays_unknown():
    hmm = _two_state()
    kinds = np.zeros(80, np.uint8)
    kinds[:20] = FieldKind.MAGIC.value
    kinds[20:40] = FieldKind.CHECKSUM.value
    hmm.name_states([(ZEROS_THEN_FF, kinds)])
    assert hmm.kind(0) is FieldKind.UNKNOWN


# -- persistence ------------------------------------------------------------


def test_state_round_trips_through_state_store(tmp_path):
    hmm = FieldHMM(n_states=3)
    hmm.fit([ZEROS_THEN_FF, b"xyz" * 30])
    store = StateStore(tmp_path)
    store.set("field_hmm", hmm.to_state())
    assert store.save()

    again = FieldHMM(n_states=3)
    again.load_state(StateStore(tmp_path).get("field_hmm"))
    assert again.spans(b"xyz" * 30 + ZEROS_THEN_FF) == hmm.spans(b"xyz" * 30 + ZEROS_THEN_FF)


def test_load_state_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        FieldHMM(n_states=3).load_state(FieldHMM(n_states=4).to_state())
