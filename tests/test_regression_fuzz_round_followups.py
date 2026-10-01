"""Regressions for bugs carried into FuzzRound from the old fuzz_one.

1. ``runner._lib.__cmplog_reset()`` / ``__tracecmp_flush()`` written inside a
   class body are name-mangled (``_FuzzRound__cmplog_reset``), so the shim
   call never ran and the swallowed AttributeError hid it.
2. ``_smt_sample`` reset ``_smt_found`` per sampled pair: a later unsolved
   pair erased an earlier hit, so ``smt_solver`` lost credit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

_RESET = "__cmplog_reset"
_FLUSH = "__tracecmp_flush"


def _lib_with(*symbols: str) -> SimpleNamespace:
    """Stand-in CDLL exposing the unmangled shim symbols."""
    lib = SimpleNamespace()
    for name in symbols:
        setattr(lib, name, MagicMock(name=name))
    return lib


def _runner(lib, direct_lite: bool = True) -> SimpleNamespace:
    return SimpleNamespace(direct_lite=direct_lite, _lib=lib)


# ── 1. Name mangling ─────────────────────────────────────────────────


def _rewind(f, collected: bool) -> None:
    rnd = FuzzRound(f, b"seed")
    rnd._collect_now = collected
    rnd._rewind_shim()


def test_regression_rewind_shim_calls_cmplog_reset():
    lib = _lib_with(_RESET)
    f = SimpleNamespace(_inprocess_runner=_runner(lib))

    _rewind(f, collected=True)

    getattr(lib, _RESET).assert_called_once_with()


def test_rewind_shim_keeps_uncollected_records():
    """Adversarial: the shim truncates, so it must only run after Python read.

    collect_tokens() runs every 1/5/20 rounds; resetting on a skipped round
    would discard the records it has not read yet.
    """
    lib = _lib_with(_RESET)
    f = SimpleNamespace(_inprocess_runner=_runner(lib))

    _rewind(f, collected=False)

    getattr(lib, _RESET).assert_not_called()


def test_regression_reset_cmplog_calls_tracecmp_flush():
    lib = _lib_with(_FLUSH)
    f = SimpleNamespace(_cmplog=MagicMock(), _inprocess_runner=_runner(lib))

    Fuzzer._reset_cmplog(f)

    getattr(lib, _FLUSH).assert_called_once_with()


def test_rewind_shim_skips_without_direct_lite():
    """Falsification: subprocess mode must not touch the shim."""
    lib = _lib_with(_RESET)
    f = SimpleNamespace(_inprocess_runner=_runner(lib, direct_lite=False))

    _rewind(f, collected=True)

    getattr(lib, _RESET).assert_not_called()


def test_rewind_shim_tolerates_missing_symbol():
    """Adversarial: a target built without cmplog exports no reset symbol."""
    f = SimpleNamespace(_inprocess_runner=_runner(_lib_with()))

    _rewind(f, collected=True)  # must not raise


# ── 2. SMT hit survives later misses ─────────────────────────────────


class _NoShuffle:
    """Keeps the sample in pair order so the scripted solver lines up."""

    def shuffle(self, seq):
        return None


def _smt_round(mutated: bytes, pairs, results) -> FuzzRound:
    solver = MagicMock(queries_attempted=0, queries_solved=0)
    solver.solve_cmplog_pair.side_effect = list(results)
    cmplog = MagicMock(pairs=list(pairs))
    f = SimpleNamespace(_smt_solver=solver, _cmplog=cmplog, _rng=_NoShuffle())
    rnd = FuzzRound(f, b"seed")
    rnd._mutated = mutated
    return rnd


def test_regression_smt_hit_survives_later_miss():
    hit = (b"AB", b"QQ")  # b"AB" occurs in the mutant
    miss = (b"CD", b"RR")
    solved = {"solved_bytes": b"ZZ"}
    rnd = _smt_round(b"xxABxx", [hit, miss], [solved, None])
    matches: list = []

    rnd._smt_sample(matches, set())

    assert rnd._smt_found is True
    assert matches == [(b"xxABxx".index(b"AB"), b"AB", b"ZZ")]


def test_smt_found_false_when_nothing_matches():
    """Falsification: no solved operand in the mutant → no credit."""
    rnd = _smt_round(b"xxxxxx", [(b"AB", b"QQ")], [{"solved_bytes": b"ZZ"}])
    matches: list = []

    rnd._smt_sample(matches, set())

    assert rnd._smt_found is False
    assert matches == []


def test_smt_scans_op_b_after_an_earlier_hit():
    """Adversarial: an earlier hit must not stop op_b being scanned later."""
    first = (b"AB", b"QQ")
    second = (b"NO", b"CD")  # only op_b occurs
    solved = {"solved_bytes": b"ZZ"}
    mutated = b"ABxxCD"
    rnd = _smt_round(mutated, [first, second], [solved, solved])
    matches: list = []

    rnd._smt_sample(matches, set())

    assert (mutated.index(b"CD"), b"CD", b"ZZ") in matches
