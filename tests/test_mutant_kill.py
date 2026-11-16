"""services/mutant_kill: corpus -> kill score (docs/action_plan_mutant_kill_metric.md P3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.adapters.mutant_build import (
    MutantSet,
    Status,
    TargetSpec,
    asan_available,
    replay,
    write_pack,
)
from fuzzer_tool.core.ir_mutants import MutOp
from fuzzer_tool.services import mutant_kill
from fuzzer_tool.services.mutant_kill import Confirm, KillScore, KillScorer
from tests.support.mutant_toy import TOY_LIB, TOY_WRAP

pytestmark = pytest.mark.skipif(not asan_available(), reason="needs clang + ASAN runtime")

_TIMEOUT_MS = 300
# Clean on the original; "abcd" overflows the n<=4 mutant, "abcdefg" and
# "ok" exercise the else branch and the 'H' compare.
_INPUTS = [b"abc", b"abcd", b"abcdefg", b"ok", b"C"]


@pytest.fixture(scope="module")
def mset(tmp_path_factory) -> MutantSet:
    root = tmp_path_factory.mktemp("kill")
    (root / "toy.c").write_text(TOY_LIB)
    (root / "toy_wrap.c").write_text(TOY_WRAP)
    return MutantSet.build(
        TargetSpec(root / "toy.c", root / "toy_wrap.c"), root / "set", cap=10_000, seed=1, jobs=4
    )


def _scorer(mset: MutantSet, confirm: Confirm = Confirm.RERUN) -> KillScorer:
    return KillScorer(mset, jobs=4, timeout_ms=_TIMEOUT_MS, confirm=confirm)


def _oracle(mset: MutantSet, inputs: list[bytes], tmp_path: Path) -> tuple[set[int], set[int]]:
    """Kill / hang sets from the definition, via unmasked full replays.

    killed: some input runs CLEAN on the original and CRASH on the mutant.
    hung:   not killed, and some original-CLEAN input HANGs the mutant.
    """
    pack = tmp_path / "oracle.pack"
    write_pack(inputs, pack)
    clean = [s == Status.CLEAN for s in replay(mset.original, pack, None, _TIMEOUT_MS)]

    killed, hung = set(), set()
    for k, binary in enumerate(mset.mutants):
        got = replay(binary, pack, None, _TIMEOUT_MS)
        eligible = [s for s, ok in zip(got, clean, strict=True) if ok]
        if Status.CRASH in eligible:
            killed.add(k)
        elif Status.HANG in eligible:
            hung.add(k)
    return killed, hung


def _site_index(mset: MutantSet, op: MutOp, needle: str) -> int:
    lines = mset.base_ir.split("\n")
    (k,) = [k for k, s in enumerate(mset.sites) if s.op == op and needle in lines[s.line]]
    return k


def test_matches_definition(mset, tmp_path) -> None:
    killed, hung = _oracle(mset, _INPUTS, tmp_path)
    got = _scorer(mset).score_inputs(_INPUTS)
    assert (set(got.killed), set(got.hung)) == (killed, hung)
    assert (got.kills, got.hangs) == (len(killed), len(hung))
    assert got.inputs == len(_INPUTS)
    assert got.clean_inputs == len(_INPUTS) - 1  # "C" crashes the original


def test_boundary_mutant_killed(mset) -> None:
    k = _site_index(mset, MutOp.ICMP_BOUNDARY, "icmp ult i64")
    assert k in _scorer(mset).score_inputs([b"abcd"]).killed
    assert k not in _scorer(mset).score_inputs([b"abc", b"abcdefg"]).killed


def test_deterministic(mset) -> None:
    """Rule 46 control: one corpus scored twice gives one answer."""
    assert _scorer(mset).score_inputs(_INPUTS) == _scorer(mset).score_inputs(_INPUTS)


def test_original_crash_is_no_evidence(mset) -> None:
    got = _scorer(mset).score_inputs([b"C", b"C"])
    assert got == KillScore(
        kills=0, hangs=0, flaky=0, inputs=2, clean_inputs=0, killed=frozenset(), hung=frozenset()
    )


def test_hang_is_not_a_kill(mset) -> None:
    # 'H' == 72: the eq -> ne mutant spins on every input not starting with 'H'.
    k = _site_index(mset, MutOp.ICMP_EQ_NE, ", 72")
    got = _scorer(mset).score_inputs([b"ok"])
    assert k in got.hung
    assert k not in got.killed


def test_empty_corpus(mset) -> None:
    got = _scorer(mset).score_inputs([])
    assert got == KillScore(
        kills=0, hangs=0, flaky=0, inputs=0, clean_inputs=0, killed=frozenset(), hung=frozenset()
    )


def test_corpus_dir_equals_inputs(mset, tmp_path) -> None:
    seeds = tmp_path / "corpus" / "seeds"
    seeds.mkdir(parents=True)
    for i, data in enumerate(_INPUTS):
        (seeds / f"s{i}").write_bytes(data)

    from_dir = _scorer(mset).score(tmp_path / "corpus")
    from_list = _scorer(mset).score_inputs(_INPUTS)
    assert (from_dir.killed, from_dir.hung, from_dir.clean_inputs) == (
        from_list.killed,
        from_list.hung,
        from_list.clean_inputs,
    )


def test_unconfirmed_kill_is_flaky(mset, monkeypatch) -> None:
    """A kill whose single-input rerun comes back CLEAN counts as flaky."""
    k = _site_index(mset, MutOp.ICMP_BOUNDARY, "icmp ult i64")
    target = mset.mutants[k]
    real = mutant_kill.replay

    def fake(binary, pack, mask, timeout_ms):
        out = real(binary, pack, mask, timeout_ms)
        is_rerun = binary == target and mask is None
        return bytes([Status.CLEAN] * len(out)) if is_rerun else out

    monkeypatch.setattr(mutant_kill, "replay", fake)
    got = _scorer(mset).score_inputs([b"abcd"])
    assert k not in got.killed
    assert got.flaky >= 1


def test_confirm_none_skips_rerun(mset, monkeypatch) -> None:
    calls: list[bytes | None] = []
    real = mutant_kill.replay

    def spy(binary, pack, mask, timeout_ms):
        calls.append(mask)
        return real(binary, pack, mask, timeout_ms)

    monkeypatch.setattr(mutant_kill, "replay", spy)
    got = _scorer(mset, Confirm.NONE).score_inputs([b"abcd"])
    # One original replay (mask None) plus one masked replay per mutant.
    assert calls.count(None) == 1
    assert len(calls) == 1 + len(mset.mutants)
    assert got.flaky == 0


def test_bad_jobs_raises(mset) -> None:
    with pytest.raises(ValueError):
        KillScorer(mset, jobs=0, timeout_ms=_TIMEOUT_MS, confirm=Confirm.NONE)
