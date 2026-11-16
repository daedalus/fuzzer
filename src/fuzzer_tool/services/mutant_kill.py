"""Corpus -> mutant-kill score (mutant-kill metric, P3).

A mutant is *killed* when some input runs CLEAN on the original and CRASHes
the mutant (Görz et al. 2023). Inputs that crash or hang the original are no
evidence either way. A HANG on a clean input is reported, never a kill.

    corpus ──pack──> original ──statuses──> mask (1 = CLEAN)
                                              │
            ┌──────── jobs threads ───────────┤
            ▼                                 ▼
       mutant_k(pack, mask)  ...  stops at the first eligible CRASH
            │
            ├── CRASH at i ──rerun input i alone──> CRASH: killed, else flaky
            ├── HANG on an eligible input ────────> hung
            └── otherwise ────────────────────────> survived

Plan: ``docs/action_plan_mutant_kill_metric.md``.
"""

from __future__ import annotations

import enum
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fuzzer_tool.adapters.filesystem import iter_seed_batches
from fuzzer_tool.adapters.mutant_build import MutantSet, Status, pack_record, replay, write_pack

# Oracle binaries are small parsers; one second separates a hang from work.
DEFAULT_TIMEOUT_MS = 1000

# Corpus read granularity: bounds memory while streaming into the pack.
_CORPUS_BATCH = 256

_PACK = "corpus.pack"


class Confirm(enum.Enum):
    """Whether a kill is re-run on its killing input alone before it counts."""

    NONE = 0
    RERUN = 1


class _Outcome(enum.Enum):
    SURVIVED = 0
    KILLED = 1
    HUNG = 2
    FLAKY = 3


@dataclass(frozen=True, slots=True)
class KillScore:
    """One corpus against one mutant set. ``killed``/``hung`` hold mutant indices."""

    kills: int
    hangs: int
    flaky: int
    inputs: int
    clean_inputs: int
    killed: frozenset[int]
    hung: frozenset[int]


_EMPTY = frozenset[int]()


class KillScorer:
    """Scores corpora against a built ``MutantSet``."""

    def __init__(
        self,
        mutants: MutantSet,
        jobs: int,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        confirm: Confirm = Confirm.RERUN,
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs must be >= 1, got {jobs}")
        self._set = mutants
        self._jobs = jobs
        self._timeout_ms = timeout_ms
        self._confirm = confirm

    @classmethod
    def open(cls, kill_set: Path, jobs: int, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> KillScorer:
        """Scorer for the mutant set built in *kill_set*."""
        return cls(MutantSet.load(kill_set), jobs, timeout_ms)

    def score(self, corpus: Path) -> KillScore:
        """Score every live seed of *corpus* (seeds/, seeds.zip, deltas)."""
        return self.score_inputs(_iter_corpus(corpus))

    def score_inputs(self, inputs: Iterable[bytes]) -> KillScore:
        with tempfile.TemporaryDirectory(prefix="mutant_kill_") as tmp:
            pack = Path(tmp) / _PACK
            n = write_pack(inputs, pack)
            if n == 0:
                return KillScore(0, 0, 0, 0, 0, _EMPTY, _EMPTY)

            orig = replay(self._set.original, pack, None, self._timeout_ms)
            mask = bytes(s == Status.CLEAN for s in orig)
            clean = sum(mask)
            if clean == 0:
                return KillScore(0, 0, 0, n, 0, _EMPTY, _EMPTY)

            with ThreadPoolExecutor(self._jobs) as pool:
                outcomes = list(
                    pool.map(lambda b: self._judge(b, pack, mask, Path(tmp)), self._set.mutants)
                )

        def ids(kind: _Outcome) -> frozenset[int]:
            return frozenset(k for k, o in enumerate(outcomes) if o is kind)

        killed, hung = ids(_Outcome.KILLED), ids(_Outcome.HUNG)
        flaky = len(ids(_Outcome.FLAKY))
        return KillScore(len(killed), len(hung), flaky, n, clean, killed, hung)

    def _judge(self, binary: Path, pack: Path, mask: bytes, tmp: Path) -> _Outcome:
        """One mutant's outcome; the driver stops right after an eligible crash."""
        got = replay(binary, pack, mask, self._timeout_ms)
        last = len(got) - 1
        if last >= 0 and got[last] == Status.CRASH and mask[last]:
            return self._confirmed(binary, pack, last, tmp)

        hang = any(s == Status.HANG and ok for s, ok in zip(got, mask, strict=False))
        return _Outcome.HUNG if hang else _Outcome.SURVIVED

    def _confirmed(self, binary: Path, pack: Path, index: int, tmp: Path) -> _Outcome:
        """Re-run the killing input alone: a kill that does not reproduce is flaky."""
        if self._confirm is Confirm.NONE:
            return _Outcome.KILLED

        one = tmp / f"rerun_{binary.name}.pack"
        write_pack([pack_record(pack, index)], one)
        try:
            again = replay(binary, one, None, self._timeout_ms)
        finally:
            one.unlink()
        return _Outcome.KILLED if again[:1] == bytes([Status.CRASH]) else _Outcome.FLAKY


def _iter_corpus(corpus: Path) -> Iterator[bytes]:
    for batch in iter_seed_batches(corpus, _CORPUS_BATCH):
        yield from batch
