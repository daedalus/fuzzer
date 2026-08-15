"""Regression: no registered mutation operator may be a pure no-op.

``byte_shuffle`` shuffled a throwaway bytearray slice copy and returned its
input byte-for-byte unchanged for every input (see
``test_mutations.TestByteShuffleRegression``). An operator like that is worse
than useless: the schedulers still select it, spend budget on it, and credit
it, so it dilutes every bandit/Elo signal while producing nothing.

This sweep drives *every* registered operator through the real dispatch table
(``REGISTRY.dispatch``) over a fixed, diverse input battery and requires that
each operator which declares itself *available* (``REGISTRY.available``) for an
input changes at least one input.

To stay deterministic and non-flaky, each operator is evaluated in isolation
with the fuzzer RNG reseeded per repetition, so one operator's RNG consumption
never cascades into another's. Operators gated on runtime scheduling state a
unit test does not build (dictionary scratch, grammar, cmplog pairs,
CEM/markov) report themselves unavailable and are skipped, so no
hand-maintained skip list is needed beyond one structural operator noted below.
"""

from __future__ import annotations

import os
import random
import tempfile
from unittest.mock import patch

import fuzzer_tool.core.mutations as _mutations
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.fuzzer import Fuzzer

_REPS = 12
_BASE_SEED = 4242


def _battery() -> list[bytes]:
    """Fixed input battery: random inputs of several lengths (deterministic via
    a local seed) plus magic-prefixed samples so format-aware operators become
    available and get exercised."""
    rng = random.Random(999)
    inputs = [
        bytes(rng.randrange(256) for _ in range(n))
        for n in (8, 32, 128, 512)
        for _ in range(4)
    ]
    inputs += [
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(30),
        b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(30),
        b"GIF89a" + bytes(30),
        b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(30),
        bytes(4) + b"ftypisom" + bytes(30),
        b"\x1aE\xdf\xa3" + bytes(30),
        b"PK\x03\x04" + bytes(30),
        b"\x1f\x8b\x08\x00" + bytes(30),
        b"Rar!\x1a\x07\x00" + bytes(30),
        b"BM\x8a\x00\x00\x00" + bytes(30),
        b"12345 6789 -3 0.5 abcdef ghij",
    ]
    return inputs


class TestNoMutationIsAPureNoOp:
    # `available` is decided by cheap magic/heuristic sniffing, but a few
    # structural mutators additionally need a well-formed file body the sweep
    # does not synthesize (a bare 4-byte magic is not a parseable file). They
    # are inert on a magic prefix *by design*, not no-ops on real input --
    # feeding a valid sample would let them be covered here too.
    _NEEDS_VALID_FILE_BODY = {"elf_chunk_mutate"}

    def _make_fuzzer(self) -> Fuzzer:
        tmp = tempfile.mkdtemp(prefix="noop_sweep_")
        os.makedirs(f"{tmp}/corpus", exist_ok=True)
        # /bin/true stands in for a target; the isfile/access patches let the
        # constructor accept it, and _setup_forkserver is neutralised so no
        # compiled fuzz_loader is required (this is a pure mutation unit test).
        with (
            patch.object(Fuzzer, "_setup_forkserver", lambda self: None),
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmp}/corpus",
                crashes_dir=f"{tmp}/crashes",
                max_len=4096,
                timeout=1,
                mutations_per_input=2,
            )
        # Dictionary/grammar/cmplog operators need per-seed scheduling state the
        # mutate loop builds; leave it empty so they report unavailable and are
        # skipped rather than flagged as false no-ops.
        f.dictionary = []
        # Cross-seed operators (splice/crossover) need >1 distinct corpus entry.
        f.corpus = [
            bytearray(s)
            for s in (
                b"the quick brown fox jumped 12345",
                b"a second, distinct corpus seed \xff\xfe\x01",
                b"\x89PNG\r\n\x1a\nIHDR....",
                b"third seed payload zzzzzzzz",
            )
        ]
        return f

    def _sweep(self, f: Fuzzer) -> tuple[set[str], set[str]]:
        """Return (available_somewhere, changed_somewhere) operator-name sets.

        Each operator is exercised on its own with the fuzzer RNG reseeded per
        repetition, so the result is deterministic and independent of operator
        ordering or of any other operator's RNG use.
        """
        table = REGISTRY.dispatch(f._operators)
        battery = _battery()
        names = sorted({n for inp in battery for n in REGISTRY.available(f, inp)})
        available: set[str] = set(names)
        changed: set[str] = set()
        for name in names:
            done = False
            for rep in range(_REPS):
                f._rand_pool.reseed(_BASE_SEED + rep)
                for inp in battery:
                    if name not in set(REGISTRY.available(f, inp)):
                        continue
                    buf = bytearray(inp)
                    idx = len(buf) // 2
                    try:
                        ret = table[name](buf, idx, bytes(inp))
                    except Exception:
                        # An operator raising is a separate concern; a no-op is
                        # about *silence*, not errors. Skip and let other tests
                        # cover exceptions.
                        continue
                    out = ret if isinstance(ret, (bytes, bytearray)) else buf
                    if bytes(out) != inp:
                        changed.add(name)
                        done = True
                        break
                if done:
                    break
        return available, changed

    def test_every_available_operator_changes_some_input(self):
        f = self._make_fuzzer()
        available, changed = self._sweep(f)

        # Guard the guard: if the registry/dispatch wiring breaks, `available`
        # collapses and the no-op check becomes vacuously true. Require a broad
        # operator set to actually have been exercised.
        assert len(available) > 80, f"only {len(available)} operators exercised"

        noops = (available - changed) - self._NEEDS_VALID_FILE_BODY
        assert not noops, (
            "operator(s) are available but never changed any input "
            f"(pure no-ops): {', '.join(sorted(noops))}"
        )

    def test_sweep_detects_a_planted_no_op(self):
        """Meta-test: the sweep must fail on an actual no-op, or it protects
        nothing. Plant a no-op ``byte_shuffle`` and confirm it is flagged."""
        original = _mutations.byte_shuffle
        _mutations.byte_shuffle = lambda data, rng=None: bytes(data)
        try:
            f = self._make_fuzzer()
            available, changed = self._sweep(f)
            assert "byte_shuffle" in available
            assert "byte_shuffle" in (available - changed), (
                "a planted no-op byte_shuffle went undetected — the sweep "
                "does not actually protect against no-op operators"
            )
        finally:
            _mutations.byte_shuffle = original
