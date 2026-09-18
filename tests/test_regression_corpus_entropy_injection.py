"""save_to_corpus() mixes each newly-admitted seed's bytes into f._rng.

RandPool.inject_entropy() was added deliberately unwired (no call site
anywhere) so a future entropy source could use it without a new mechanism
being invented at that point. This wires the first (and, as of this patch,
only) call site: a genuinely new corpus admission is real "raw" material --
bytes from an actual program execution outcome, not a re-derivation of the
pool's own state -- and admission order is itself a deterministic function
of the campaign seed in single-worker mode, so wiring it here must not
reintroduce the run-to-run variance the rest of this module's determinism
work removed.

Three properties pinned down:

1. Determinism: two campaigns with the same seed and the same sequence of
   admitted seeds produce identical post-admission draws from f._rng.
2. It actually has an effect: admitting different bytes leaves f._rng in a
   different state (otherwise "wired" would be a no-op in disguise).
3. Rejected admissions (exact duplicate; the seen_hashes/bloom novelty gate
   inside the free-function save_to_corpus() returns False) do not inject
   -- only genuine new content does.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch


def _make_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="corpus_entropy_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            seed=1234,
            **kwargs,
        )


def _draw16(f):
    return [f._rng.randint(0, 255) for _ in range(16)]


class TestCorpusAdmissionInjectsEntropy:
    def test_same_admissions_same_seed_same_downstream_draws(self):
        f1 = _make_fuzzer()
        f2 = _make_fuzzer()
        for chunk in (b"AAAA", b"BBBBBB", b"CC"):
            f1.save_to_corpus(chunk)
            f2.save_to_corpus(chunk)
        assert _draw16(f1) == _draw16(f2)

    def test_different_admissions_diverge_downstream_draws(self):
        f1 = _make_fuzzer()
        f2 = _make_fuzzer()
        f1.save_to_corpus(b"AAAA")
        f2.save_to_corpus(b"ZZZZ")
        assert _draw16(f1) != _draw16(f2)

    def test_duplicate_admission_does_not_inject_twice(self):
        # Same seed, same single admission repeated on one fuzzer vs. once
        # on another: the second save_to_corpus(same bytes) call is a
        # duplicate under the seen_hashes/bloom gate and must not fire a
        # second injection, or the two fuzzers' streams would diverge.
        f1 = _make_fuzzer()
        f1.save_to_corpus(b"AAAA")
        f1.save_to_corpus(b"AAAA")  # duplicate: rejected before injection

        f2 = _make_fuzzer()
        f2.save_to_corpus(b"AAAA")  # admitted once

        assert _draw16(f1) == _draw16(f2)
