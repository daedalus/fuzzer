"""Regression: havoc must not discard the rest of a round's n_mutations.

``mutate()``'s per-round loop used to ``return result`` the instant havoc
was selected, discarding whatever iterations of ``n_mutations`` remained.
Since ``n_mutations`` is scaled up by ``_last_perf_score`` (SeedScorer's
energy multiplier), a seed that earned extra mutation budget got none of
the extra whenever havoc was drawn early -- which was often, since havoc
is the most-drawn operator. Fixed by falling through to the shared loop
tail (``continue``) instead of returning, same as every other operator.
"""

from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine


def _make_real_fuzzer(**kwargs):
    """A real Fuzzer instance, for exercising the full mutate() orchestrator."""
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=64,
        timeout=1,
        mutations_per_input=1,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


class TestHavocDoesNotShortCircuitMutate:
    def test_havoc_consumes_full_n_mutations_budget(self):
        # Force every draw to be havoc, and force a seed-energy multiplier
        # well above 1x so n_mutations > 1. Before the fix, only the first
        # havoc draw ever ran; _last_ops_used would have length 1 no matter
        # how large n_mutations was.
        f = _make_real_fuzzer(mutations_per_input=5)
        f._last_perf_score = 300.0  # 3x multiplier -> n_mutations = 15
        engine = OperatorEngine(f)
        engine.select_op = lambda _ops: "havoc"  # noqa: E731

        data = bytes((i * 41) % 256 for i in range(20))
        engine.mutate(data)

        assert len(f._last_ops_used) == 15
        assert all(op == "havoc" for op in f._last_ops_used)

    def test_havoc_result_still_respects_max_len(self):
        f = _make_real_fuzzer(mutations_per_input=10, max_len=10)
        engine = OperatorEngine(f)
        engine.select_op = lambda _ops: "havoc"  # noqa: E731

        data = bytes((i * 41) % 256 for i in range(10))
        for _ in range(50):
            result = engine.mutate(data)
            assert len(result) <= f.max_len

    def test_havoc_mid_round_does_not_end_the_round(self):
        # Alternate havoc with a cheap deterministic op so we can see
        # whether havoc terminates the round early: with the fix, the
        # non-havoc op after havoc should still run.
        f = _make_real_fuzzer(mutations_per_input=4)
        engine = OperatorEngine(f)
        calls = iter(["havoc", "bit_flip", "havoc", "bit_flip"])
        engine.select_op = lambda _ops: next(calls)  # noqa: E731

        data = bytes((i * 41) % 256 for i in range(20))
        engine.mutate(data)

        assert f._last_ops_used == ["havoc", "bit_flip", "havoc", "bit_flip"]
