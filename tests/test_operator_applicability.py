"""Per-operator rates must not pool an operator's two regimes.

A sniffer-gated operator is routinely handed input its format does not
match, by two separate mechanisms in ``_format_available``:

  * the bootstrap trickle (``_FORMAT_BOOTSTRAP_RATE``) offers a never-seen
    format on non-matching input, so a target that parses it stays
    reachable from a garbage corpus;
  * the live-format short circuit -- once a format has been seen *once*,
    ``name in live`` returns True for every later input, matching or not.
    On a mixed corpus this is much the larger of the two, and the handover
    item describing this defect mentions only the trickle.

What it does with that input is the part that matters here, and it is not
what the handover assumed. It does not decline: ``_op_png_chunk_mutate``
runs ``parse_png_chunks(buf)`` and, failing that, calls
``_generate_random_png()`` -- it synthesises a whole file of its format
from scratch. Every format op has that shape. So these selections do fire,
and they do find edges; a first cut at this change dropped them from the
denominator only, and promptly reported successes against a denominator of
zero on a live run.

Both regimes are therefore counted, separately: Count/Success for
everything, Applic/SuccA for the mutate-a-real-file regime alone. The
numerator and denominator of each pair have to cover the same selections.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.operator_registry import (
    _FORMAT_SNIFFERS,
    format_gate_matches,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
PLAIN = b"hello world, definitely not an image"


class TestFormatGateMatches:
    def test_none_for_an_ungated_operator(self):
        """None is 'applicable to anything', which is not the same as False."""
        assert format_gate_matches("bit_flip", PLAIN) is None
        assert format_gate_matches("havoc", PNG) is None

    def test_true_on_matching_input(self):
        assert format_gate_matches("png_chunk_mutate", PNG) is True
        assert format_gate_matches("jpeg_chunk_mutate", JPEG) is True

    def test_false_on_non_matching_input(self):
        assert format_gate_matches("png_chunk_mutate", JPEG) is False
        assert format_gate_matches("png_chunk_mutate", PLAIN) is False

    def test_false_on_empty_input(self):
        assert format_gate_matches("png_chunk_mutate", b"") is False

    @pytest.mark.parametrize("name", sorted(_FORMAT_SNIFFERS))
    def test_every_gated_operator_answers_without_raising(self, name):
        """Sniffers index into the buffer; a short one must not except.

        Accounting runs on every exec, so a sniffer that raises on a
        1-byte input would take the fuzz loop down rather than mis-count.
        """
        for probe in (b"", b"\x00", b"\xff\xd8", PLAIN, PNG):
            assert format_gate_matches(name, probe) in (True, False)


class TestApplicabilityAccounting:
    """The counters as fuzz_one maintains them, without a target."""

    @staticmethod
    def _count(ops, data, counts, applicable):
        """Mirrors fuzz_one's accounting, setdefault included.

        The setdefault is load-bearing: without it an operator that never
        meets its own format is absent from the dict rather than present at
        zero, and the report cannot tell that from a state file written
        before this was tracked. It falls back to the raw count and prints
        the pooled rate -- which is the thing being separated out.
        """
        for op in set(ops):
            counts[op] = counts.get(op, 0) + 1
            applicable.setdefault(op, 0)
            if format_gate_matches(op, data) is not False:
                applicable[op] += 1

    def test_ungated_operators_have_identical_counters(self):
        counts, applicable = {}, {}
        for data in (PNG, JPEG, PLAIN):
            self._count(["bit_flip", "havoc"], data, counts, applicable)
        assert counts == applicable == {"bit_flip": 3, "havoc": 3}

    def test_gated_operator_only_counts_matching_input(self):
        counts, applicable = {}, {}
        for data in (PNG, JPEG, PLAIN, PLAIN):
            self._count(["png_chunk_mutate"], data, counts, applicable)
        assert counts["png_chunk_mutate"] == 4
        assert applicable["png_chunk_mutate"] == 1

    def test_the_distortion_it_separates(self):
        """A working operator on a corpus that is 5% its format.

        It succeeds on every PNG it is handed and is offered the other 95%
        anyway, where it synthesises PNGs that happen to find nothing. The
        pooled rate reads 5%; restricted to the regime being asked about it
        is the 100% it actually is. Neither number is wrong -- they answer
        different questions -- but only the second is about the operator,
        and the first moves when the corpus does.
        """
        counts, applicable = {}, {}
        successes, succ_applicable = {}, {}
        corpus = [PNG] * 5 + [PLAIN] * 95
        for data in corpus:
            self._count(["png_chunk_mutate"], data, counts, applicable)
            if format_gate_matches("png_chunk_mutate", data):
                successes["png_chunk_mutate"] = successes.get("png_chunk_mutate", 0) + 1
                succ_applicable["png_chunk_mutate"] = succ_applicable.get("png_chunk_mutate", 0) + 1

        pooled = successes["png_chunk_mutate"] / counts["png_chunk_mutate"]
        restricted = succ_applicable["png_chunk_mutate"] / applicable["png_chunk_mutate"]
        assert pooled == pytest.approx(0.05)
        assert restricted == pytest.approx(1.0)

    def test_numerator_and_denominator_cover_the_same_selections(self):
        """The invariant the first cut of this change broke.

        A success credited on a selection the denominator excluded gives
        SuccA > Applic, and at the limit a success with Applic == 0.
        """
        counts, applicable = {}, {}
        successes, succ_applicable = {}, {}
        for data in [PLAIN] * 20:
            self._count(["png_chunk_mutate"], data, counts, applicable)
            # Synthesis regime: the op fires and may well succeed.
            successes["png_chunk_mutate"] = successes.get("png_chunk_mutate", 0) + 1
            if format_gate_matches("png_chunk_mutate", data):
                succ_applicable["png_chunk_mutate"] = succ_applicable.get("png_chunk_mutate", 0) + 1

        assert counts["png_chunk_mutate"] == 20
        assert successes["png_chunk_mutate"] == 20
        assert applicable["png_chunk_mutate"] == 0
        assert succ_applicable.get("png_chunk_mutate", 0) == 0


class TestReportUsesApplicable:
    def test_ratea_column_divides_by_applicable(self):
        from fuzzer_tool.services.report import _mutation_effectiveness

        class _F:
            op_counts = {"png_chunk_mutate": 100}
            op_success = {"png_chunk_mutate": 5}
            op_applicable = {"png_chunk_mutate": 5}
            op_success_applicable = {"png_chunk_mutate": 5}

        out = _mutation_effectiveness(_F())
        assert "100.0%" in out  # RateA: 5/5
        assert "  5.0%" in out  # Rate: 5/100, still reported
        assert "Applic" in out

    def test_na_when_the_operator_never_met_its_format(self):
        """Applic 0 with successes from the synthesis regime. The rate in
        that regime is undefined, and 0.0% would read as broken."""
        from fuzzer_tool.services.report import _mutation_effectiveness

        class _F:
            op_counts = {"png_chunk_mutate": 100}
            op_success = {"png_chunk_mutate": 3}
            op_applicable = {"png_chunk_mutate": 0}
            op_success_applicable: dict = {}

        out = _mutation_effectiveness(_F())
        assert "n/a" in out
        assert "  3.0%" in out  # the raw rate still says what happened

    def test_falls_back_to_the_raw_pair_when_unknown(self):
        """A state file written before this was tracked.

        Missing means unknown, not zero -- backfilling zero would divide
        by it and erase the operator from the report.
        """
        from fuzzer_tool.services.report import _mutation_effectiveness

        class _F:
            op_counts = {"png_chunk_mutate": 100}
            op_success = {"png_chunk_mutate": 5}
            op_applicable: dict = {}
            op_success_applicable: dict = {}

        out = _mutation_effectiveness(_F())
        assert "  5.0%" in out
        assert "n/a" not in out

    def test_ungated_operator_has_identical_pairs(self):
        from fuzzer_tool.services.report import _mutation_effectiveness

        class _F:
            op_counts = {"bit_flip": 100}
            op_success = {"bit_flip": 5}
            op_applicable = {"bit_flip": 100}
            op_success_applicable = {"bit_flip": 5}

        out = _mutation_effectiveness(_F())
        row = next(ln for ln in out.splitlines() if "bit_flip" in ln)
        assert row.split() == [
            "bit_flip",
            "100",
            "5",
            "5.0%",
            "100",
            "5",
            "5.0%",
            "2.2%",
            "4.4%",
            "6.5%",
        ]

    def test_note_only_appears_when_the_regimes_split(self):
        from fuzzer_tool.services.report import _mutation_effectiveness

        class _Split:
            op_counts = {"png_chunk_mutate": 100}
            op_success = {"png_chunk_mutate": 5}
            op_applicable = {"png_chunk_mutate": 5}
            op_success_applicable = {"png_chunk_mutate": 5}

        class _Ungated:
            op_counts = {"bit_flip": 100}
            op_success = {"bit_flip": 5}
            op_applicable = {"bit_flip": 100}
            op_success_applicable = {"bit_flip": 5}

        assert "synthesising one from scratch" in _mutation_effectiveness(_Split())
        assert "synthesising one from scratch" not in _mutation_effectiveness(_Ungated())


class TestStatePersistence:
    def test_absent_key_loads_as_empty_not_backfilled(self):
        """Resuming a pre-applicability state must not assert that every
        historic selection was applicable."""
        state: dict = {"op_counts": {"png_chunk_mutate": 100}}
        loaded = state.get("op_applicable", {})
        assert loaded == {}
