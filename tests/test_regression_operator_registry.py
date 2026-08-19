"""Regression tests: operator registry is the single source of truth.

Covers the dispatcher refactor: every mutation operator is registered exactly
once in ``fuzzer_tool.core.operator_registry.REGISTRY``; the legacy lists in
``core/mutations/generic.py``, the runtime dispatch table, and
``OPERATOR_CATEGORIES`` all derive from it. Guards the two historical drift
instances (``colorization`` registered with no op-list entry,
``block_shuffle_variable`` categorized nowhere) from returning.
"""

import os
import types

import pytest

from fuzzer_tool.core.mutations import DICT_MUTATIONS, FORMAT_MUTATIONS, MUTATIONS
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import (
    REGISTRY,
    OperatorRegistry,
    OperatorSpec,
    _sniff_dct_transform_coded,
    _sniff_der,
    _sniff_mesh_or_vector_geometry,
    _sniff_rar,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

# Ops that are dispatchable but intentionally live outside the legacy lists:
# gated by per-run conditions (markov/cem/grammar/cmplog/redqueen) or
# dispatch-only or gated on cmplog pairs (colorization).
_CONDITIONAL_OPS = {
    "markov_bytes",
    "cem_bytes",
    "grammar_mutate",
    "grammar_tree_mutate",
    "redqueen",
    "colorization",
}


def _build_fuzzer():
    """Build a minimal real Fuzzer instance for dispatch coverage checks."""
    import tempfile
    from pathlib import Path

    from fuzzer_tool.services.fuzzer import Fuzzer

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        crashes = Path(tmp) / "crashes"
        corpus.mkdir()
        crashes.mkdir()
        return Fuzzer(
            target=str(Path(__file__).resolve().parent.parent / "targets" / "test_target"),
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
        )


class _MockFuzzer:
    """Minimal fuzzer-shaped object for availability-predicate tests."""

    def __init__(self):
        self.dictionary = []
        self.markov_trained = False
        self.mc = None
        self.mc_cem = False
        self.grammar = None
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self.seed_meta = {}
        self.corpus = []


class _CmplogWithPairs:
    pairs = [("target", "replacement")]


class TestRegistrySingleSourceOfTruth:
    """The legacy op lists must never drift from the registry."""

    def test_every_legacy_op_is_registered(self):
        registered = set(REGISTRY.names())
        legacy = set(MUTATIONS) | set(FORMAT_MUTATIONS) | set(DICT_MUTATIONS)
        missing = sorted(legacy - registered)
        assert not missing, f"Legacy ops missing from REGISTRY: {missing}"

    def test_conditional_ops_registered(self):
        registered = set(REGISTRY.names())
        missing = sorted(_CONDITIONAL_OPS - registered)
        assert not missing, f"Gated/dispatch-only ops missing from REGISTRY: {missing}"

    def test_format_ops_categorized_format(self):
        for op in FORMAT_MUTATIONS:
            assert REGISTRY.category_of(op) == "format", f"{op} not in format band"

    def test_der_ops_categorized_format(self):
        for op in ("der_len_mutate", "der_tag_mutate", "der_tlv_reorder", "der_tlv_insert"):
            assert REGISTRY.category_of(op) == "format", f"{op} not in format band"

    def test_dict_ops_categorized_dict(self):
        for op in DICT_MUTATIONS:
            assert REGISTRY.category_of(op) == "dict", f"{op} not in dict band"

    def test_every_op_in_exactly_one_category(self):
        cats = REGISTRY.categories()
        union: set[str] = set()
        for ops in cats.values():
            union |= ops
        assert len(union) == len(REGISTRY.names())
        assert sum(len(ops) for ops in cats.values()) == len(union)

    def test_operator_categories_derived_from_registry(self):
        assert REGISTRY.categories() == OPERATOR_CATEGORIES

    def test_known_drift_fixes_stay_fixed(self):
        # block_shuffle_variable was in MUTATIONS + the dispatch table but
        # absent from OPERATOR_CATEGORIES before the refactor.
        assert "block_shuffle_variable" in OPERATOR_CATEGORIES["block"]
        # colorization was in the dispatch table but in no op list.
        assert "colorization" in REGISTRY.names()
        assert "colorization" in OPERATOR_CATEGORIES["adaptive"]
        # grammar ops were absent from OPERATOR_CATEGORIES before the refactor.
        assert "grammar_mutate" in OPERATOR_CATEGORIES["adaptive"]
        assert "grammar_tree_mutate" in OPERATOR_CATEGORIES["adaptive"]


class TestRegistryDispatchCoverage:
    """Every registered operator must have a live dispatch handler."""

    def test_fuzzer_dispatch_matches_registry(self):
        fuzzer = _build_fuzzer()
        assert set(fuzzer._op_dispatch) == set(REGISTRY.names())


class TestAvailabilityPredicates:
    """REGISTRY.available() mirrors the historic build_ops() gating."""

    def _available(self, **attrs):
        fuzzer = _MockFuzzer()
        for name, value in attrs.items():
            setattr(fuzzer, name, value)
        return set(REGISTRY.available(fuzzer, b"seed"))

    def test_base_and_format_ops_unconditional(self):
        avail = self._available()
        assert "bit_flip" in avail
        assert "havoc" in avail
        assert "png_chunk_mutate" in avail

    def test_der_ops_available(self):
        avail = self._available()
        assert {"der_len_mutate", "der_tag_mutate", "der_tlv_reorder", "der_tlv_insert"} <= avail

    def test_der_sniffer_matches_seed_bytes(self):
        assert _sniff_der(b"\x30\x03\x02\x01\x05")
        assert _sniff_der(b"\x31\x81\x02")  # SET with long form
        assert _sniff_der(b"\x30\x80")  # indefinite (BER)
        assert not _sniff_der(b"\x30")  # too short for a length byte
        assert not _sniff_der(b"\x02\x01\x05")  # not SEQUENCE/SET
        assert not _sniff_der(b"PK\x03\x04")
        assert not _sniff_der(b"\x30\x85\x00\x00\x00\x00")  # 5 length bytes: out of range

    def test_dict_ops_gated_on_dictionary(self):
        assert not {"dict_insert", "dict_replace", "checksum_repair"} & self._available()
        avail = self._available(dictionary=["token"])
        assert {"dict_insert", "dict_replace", "checksum_repair"} <= avail

    def test_markov_gated_on_trained(self):
        assert "markov_bytes" not in self._available()
        assert "markov_bytes" in self._available(markov_trained=True)

    def test_cem_gated_on_fitted(self):
        assert "cem_bytes" not in self._available()
        assert "cem_bytes" not in self._available(mc=object(), mc_cem=True)
        fitted = type("MC", (), {"cem_fitted": True})()
        assert "cem_bytes" in self._available(mc=fitted, mc_cem=True)

    def test_grammar_gated_on_grammar(self):
        assert "grammar_mutate" not in self._available()
        avail = self._available(grammar=object())
        assert {"grammar_mutate", "grammar_tree_mutate"} <= avail

    def test_cmplog_ops_gated_on_pairs(self):
        assert "redqueen_xform" not in self._available()
        assert "gradient_cmp" not in self._available()
        avail = self._available(_cmplog=_CmplogWithPairs())
        assert {"redqueen_xform", "gradient_cmp"} <= avail

    def test_regex_bomb_gated_on_flag(self):
        assert "regex_bomb" not in self._available()
        assert "regex_bomb" in self._available(enable_regex_bomb=True)

    def test_x86_mutate_gated_on_flag(self):
        assert "x86_chunk_mutate" not in self._available()
        assert "x86_chunk_mutate" in self._available(enable_x86_mutator=True)

    def test_arm_mutate_gated_on_flag(self):
        assert "arm_chunk_mutate" not in self._available()
        assert "arm_chunk_mutate" in self._available(enable_arm_mutator=True)

    def test_redqueen_gated_on_seed_meta(self):
        assert "redqueen" not in self._available()
        fuzzer = _MockFuzzer()
        fuzzer.seed_meta = {b"seed": {"redqueen_matches": ["x"]}}
        assert "redqueen" in REGISTRY.available(fuzzer, b"seed")
        fuzzer.seed_meta = {b"seed": {"redqueen_offsets": [1]}}
        assert "redqueen" in REGISTRY.available(fuzzer, b"seed")

    def test_colorization_gated_on_cmplog_pairs(self):
        """Was `_never`, on the stated grounds of "historic build_ops
        behavior". That behaviour was an accident -- colorization was in the
        dispatch table and in no op list, so nothing could draw it -- and the
        refactor preserved it. It is gated on the cmplog signal it needs now:
        without pairs the handler picks offsets at random, which havoc
        already does better."""
        assert "colorization" not in self._available()
        fuzzer = _MockFuzzer()
        fuzzer._cmplog = types.SimpleNamespace(pairs=[(b"IHDR", b"IDAT")])
        assert "colorization" in REGISTRY.available(fuzzer, b"seed")

    def test_build_ops_wired_to_registry(self):
        from fuzzer_tool.services.operators import OperatorEngine

        fuzzer = _MockFuzzer()
        engine = OperatorEngine(fuzzer)
        assert engine.build_ops(b"seed") == REGISTRY.available(fuzzer, b"seed")


class TestRegistryMechanics:
    """Registry invariants: uniqueness, handler contract, cache freshness."""

    def test_duplicate_registration_raises(self):
        registry = OperatorRegistry()
        spec = OperatorSpec(name="dup_op", category="bit", handler_name="_op_dup_op")
        registry.register(spec)
        with pytest.raises(ValueError):
            registry.register(spec)

    def test_dispatch_raises_on_missing_handler(self):
        registry = OperatorRegistry()
        registry.register(
            OperatorSpec(name="no_handler", category="bit", handler_name="_op_no_handler")
        )
        with pytest.raises(AttributeError):
            registry.dispatch(object())

    def test_register_invalidates_categories_cache(self):
        registry = OperatorRegistry()
        registry.register(OperatorSpec(name="cache_a", category="bit", handler_name="_op_cache_a"))
        registry.categories()
        registry.register(OperatorSpec(name="cache_b", category="bit", handler_name="_op_cache_b"))
        assert "cache_b" in registry.categories()["bit"]


class TestRegularityOperators:
    """The dieharder-inverse band registers and dispatches like any other.

    These operators arrived as a new category rather than as additions to an
    existing one, which is the case most likely to drift: a band that no
    availability predicate, no scheduler and no category test knows about
    would still import cleanly and simply never be selected.
    """

    REGULARITY_OPS = frozenset(
        {
            "birthday_collide",
            "degenerate_geometry",
            "float_squeeze",
            "gcd_worst_case",
            "invariant_break",
            "kmer_saturate",
            "kmer_saturate_bits",
            "kmer_starve",
            "lag_correlate",
            "monotone_fill",
            "perm_lock",
            "popcount_lock",
            "rank_deficient",
            "spectral_peak",
        }
    )

    def test_band_is_registered(self):
        assert set(REGISTRY.names()) >= self.REGULARITY_OPS

    def test_band_categorized_regularity(self):
        assert OPERATOR_CATEGORIES["regularity"] == self.REGULARITY_OPS

    def test_every_op_has_a_handler(self):
        engine = OperatorEngine(_MockFuzzer())
        dispatch = REGISTRY.dispatch(engine)
        for name in self.REGULARITY_OPS:
            assert callable(dispatch[name])

    def test_all_but_invariant_break_are_unconditional(self):
        """Only the corpus-measuring operator is gated on corpus state.

        spectral_peak/degenerate_geometry/rank_deficient are gated on format
        relevance (see TestRegularityFormatGating below), but _MockFuzzer has
        no ``_rand_pool``, and the format gate is permissive when it can't
        draw a bootstrap-trickle random number -- so they still come back
        available here.
        """
        fuzzer = _MockFuzzer()
        available = set(REGISTRY.available(fuzzer, b"seed"))
        assert self.REGULARITY_OPS - {"invariant_break"} <= available

    def test_invariant_break_gated_on_corpus_size(self):
        """Below the sample floor, "every input agrees here" is a coincidence.

        Letting the operator run on a two-seed corpus would have it treat
        nearly every offset as structural and overwrite the whole file.
        """
        fuzzer = _MockFuzzer()
        fuzzer.corpus = []
        assert "invariant_break" not in REGISTRY.available(fuzzer, b"seed")
        fuzzer.corpus = [b"x"] * 4
        assert "invariant_break" not in REGISTRY.available(fuzzer, b"seed")
        fuzzer.corpus = [b"x"] * 16
        assert "invariant_break" in REGISTRY.available(fuzzer, b"seed")

    def test_handlers_preserve_length(self):
        """The band's contract: overwrite in place, never resize.

        Everything downstream (frameshift bookkeeping, the max_len clamp)
        stays simple only as long as this holds for every operator here.
        """
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool()
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(self.REGULARITY_OPS - {"invariant_break"}):
            for _ in range(10):
                buf = bytearray(os.urandom(512))
                result = dispatch[name](buf, 0, bytes(buf))
                assert result is not None, name
                assert len(result) == 512, name

    def test_invariant_break_handler_needs_a_corpus(self):
        """Dispatching it directly with no corpus must be a no-op, not a crash.

        The availability predicate normally prevents this, but dispatch is
        reachable independently of build_ops (havoc, deterministic replay), so
        the handler cannot rely on the gate having run.
        """
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool()
        fuzzer.corpus = []
        engine = OperatorEngine(fuzzer)
        buf = bytearray(os.urandom(128))
        assert engine._op_invariant_break(buf, 0, bytes(buf)) is None

    def test_invariant_cache_rebuilds_only_on_growth(self):
        """The corpus scan is O(samples x length); it must not run per call."""
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool()
        fuzzer.corpus = [b"\x89MAGIC\x00\x01" + os.urandom(56) for _ in range(32)]
        engine = OperatorEngine(fuzzer)
        first = engine.corpus_invariants()
        assert first is not None
        assert engine.corpus_invariants() is first
        fuzzer.corpus.append(b"\x89MAGIC\x00\x01" + os.urandom(56))
        assert engine.corpus_invariants() is not first


class TestRegularityFormatGating:
    """spectral_peak/degenerate_geometry/rank_deficient are format-shaped.

    docs/TODO.md flagged these three as burning budget on targets that
    can't use them: spectral_peak imposes a frequency-domain peak (relevant
    to DCT/DST-transform-coded formats), degenerate_geometry constructs
    near-collinear/near-coplanar point sets (relevant to vector/mesh
    formats), rank_deficient constructs a rank-deficient matrix (relevant to
    erasure-coded formats). Gated the same way as the format band: live once
    a real match is seen, a thin bootstrap trickle otherwise.
    """

    def test_spectral_peak_sniffer_matches_dct_coded_formats(self):
        assert _sniff_dct_transform_coded(b"\xff\xd8\xff\xe0rest")  # JPEG
        assert _sniff_dct_transform_coded(b"\x00\x00\x00\x18ftypmp42")  # MP4/ISOBMFF
        assert _sniff_dct_transform_coded(b"\x1a\x45\xdf\xa3rest")  # WebM/Matroska
        assert not _sniff_dct_transform_coded(b"\x89PNG\r\n\x1a\n")
        assert not _sniff_dct_transform_coded(b"plain text")

    def test_degenerate_geometry_sniffer_matches_vector_mesh_formats(self):
        assert _sniff_mesh_or_vector_geometry(b"<?xml version='1.0'?><svg/>")
        assert _sniff_mesh_or_vector_geometry(b"<svg xmlns='...'>")
        assert _sniff_mesh_or_vector_geometry(b"solid mymesh\nfacet normal 0 0 0")
        binary_stl = b"\x00" * 80 + (2).to_bytes(4, "little") + b"\x00" * 100
        assert _sniff_mesh_or_vector_geometry(binary_stl)
        assert not _sniff_mesh_or_vector_geometry(b"\x00" * 80 + (3).to_bytes(4, "little"))
        assert not _sniff_mesh_or_vector_geometry(b"plain text")
        assert not _sniff_mesh_or_vector_geometry(b"")

    def test_rank_deficient_sniffer_matches_rar(self):
        assert _sniff_rar(b"Rar!\x1a\x07\x00")  # RAR4
        assert _sniff_rar(b"Rar!\x1a\x07\x01\x00")  # RAR5
        assert not _sniff_rar(b"PK\x03\x04")
        assert not _sniff_rar(b"Rar!")

    def _fuzzer_with_rand_pool(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        return fuzzer

    def test_unmatched_data_gets_bootstrap_trickle_not_full_availability(self):
        """A target with no format match should mostly (but not always) miss.

        This is the actual budget fix: before gating, these three fired on
        every selection regardless of target; now a non-matching target only
        gets them on the same thin trickle as the format band.
        """
        fuzzer = self._fuzzer_with_rand_pool()
        hits = sum(
            1 for _ in range(4000) if "spectral_peak" in REGISTRY.available(fuzzer, b"plain data")
        )
        rate = hits / 4000
        assert 0.0 < rate < 0.10  # bootstrap trickle (~2%), not 0% or 100%

    def test_matching_seed_makes_op_available_and_live(self):
        fuzzer = self._fuzzer_with_rand_pool()
        jpeg_seed = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        assert "spectral_peak" in REGISTRY.available(fuzzer, jpeg_seed)
        assert "spectral_peak" in fuzzer._live_formats
        # Once live, stays available even on a non-matching seed this round.
        assert "spectral_peak" in REGISTRY.available(fuzzer, b"plain data")

    def test_other_regularity_ops_unaffected_by_gating(self):
        """Only the three format-shaped ops are gated; the rest stay as-is."""
        fuzzer = self._fuzzer_with_rand_pool()
        available = set(REGISTRY.available(fuzzer, b"plain data"))
        untouched = TestRegularityOperators.REGULARITY_OPS - {
            "spectral_peak",
            "degenerate_geometry",
            "rank_deficient",
            "invariant_break",
        }
        assert untouched <= available


class TestFormatAvailableSkipsSniffOnceLive:
    """_format_available's _check() checks the live set before re-running
    sniff() -- docs/TODO.md flags REGISTRY.available() re-evaluating
    data-independent predicates every exec as an open perf lever; this is a
    first, narrower cut at the ~27 sniffer-gated ops specifically. Once a
    format is confirmed live (the steady-state case for the rest of a real
    run once any matching seed has been seen), later calls for that op skip
    straight to a set lookup instead of re-running sniff() -- a struct.unpack
    for STL, a chained byte-prefix check for others -- on every exec.
    """

    def _fuzzer(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        return fuzzer

    def test_sniff_not_called_once_format_is_live(self):
        from fuzzer_tool.core import operator_registry as reg_mod

        fuzzer = self._fuzzer()
        jpeg_seed = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        # First call: not yet live, sniff() must run and match.
        assert "spectral_peak" in REGISTRY.available(fuzzer, jpeg_seed)
        assert "spectral_peak" in fuzzer._live_formats

        calls = []
        real_sniff = reg_mod._sniff_dct_transform_coded

        def spy(data):
            calls.append(data)
            return real_sniff(data)

        reg_mod._FORMAT_SNIFFERS["spectral_peak"] = spy
        try:
            # Now live: sniff() must not run at all, regardless of data.
            assert "spectral_peak" in REGISTRY.available(fuzzer, b"not jpeg at all")
            assert calls == []
        finally:
            reg_mod._FORMAT_SNIFFERS["spectral_peak"] = real_sniff

    def test_behavior_unchanged_from_sniff_first_order(self):
        """Reordering must not change any outcome, only which branch does
        the work: not-live+non-matching -> trickle; not-live+matching ->
        live; live (any data) -> available."""
        fuzzer = self._fuzzer()
        jpeg_seed = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        non_matching = b"plain data, definitely not jpeg"

        # Not yet live + non-matching: thin trickle, not unconditional.
        # (Fresh fuzzer per trial, varied seeds -- reusing one fuzzer would
        # advance the same RNG stream repeatedly, which is fine too, but a
        # fresh instance per trial matches how _MockFuzzer is used elsewhere
        # in this file.)
        hits = 0
        for seed in range(3000):
            trial_fuzzer = _MockFuzzer()
            trial_fuzzer._rand_pool = RandPool(seed=seed)
            if "spectral_peak" in REGISTRY.available(trial_fuzzer, non_matching):
                hits += 1
        assert 0.0 < hits / 3000 < 0.10

        # Not yet live + matching: available now, and becomes live.
        assert "spectral_peak" in REGISTRY.available(fuzzer, jpeg_seed)
        assert "spectral_peak" in fuzzer._live_formats

        # Live: available regardless of what this round's data looks like.
        assert "spectral_peak" in REGISTRY.available(fuzzer, non_matching)
        assert "spectral_peak" in REGISTRY.available(fuzzer, jpeg_seed)

    def test_live_check_precedes_sniff_for_every_gated_format(self):
        """Same guarantee, exercised across every sniffer-gated op at once,
        not just spectral_peak."""
        from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS

        for op_name in _FORMAT_SNIFFERS:
            fuzzer = self._fuzzer()
            fuzzer._live_formats = {op_name}  # force live without a real seed
            spec = REGISTRY._ops[op_name]
            # sniff() would reject this trivially, but live short-circuits
            # before sniff() is ever consulted.
            assert spec.available(fuzzer, b"")


class TestNewByteOperators:
    """insert_repeated_bytes, sort_bytes, and leb128_encode are unconditional
    byte-band operators with live dispatch handlers."""

    NEW_OPS = frozenset(
        {
            "insert_repeated_bytes",
            "sort_bytes",
            "leb128_encode",
        }
    )

    def test_band_is_registered(self):
        assert set(REGISTRY.names()) >= self.NEW_OPS

    def test_band_categorized_byte(self):
        for op in self.NEW_OPS:
            assert REGISTRY.category_of(op) == "byte", f"{op} not in byte band"

    def test_unconditional_availability(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        available = set(REGISTRY.available(fuzzer, b"seed"))
        assert available >= self.NEW_OPS

    def test_every_op_has_a_handler(self):
        engine = OperatorEngine(_MockFuzzer())
        dispatch = REGISTRY.dispatch(engine)
        for name in self.NEW_OPS:
            assert callable(dispatch[name]), name

    def test_handlers_mutate_non_empty_input(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(self.NEW_OPS):
            buf = bytearray(b"ABCDEFGH")
            result = dispatch[name](buf, 0, bytes(buf))
            if name == "sort_bytes":
                # sort_bytes may mutate in place or return replacement.
                assert result is None or bytes(result) != b"ABCDEFGH"
            elif name == "insert_repeated_bytes":
                assert result is None or len(result) > len(buf)
            else:
                assert result is None or result != buf


class TestGoFuzzPorts:
    """Regression tests for mutations ported from ~/code/go-fuzz."""

    NEW_OPS = frozenset(
        {
            "ascii_num_replace",
            "splice_common_prefix",
            "corpus_literal_insert",
            "versifier_generate",
        }
    )

    def test_band_is_registered(self):
        assert set(REGISTRY.names()) >= self.NEW_OPS

    def test_band_categorized_structural(self):
        for op in self.NEW_OPS:
            assert REGISTRY.category_of(op) == "structural", f"{op} not in structural band"

    def test_unconditional_availability(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        available = set(REGISTRY.available(fuzzer, b"seed"))
        assert available >= self.NEW_OPS

    def test_every_op_has_a_handler(self):
        engine = OperatorEngine(_MockFuzzer())
        dispatch = REGISTRY.dispatch(engine)
        for name in self.NEW_OPS:
            assert callable(dispatch[name]), name

    def test_handlers_preserve_length(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(self.NEW_OPS - {"ascii_num_replace"}):
            for _ in range(10):
                buf = bytearray(os.urandom(512))
                result = dispatch[name](buf, 0, bytes(buf))
                assert result is None or len(result) == 512, name

    def test_ascii_num_replace_substitutes_digits(self):
        from fuzzer_tool.core.mutations import ascii_num_replace

        result = ascii_num_replace(b"err=12345;ok", rng=RandPool(seed=1))
        # The number token must be replaced with another numeric string.
        assert b"err=" in result
        assert b";ok" in result
        # The digits between = and ; must differ from the original.
        import re

        m = re.search(rb"err=(-?\d+);ok", result)
        assert m is not None, result
        assert m.group(1) != b"12345"

    def test_choose_len_prefers_short(self):
        from fuzzer_tool.core.mutations import choose_len

        rng = RandPool(seed=1)
        lengths = [choose_len(64, rng=rng) for _ in range(1000)]
        short = sum(1 for length in lengths if length <= 8)
        medium = sum(1 for length in lengths if 8 < length <= 32)
        assert short >= 800, f"expected >= 800 short, got {short}"
        assert medium >= 50, f"expected >= 50 medium, got {medium}"

    def test_splice_common_prefix_skips_small_diff(self):
        from fuzzer_tool.core.mutations import splice_common_prefix

        a = b"AAAAAA"
        b = b"AAAAAB"
        result = splice_common_prefix(a, b, rng=RandPool(seed=1))
        assert result == a

    def test_extract_corpus_literals(self):
        from fuzzer_tool.core.mutations import extract_corpus_literals

        int_lits, str_lits = extract_corpus_literals([b"count=1000;name=foo"])
        assert b"1000" in int_lits
        assert any(b"foo" in s for s in str_lits)

    def test_versifier_skips_binary(self):
        from fuzzer_tool.core.mutations.generic import _build_verse

        data = bytes(range(256))
        assert _build_verse(data, RandPool(seed=1)) is None

    def test_versifier_builds_from_text(self):
        from fuzzer_tool.core.mutations.generic import _build_verse

        # Versifier requires >= 90% printable ASCII (0x20-0x7E); newlines
        # are not counted, so use a delimiter that stays in-range.
        verse = _build_verse(b"a=1;b=2;c=3", RandPool(seed=1))
        assert verse is not None
        out = verse.Rhyme()
        assert isinstance(out, bytes)

    def test_splice_common_prefix_middle_aligned(self):
        from fuzzer_tool.core.mutations import splice_common_prefix

        a = b"prefix_" + b"A" * 64 + b"_suffix"
        b = b"prefix_" + b"B" * 64 + b"_suffix"
        rng = RandPool(seed=1)
        results = [splice_common_prefix(a, b, rng=rng) for _ in range(50)]
        assert any(r != a for r in results), "expected at least one non-trivial splice"
        for r in results:
            assert r.startswith(b"prefix_")
            assert r.endswith(b"_suffix")


class TestGoFuzzPorts2:
    """Regression tests for the remaining go-fuzz ports: digit_replace and
    insert_range_from_other."""

    NEW_OPS = frozenset({"digit_replace", "insert_range_from_other"})

    def test_band_is_registered(self):
        assert set(REGISTRY.names()) >= self.NEW_OPS

    def test_band_categorized_structural(self):
        for op in self.NEW_OPS:
            assert REGISTRY.category_of(op) == "structural", f"{op} not in structural band"

    def test_unconditional_availability(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        available = set(REGISTRY.available(fuzzer, b"seed"))
        assert available >= self.NEW_OPS

    def test_every_op_has_a_handler(self):
        engine = OperatorEngine(_MockFuzzer())
        dispatch = REGISTRY.dispatch(engine)
        for name in self.NEW_OPS:
            assert callable(dispatch[name]), name

    def test_handlers_preserve_length(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        # digit_replace preserves length; insert_range_from_other does not.
        for _ in range(10):
            buf = bytearray(os.urandom(512))
            result = dispatch["digit_replace"](buf, 0, bytes(buf))
            assert result is None or len(result) == 512

    def test_digit_replace_changes_one_digit(self):
        data = b"err=12345;ok"
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        buf = bytearray(data)
        result = dispatch["digit_replace"](buf, 0, data)
        assert result is None
        assert len(buf) == len(data)
        diffs = [i for i, (a, b) in enumerate(zip(data, buf, strict=True)) if a != b]
        assert len(diffs) == 1, f"expected exactly one digit changed, got {diffs}"
        assert 0x30 <= buf[diffs[0]] <= 0x39

    def test_digit_replace_skips_non_digit(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        buf = bytearray(b"abc")
        result = dispatch["digit_replace"](buf, 0, b"abc")
        assert result is None
        assert bytes(buf) == b"abc"

    def test_insert_range_from_other_skips_small_inputs(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer.corpus = [b"AAAA"]
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        # Buffer too small.
        buf = bytearray(b"abc")
        result = dispatch["insert_range_from_other"](buf, 0, b"abc")
        assert result is None
        # Corpus too small.
        fuzzer.corpus = [b"AAAA"]
        buf = bytearray(b"AAAA")
        result = dispatch["insert_range_from_other"](buf, 0, b"AAAA")
        assert result is None

    def test_insert_range_from_other_inserts_range(self):
        fuzzer = _MockFuzzer()
        fuzzer.max_len = 4096
        fuzzer.corpus = [b"prefix_" + b"A" * 64 + b"_suffix", b"prefix_" + b"B" * 64 + b"_suffix"]
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        buf = bytearray(b"AAAA")
        dispatch["insert_range_from_other"](buf, 0, b"AAAA")
        # Result should be longer than the original buffer.
        assert len(buf) > 4
        # The inserted bytes must come from one of the corpus entries.
        assert any(b"B" in buf or b"A" * 64 in buf for _ in [0])
