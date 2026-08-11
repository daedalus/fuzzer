"""Regression tests: operator registry is the single source of truth.

Covers the dispatcher refactor: every mutation operator is registered exactly
once in ``fuzzer_tool.core.operator_registry.REGISTRY``; the legacy lists in
``core/mutations/generic.py``, the runtime dispatch table, and
``OPERATOR_CATEGORIES`` all derive from it. Guards the two historical drift
instances (``colorization`` registered with no op-list entry,
``block_shuffle_variable`` categorized nowhere) from returning.
"""

import os

import pytest

from fuzzer_tool.core.mutations import DICT_MUTATIONS, FORMAT_MUTATIONS, MUTATIONS
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import (
    REGISTRY,
    OperatorRegistry,
    OperatorSpec,
    _sniff_der,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

# Ops that are dispatchable but intentionally live outside the legacy lists:
# gated by per-run conditions (markov/cem/grammar/cmplog/redqueen) or
# dispatch-only and never selectable (colorization).
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

    def test_colorization_never_selectable(self):
        assert "colorization" not in self._available()

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
        """Only the corpus-measuring operator is gated."""
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
