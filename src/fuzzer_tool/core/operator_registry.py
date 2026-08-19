"""Operator dispatcher: single source of truth for mutation operators.

Every mutation operator (name, category, handler method, availability
predicate) is registered here exactly once. All consumers derive from
``REGISTRY`` instead of maintaining their own lists:

- ``OperatorEngine.build_dispatch()`` -> ``REGISTRY.dispatch(engine)``
- ``OperatorEngine.build_ops(data)``  -> ``REGISTRY.available(fuzzer, data)``
- ``core/operator_categories.OPERATOR_CATEGORIES`` -> ``REGISTRY.categories()``
- ``Fuzzer._register_arms()``         -> ``REGISTRY.names()``

Schedulers learn which operators exist through the services layer, which
queries this dispatcher; they never hardcode operator lists themselves.
Adding an operator means one registration here plus a ``_op_<name>`` handler
on ``OperatorEngine`` — nothing else.
"""

import contextlib
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Category taxonomy, kept here so it lives next to the operators it classifies.
# Every operator must appear in exactly one category set.
_CATEGORIES: dict[str, set[str]] = {
    "bit": {
        "bit_flip",
        "bit_offset_flip",
        "bit_offset_span",
        "bit_transpose_8",
        "bit_transpose_16",
        "bit_transpose_32",
        "bit_transpose_64",
    },
    "byte": {
        "byte_flip",
        "interesting_8",
        "interesting_16",
        "interesting_32",
        "arithmetic",
        "random_bytes",
        "radamsa_num",
        "byte_shuffle",
        "byte_delete",
        "byte_insert",
        "swap_bytes",
        "endianness_swap",
        "insert_repeated_bytes",
        "sort_bytes",
        "leb128_encode",
    },
    "block": {
        "block_insert",
        "block_delete",
        "block_duplicate",
        "swap_regions",
        "repeat_clone",
        "truncate",
        "length_grow",
        "length_shrink",
        "length_boundary",
        "transpose_16",
        "transpose_32",
        "transpose_64",
        "simd_boundary",
        "block_shuffle_variable",
    },
    "dict": {
        "dict_insert",
        "dict_replace",
        "dict_overwrite",
        "dict_prepend",
        "dict_append",
        "checksum_repair",
        "token_dup",
        "dict_compound",
    },
    "structural": {
        "splice",
        "splice_diff_located",
        "elite_fuse",
        "crossover",
        "type_replace",
        "ascii_num",
        "ascii_num_arithmetic",
        "insert_ascii_num",
        "ascii_num_replace",
        "digit_replace",
        "tlv_mutate",
        "token_shuffle",
        "chunk_shuffle",
        "punctuation_insert",
        "special_strings",
        "magic_values",
        "splice_common_prefix",
        "corpus_literal_insert",
        "insert_range_from_other",
        "versifier_generate",
    },
    "radamsa": {
        "fuse_this",
        "fuse_next",
        "fuse_old",
        "tree_mutate",
        "line_mutate",
        "utf8_widen",
        "utf8_insert",
    },
    "format": {
        "png_chunk_mutate",
        "png_crc_fix",
        "jpeg_chunk_mutate",
        "jpeg_crc_fix",
        "bmp_chunk_mutate",
        "gzip_chunk_mutate",
        "zlib_chunk_mutate",
        "format_lock",
        "pgs_chunk_mutate",
        "isobmff_chunk_mutate",
        "nal_chunk_mutate",
        "protobuf_chunk_mutate",
        "gif_chunk_mutate",
        "webp_chunk_mutate",
        "webm_chunk_mutate",
        "zip_chunk_mutate",
        "x86_chunk_mutate",
        "arm_chunk_mutate",
        "elf_chunk_mutate",
        "recompress_zlib",
        "recompress_gzip",
        "field_repair",
        "tlv_nest_mutate",
        "der_len_mutate",
        "der_tag_mutate",
        "der_tlv_reorder",
        "der_tlv_insert",
    },
    # Constructive inverses of the diehard/dieharder statistical tests: each
    # one builds a buffer whose test statistic sits in a tail the uniform
    # null essentially never reaches. Kept out of "structural" (which is
    # about a format's own structure) because these impose a *statistical*
    # regularity that is format-independent -- see mutations/structured.py.
    "regularity": {
        "gcd_worst_case",
        "monotone_fill",
        "kmer_saturate",
        "kmer_saturate_bits",
        "kmer_starve",
        "rank_deficient",
        "perm_lock",
        "lag_correlate",
        "spectral_peak",
        "birthday_collide",
        "invariant_break",
        "degenerate_geometry",
        "float_squeeze",
        "popcount_lock",
    },
    "adaptive": {
        "markov_bytes",
        "cem_bytes",
        "colorization",
        "skipdet_probe",
        "auto_extras",
        "redqueen_xform",
        "gradient_cmp",
        "gradient_descent",
        "condstmt_solve",
        "magic_byte_search",
        "climb_hill",
        "path_negate",
        "length_offset_goal",
        "redqueen",
        "havoc",
        "overwrite_copy",
        "overwrite_fixed",
        "clone_fixed",
        "regex_bomb",
        "grammar_mutate",
        "grammar_tree_mutate",
        "crc_learn",
    },
}


def _has_cmplog_pairs(fuzzer, _data) -> bool:
    return bool(getattr(fuzzer, "_cmplog", None) and fuzzer._cmplog.pairs)


def _has_branch_records(fuzzer, _data) -> bool:
    """Needs recorded outcomes *and* an enabled solver (--path-negation)."""
    cmplog = getattr(fuzzer, "_cmplog", None)
    if not (cmplog and getattr(cmplog, "_pair_cmp", None)):
        return False
    return getattr(fuzzer, "_path_solver", None) is not None


def _redqueen_available(fuzzer, data) -> bool:
    seed_meta = getattr(fuzzer, "seed_meta", None)
    parent_meta = seed_meta.get(data) if hasattr(seed_meta, "get") else None
    return bool(
        parent_meta and (parent_meta.get("redqueen_matches") or parent_meta.get("redqueen_offsets"))
    )


# corpus_invariants() defaults to this many samples before it will call an
# offset invariant; below it, every offset looks fixed by coincidence and the
# operator would scribble over the whole file.
_INVARIANT_MIN_SAMPLES = 16


def _has_corpus_samples(fuzzer, _data) -> bool:
    corpus = getattr(fuzzer, "corpus", None)
    return bool(corpus) and len(corpus) >= _INVARIANT_MIN_SAMPLES


# ── Format-relevance gating ────────────────────────────────────────────
#
# The format operators (png/jpeg/webm/...) parse the input and, when it is
# NOT that format, fall back to generating a whole random file of that
# format from scratch. That fallback is a legitimate bootstrap when the
# target really does parse that format and the corpus has not picked one up
# yet — but when the target has nothing to do with the format (fuzzing a
# text parser with the jpeg operator enabled), it burns a large share of the
# execution budget building random JPEGs that the target rejects instantly.
#
# Measured on a non-image target: format operators consumed ~50% of total
# runtime, and gating them doubled throughput (2213 -> 4469 eps).
#
# The gate keeps both cases working:
#   * once any input has parsed as format F, F is "live" and stays fully
#     available for the rest of the run (real files must be mutable);
#   * until then F is still offered, but only on a small fraction of
#     selections, so bootstrap-from-garbage-corpus still happens — it just
#     no longer dominates the budget.
def _sniff_der(d: bytes) -> bool:
    """True when d starts like a BER/DER structure: SEQUENCE/SET tag plus a
    plausible length byte (short form, indefinite, or long form with <= 4
    length bytes)."""
    return (
        len(d) >= 2
        and d[0] in (0x30, 0x31)
        and (d[1] < 0x80 or d[1] in (0x80, 0x81, 0x82, 0x83, 0x84))
    )


# ── Format gating for the "regularity" (statistical) mutation band ───────
#
# Most regularity operators (kmer_saturate, birthday_collide, ...) impose a
# distributional property that is meaningful for any byte stream, so they
# stay unconditionally available. A few instead target a structural property
# that only certain formats have, and burn budget building it into inputs
# that can't use it -- see docs/TODO.md's "Per-format tuning of the
# regularity band" item. These three get the same sniffer-gate treatment as
# the format band above:
#   * spectral_peak imposes a strong frequency-domain peak, relevant to
#     DCT/DST-transform-coded data (JPEG stills, and MP4/WebM containers,
#     which typically carry AVC/HEVC/VP8/VP9 video using the same transform);
#   * degenerate_geometry constructs near-collinear/near-coplanar point
#     sets, relevant to vector/mesh geometry formats (SVG, STL);
#   * rank_deficient constructs a rank-deficient matrix, relevant to
#     erasure-coded formats -- RAR's recovery-record feature is the only
#     erasure-coded format this project targets (unrar_read.c).
def _sniff_dct_transform_coded(d: bytes) -> bool:
    return (
        d[:2] == b"\xff\xd8"  # JPEG
        or d[4:8] == b"ftyp"  # MP4/ISOBMFF
        or d[:4] == b"\x1a\x45\xdf\xa3"  # WebM/Matroska
    )


def _sniff_mesh_or_vector_geometry(d: bytes) -> bool:
    if d[:5] == b"<?xml" or d[:4] == b"<svg":  # SVG
        return True
    if d[:5] == b"solid":  # ASCII STL
        return True
    # Binary STL: 80-byte header, then a uint32 triangle count, then exactly
    # count * 50 bytes of triangle data -- no magic bytes, so check the
    # length invariant instead.
    if len(d) >= 84:
        (tri_count,) = struct.unpack_from("<I", d, 80)
        if len(d) == 84 + tri_count * 50:
            return True
    return False


def _sniff_rar(d: bytes) -> bool:
    return d[:6] == b"Rar!\x1a\x07"


_FORMAT_SNIFFERS: dict[str, Callable[[bytes], bool]] = {
    "png_chunk_mutate": lambda d: d[:8] == b"\x89PNG\r\n\x1a\n",
    "png_crc_fix": lambda d: d[:8] == b"\x89PNG\r\n\x1a\n",
    "jpeg_chunk_mutate": lambda d: d[:2] == b"\xff\xd8",
    "jpeg_crc_fix": lambda d: d[:2] == b"\xff\xd8",
    "bmp_chunk_mutate": lambda d: d[:2] == b"BM",
    "gzip_chunk_mutate": lambda d: d[:2] == b"\x1f\x8b",
    "zlib_chunk_mutate": lambda d: len(d) > 1 and d[0] == 0x78,
    "gif_chunk_mutate": lambda d: d[:3] == b"GIF",
    "webp_chunk_mutate": lambda d: d[:4] == b"RIFF" and d[8:12] == b"WEBP",
    "webm_chunk_mutate": lambda d: d[:4] == b"\x1a\x45\xdf\xa3",
    "zip_chunk_mutate": lambda d: d[:2] == b"PK",
    "isobmff_chunk_mutate": lambda d: d[4:8] == b"ftyp",
    "pgs_chunk_mutate": lambda d: d[:2] == b"PG",
    "nal_chunk_mutate": lambda d: (
        d[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\x00") or d[:3] == b"\x00\x00\x01"
    ),
    # ELF: magic + a valid class/endianness byte pair. Cheap enough to run on
    # every selection; the mutator itself never touches non-ELF input.
    "elf_chunk_mutate": lambda d: (
        len(d) >= 64 and d[:4] == b"\x7fELF" and d[4] in (1, 2) and d[5] in (1, 2)
    ),
    # Recompression operators reuse the same header checks as the in-place
    # zlib/gzip mutators, but additionally require enough bytes for a trailer
    # — there is nothing to round-trip through an empty stream.
    "recompress_zlib": lambda d: (
        len(d) >= 6 and (d[0] & 0x0F) == 8 and ((d[0] << 8) | d[1]) % 31 == 0
    ),
    "recompress_gzip": lambda d: len(d) >= 18 and d[:3] == b"\x1f\x8b\x08",
    # Only formats with modelled derived fields; PNG for now.
    "field_repair": lambda d: len(d) >= 8 and d[:8] == b"\x89PNG\r\n\x1a\n",
    # BER/DER: a SEQUENCE (0x30) / SET (0x31) leading tag with a plausible
    # length byte — short form, indefinite (BER), or long form with <= 4
    # length bytes. Covers X.509 / EC-key / ECDSA-signature material.
    "der_len_mutate": _sniff_der,
    "der_tag_mutate": _sniff_der,
    "der_tlv_reorder": _sniff_der,
    "der_tlv_insert": _sniff_der,
    # regularity band: format-shaped ops (see comment above their sniffers)
    "spectral_peak": _sniff_dct_transform_coded,
    "degenerate_geometry": _sniff_mesh_or_vector_geometry,
    "rank_deficient": _sniff_rar,
}

# Fraction of selections on which a not-yet-seen format is still offered.
_FORMAT_BOOTSTRAP_RATE = 0.02


def _format_available(name: str) -> Callable[[object, bytes], bool]:
    sniff = _FORMAT_SNIFFERS[name]

    def _check(fuzzer, data) -> bool:
        if fuzzer is None:
            return True
        live = getattr(fuzzer, "_live_formats", None)
        if live is None:
            live = set()
            with contextlib.suppress(AttributeError):
                fuzzer._live_formats = live
        # Check live first: once a format has been confirmed live, every
        # later call for that op is a plain set lookup instead of re-running
        # sniff() (a struct.unpack for STL, a chained byte-prefix check for
        # others) on every exec -- and live is the steady-state case for the
        # rest of a real fuzzing run once any matching seed has been seen.
        # Equivalent to the old sniff-first order: sniff() re-matching an
        # already-live format was always a same-result no-op add() before.
        # Docs/TODO.md's "REGISTRY.available() re-evaluates data-independent
        # predicates per exec" lever is about all 51 gated ops broadly; this
        # is a first, narrower cut at the ~27 sniffer-gated ones specifically.
        if name in live:
            return True
        if data and sniff(data):
            live.add(name)  # real file of this format seen — keep it live
            return True
        # Never seen this format: keep a thin bootstrap trickle so a target
        # that does parse it can still be reached from a garbage corpus.
        rng = getattr(fuzzer, "_rand_pool", None)
        if rng is None:
            return True
        try:
            return float(rng.random()) < _FORMAT_BOOTSTRAP_RATE
        except (TypeError, ValueError):
            # Mock/stub fuzzer in tests — stay permissive rather than
            # silently hiding operators.
            return True

    return _check


def format_gate_matches(name: str, data: bytes) -> bool | None:
    """Did *name*'s own format sniffer match *data*?

    ``None`` for an operator that is not sniffer-gated -- it is applicable
    to anything, which is a different answer from "no".

    Exists because availability and applicability are not the same thing,
    and per-operator success rates were quietly dividing by the wrong one.
    An operator is *offered* on inputs it cannot possibly act on, by two
    separate mechanisms in ``_format_available``: the bootstrap trickle
    offers a never-seen format on non-matching input, and -- much larger on
    a mixed corpus -- once a format has been seen once, ``name in live``
    short-circuits and the operator is offered on *every* input for the
    rest of the run, matching or not. One PNG in the corpus makes
    png_chunk_mutate available on every JPEG thereafter.

    Both are deliberate: the gate is about whether a format is relevant to
    this target, and the mutators decline on input they cannot parse. But
    counting those selections in an operator's denominator measures the
    corpus, not the operator. On a corpus containing none of its format an
    operator's reported rate is essentially the trickle rate, which says
    nothing about whether it works.
    """
    sniff = _FORMAT_SNIFFERS.get(name)
    if sniff is None:
        return None
    if not data:
        return False
    try:
        return bool(sniff(data))
    except Exception:  # pragma: no cover - a sniffer must not break accounting
        return False


# Availability predicates mirror the historic build_ops() conditions.
_AVAILABLE: dict[str, Callable[[object, bytes], bool] | None] = {
    # format ops — gated on the format being relevant to this target
    **{n: _format_available(n) for n in _FORMAT_SNIFFERS},
    # dictionary ops — only when a dictionary is loaded
    "dict_insert": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "dict_replace": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "dict_overwrite": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "dict_prepend": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "dict_append": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "checksum_repair": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "token_dup": lambda f, _d: bool(getattr(f, "dictionary", None)),
    "dict_compound": lambda f, _d: bool(getattr(f, "dictionary", None)),
    # model / engine-gated ops
    "markov_bytes": lambda f, _d: bool(getattr(f, "markov_trained", False)),
    "cem_bytes": lambda f, _d: bool(
        getattr(f, "mc", None)
        and getattr(f, "mc_cem", False)
        and getattr(f.mc, "cem_fitted", False)
    ),
    "grammar_mutate": lambda f, _d: bool(getattr(f, "grammar", None)),
    "grammar_tree_mutate": lambda f, _d: bool(getattr(f, "grammar", None)),
    "redqueen_xform": _has_cmplog_pairs,
    "gradient_cmp": _has_cmplog_pairs,
    "gradient_descent": _has_cmplog_pairs,
    "condstmt_solve": _has_cmplog_pairs,
    "magic_byte_search": _has_cmplog_pairs,
    "climb_hill": _has_cmplog_pairs,
    # Needs recorded comparison *outcomes*, not just operand pairs: the
    # shim only emits the result field in trace mode, and without it there
    # is no predicate to negate.
    "path_negate": _has_branch_records,
    # per-input ops
    "redqueen": _redqueen_available,
    # regularity op that measures the corpus rather than the seed
    "invariant_break": _has_corpus_samples,
    # learned checksum model (gated on a recovered GF(2) polynomial OR a
    # recovered integer-modulus model -- an Adler/Fletcher target would
    # otherwise recover a model the operator could never use)
    "crc_learn": lambda f, _d: bool(
        getattr(f, "checksum_learner", None) and f.checksum_learner.ensure_model()
    ),
    # flag-gated base op
    "regex_bomb": lambda f, _d: bool(getattr(f, "enable_regex_bomb", False)),
    "x86_chunk_mutate": lambda f, _d: bool(getattr(f, "enable_x86_mutator", False)),
    "arm_chunk_mutate": lambda f, _d: bool(getattr(f, "enable_arm_mutator", False)),
    # dispatch-only, never selectable
    # colorization: gated on cmplog pairs. The handler is a byte randomizer
    # that prefers offsets appearing in comparison operands (CmplogColorizer),
    # and without pairs it degrades to picking offsets at random -- weaker
    # than havoc, which already covers that. Gating it on the signal it needs
    # is what makes it worth a selection slot.
    #
    # It was `_never` before, which the comment justified only as "keeps
    # historic build_ops behavior". That behaviour was an accident, not a
    # decision: colorization sat in the dispatch table and in no op list, so
    # nothing could ever draw it, and the registry refactor preserved the
    # accident faithfully enough to pin it with a test.
    #
    # Note this operator is not colorization in the AFL++ sense -- it does not
    # hold the execution path fixed, because a mutation operator has no way to
    # execute anything. The real pass is core/colorization.py, driven by
    # Fuzzer._colorize_seed() under --colorize.
    "colorization": lambda f, _d: bool(getattr(getattr(f, "_cmplog", None), "pairs", None)),
}


def _mutator_adapter(mutator, engine) -> Callable:
    """Adapt MutatorBase.mutate() to the `_op_*` handler signature.

    Handlers are called as ``(buf, byte_idx, data)`` and return a
    bytearray replacement or None. MutatorBase works in bytes and takes
    the rng explicitly, so the adapter bridges the two and applies the
    max_len clamp centrally -- implementations must not be trusted to do
    it themselves, since an operator that silently grows the buffer past
    max_len is exactly the class of bug that has bitten this codebase
    before.
    """

    def _handler(buf, _byte_idx, data):
        f = engine.f
        max_len = getattr(f, "max_len", 0)
        try:
            result = mutator.mutate(bytes(buf), f._rand_pool, max_len=max_len, fuzzer=f)
        except Exception:  # noqa: BLE001 - third-party mutator
            log.warning("mutator %r raised during mutate()", mutator, exc_info=True)
            return None
        if not result or result == bytes(buf):
            return None
        return bytearray(result[:max_len] if max_len else result)

    _handler.__name__ = f"_op_{mutator.name}"
    return _handler


@dataclass(frozen=True)
class OperatorSpec:
    """Static registration metadata for one mutation operator.

    ``handler_name`` names a ``_op_<name>`` method on ``OperatorEngine``
    for the function-based operators (all the built-ins). ``mutator``
    is set instead for class-based mutators registered through
    ``register_mutator()``; exactly one of the two applies, and
    ``dispatch()`` resolves whichever is present.
    """

    name: str
    category: str
    handler_name: str
    available: Callable[[object, bytes], bool] | None = None
    mutator: object | None = None


class OperatorRegistry:
    """Registry/dispatcher of mutation operators (single source of truth)."""

    def __init__(self) -> None:
        self._ops: dict[str, OperatorSpec] = {}
        self._categories_cache: dict[str, set[str]] | None = None

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._ops:
            raise ValueError(f"duplicate operator registration: {spec.name!r}")
        self._ops[spec.name] = spec
        self._categories_cache = None

    def register_mutator(self, mutator) -> None:
        """Register a ``MutatorBase`` instance as an operator.

        Lets a self-contained mutator class join the operator table without
        adding a ``_op_<name>`` method to ``OperatorEngine`` — see
        ``core/mutator_interface``. Existing function-based operators are
        unaffected; the two kinds coexist in one table and schedulers
        cannot tell them apart, which is the point.
        """
        name = getattr(mutator, "name", "")
        if not name:
            raise ValueError(f"mutator {mutator!r} has no name")
        if not callable(getattr(mutator, "mutate", None)):
            raise TypeError(f"mutator {name!r} has no callable mutate()")
        self.register(
            OperatorSpec(
                name=name,
                category=getattr(mutator, "category", "adaptive"),
                handler_name="",  # resolved via the mutator, not the engine
                available=lambda f, d, _m=mutator: _m.is_available(f, d),
                mutator=mutator,
            )
        )

    def mutators(self) -> list[object]:
        """Every registered ``MutatorBase`` instance (not function ops)."""
        return [s.mutator for s in self._ops.values() if s.mutator is not None]

    def notify_new_coverage(self, seed: bytes, new_edges: int) -> None:
        """Fan out a new-coverage event to every registered mutator.

        Exceptions are contained: a misbehaving third-party mutator must
        not take down the fuzz loop over a feedback hook.
        """
        for mutator in self.mutators():
            try:
                mutator.on_new_coverage(seed, new_edges)
            except Exception:  # noqa: BLE001 - third-party hook
                log.warning("on_new_coverage failed for %r", mutator, exc_info=True)

    def names(self) -> list[str]:
        """All registered operator names, in registration order."""
        return list(self._ops)

    def dispatch(self, engine) -> dict[str, Callable]:
        """Map every registered operator name to its handler on *engine*.

        Function-based operators resolve to their ``_op_<name>`` method;
        raises AttributeError if one is registered without a handler,
        since that is a bug. Class-based mutators resolve to an adapter
        that presents the same ``(buf, byte_idx, data)`` signature.
        """
        table: dict[str, Callable] = {}
        for name, spec in self._ops.items():
            if spec.mutator is not None:
                table[name] = _mutator_adapter(spec.mutator, engine)
            else:
                table[name] = getattr(engine, spec.handler_name)
        return table

    def available(self, fuzzer, data: bytes) -> list[str]:
        """Operator names whose availability predicate passes for *fuzzer*.

        Mirrors the historic build_ops() conditions (dictionary, markov,
        cem, grammar, cmplog, per-input redqueen).
        """
        return [
            name
            for name, spec in self._ops.items()
            if spec.available is None or spec.available(fuzzer, data)
        ]

    def categories(self) -> dict[str, set[str]]:
        """Derived category -> operator-names taxonomy."""
        if self._categories_cache is None:
            cats: dict[str, set[str]] = {}
            for spec in self._ops.values():
                cats.setdefault(spec.category, set()).add(spec.name)
            self._categories_cache = cats
        return self._categories_cache

    def category_of(self, name: str) -> str:
        return self._ops[name].category


REGISTRY = OperatorRegistry()

# Deterministic registration order: category-band order (the historic taxonomy
# band order, bit first) with operator names sorted within each band. Set
# iteration order is hash-dependent, so registration must not rely on it —
# REGISTRY.names()/available() ordering feeds scheduler arm creation and the
# op list handed to select_op, which must be reproducible for a fixed seed.
for _cat in _CATEGORIES:
    for _op in sorted(_CATEGORIES[_cat]):
        REGISTRY.register(
            OperatorSpec(
                name=_op,
                category=_cat,
                handler_name=f"_op_{_op}",
                available=_AVAILABLE.get(_op),
            )
        )
