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

from collections.abc import Callable
from dataclasses import dataclass

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
        "crossover",
        "type_replace",
        "ascii_num",
        "ascii_num_arithmetic",
        "insert_ascii_num",
        "tlv_mutate",
        "token_shuffle",
        "chunk_shuffle",
        "punctuation_insert",
        "special_strings",
        "magic_values",
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
    },
    "adaptive": {
        "markov_bytes",
        "cem_bytes",
        "colorization",
        "skipdet_probe",
        "auto_extras",
        "redqueen_xform",
        "gradient_cmp",
        "redqueen",
        "havoc",
        "overwrite_copy",
        "overwrite_fixed",
        "clone_fixed",
        "regex_bomb",
        "grammar_mutate",
        "grammar_tree_mutate",
    },
}


def _has_cmplog_pairs(fuzzer, _data) -> bool:
    return bool(getattr(fuzzer, "_cmplog", None) and fuzzer._cmplog.pairs)


def _redqueen_available(fuzzer, data) -> bool:
    seed_meta = getattr(fuzzer, "seed_meta", None)
    parent_meta = seed_meta.get(data) if hasattr(seed_meta, "get") else None
    return bool(
        parent_meta and (parent_meta.get("redqueen_matches") or parent_meta.get("redqueen_offsets"))
    )


def _never(_fuzzer, _data) -> bool:
    # colorization is dispatchable but never selectable (keeps historic
    # build_ops behavior where it is not returned by available ops).
    return False


# Availability predicates mirror the historic build_ops() conditions.
_AVAILABLE: dict[str, Callable[[object, bytes], bool] | None] = {
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
    # per-input ops
    "redqueen": _redqueen_available,
    # flag-gated base op
    "regex_bomb": lambda f, _d: bool(getattr(f, "enable_regex_bomb", False)),
    # dispatch-only, never selectable
    "colorization": _never,
}


@dataclass(frozen=True)
class OperatorSpec:
    """Static registration metadata for one mutation operator."""

    name: str
    category: str
    handler_name: str
    available: Callable[[object, bytes], bool] | None = None


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

    def names(self) -> list[str]:
        """All registered operator names, in registration order."""
        return list(self._ops)

    def dispatch(self, engine) -> dict[str, Callable]:
        """Map every registered operator name to its handler on *engine*.

        Raises AttributeError if an operator is registered but has no
        ``_op_<name>`` handler — registering without a handler is a bug.
        """
        return {name: getattr(engine, spec.handler_name) for name, spec in self._ops.items()}

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
