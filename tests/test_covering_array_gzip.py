"""Tests for core/mutations/covering_array_gzip.py -- covering_array_gzip."""

from __future__ import annotations

import gzip
import random

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core import field_spec
from fuzzer_tool.core.mutations.covering_array_gzip import (
    _FIELDS,
    HEADER_LEN,
    GzipCoveringArrayMutator,
)
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY

_MAGIC = b"\x1f\x8b"


class _Rng:
    def __init__(self, seed: int = 7):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


def _gz() -> bytes:
    return gzip.compress(b"hello world" * 10, mtime=0)


class TestRegistration:
    def test_registered(self):
        assert "covering_array_gzip" in REGISTRY.names()
        assert REGISTRY.category_of("covering_array_gzip") == "format"


class TestAvailability:
    def test_available_on_gzip(self):
        assert GzipCoveringArrayMutator().is_available(MutationContext(), _gz())

    def test_unavailable_on_other_and_empty(self):
        m = GzipCoveringArrayMutator()
        assert not m.is_available(MutationContext(), b"PK\x03\x04....")
        assert not m.is_available(MutationContext(), b"")

    def test_unavailable_on_bare_magic(self):
        assert not GzipCoveringArrayMutator().is_available(MutationContext(), _MAGIC)


class TestMutate:
    def test_declines_non_gzip_and_short(self):
        m = GzipCoveringArrayMutator()
        assert m.mutate(b"nope" * 8, _Rng()) is None
        assert m.mutate(_MAGIC + b"\x08", _Rng()) is None
        assert m.mutate(b"", _Rng()) is None

    def test_preserves_magic_and_body(self):
        m = GzipCoveringArrayMutator()
        data = _gz()
        out = m.mutate(data, _Rng())
        assert out is not None
        assert out[:2] == _MAGIC
        assert out[HEADER_LEN:] == data[HEADER_LEN:]
        assert len(out) == len(data)

    def test_max_len_truncates(self):
        out = GzipCoveringArrayMutator().mutate(_gz(), _Rng(), max_len=12)
        assert out is not None and len(out) <= 12

    def test_round_robin_covers_every_pair(self):
        # Falsification: full sweep must equal a verified pairwise array.
        m = GzipCoveringArrayMutator()
        data = _gz()
        rng = _Rng()
        rows = []
        first = m.mutate(data, rng)
        assert first is not None
        n_rows = len(m._rows)  # noqa: SLF001
        rows.append(field_spec.read_row(first, _FIELDS))
        for _ in range(n_rows - 1):
            out = m.mutate(data, rng)
            rows.append(field_spec.read_row(out if out is not None else data, _FIELDS))
        vs = field_spec.value_sets(list(_FIELDS))
        assert ca.verify_coverage(rows, vs, t=2)

    def test_adversarial_row_equal_to_input_returns_none(self):
        # An input whose header already equals the next row must not be
        # returned unchanged as a "mutation".
        m = GzipCoveringArrayMutator()
        data = _gz()
        assert m.mutate(data, _Rng()) != data
        m2 = GzipCoveringArrayMutator()
        m2.mutate(data, _Rng())
        row0 = m2._rows[0]  # noqa: SLF001
        same = field_spec.apply_row(data, _FIELDS, row0)
        m3 = GzipCoveringArrayMutator()
        m3._rows = m2._rows  # noqa: SLF001
        assert m3.mutate(same, _Rng()) is None
