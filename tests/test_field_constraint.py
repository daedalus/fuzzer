"""core/field_constraint: pair-relation solver, and the field_relation_* operators."""

from __future__ import annotations

import itertools
import random

import pytest

from fuzzer_tool.core import field_constraint as fc
from fuzzer_tool.core.field_constraint import Relation, polarity, relation_rows, solve
from fuzzer_tool.core.mutations import field_relation as fr
from fuzzer_tool.core.operator_registry import REGISTRY


def _brute(value_sets, relations, violate):
    out = []
    for row in itertools.product(*value_sets):
        if set(polarity(row, relations)) == set(violate):
            out.append(row)
    return out


def _random_problem(rng: random.Random):
    k = rng.randrange(2, 5)
    vs = [tuple(rng.sample(range(10), rng.randrange(1, 5))) for _ in range(k)]
    rels = []
    for n in range(rng.randrange(1, 5)):
        a, b = rng.sample(range(k), 2)
        mod = rng.randrange(2, 4)
        shift = rng.randrange(mod)
        rels.append(Relation(f"r{n}", a, b, lambda x, y, m=mod, s=shift: (x + y) % m == s))
    return vs, rels


@pytest.mark.parametrize("seed", range(80))
def test_solve_agrees_with_brute_force(seed):
    rng = random.Random(seed)
    vs, rels = _random_problem(rng)
    names = [r.name for r in rels]
    violate = tuple(n for n in names if rng.random() < 0.4)
    want = _brute(vs, rels, violate)
    got = solve(vs, rels, violate, limit=10_000, rng=random.Random(seed))
    assert sorted(got) == sorted(want)
    assert len(set(got)) == len(got)
    one = solve(vs, rels, violate, limit=1)
    assert bool(one) == bool(want)
    assert not one or one[0] in want


def test_unsatisfiable_polarity_is_empty():
    rels = [Relation("eq", 0, 1, lambda a, b: a == b)]
    assert solve([(1,), (2,)], rels, ()) == []
    assert solve([(1,), (2,)], rels, ("eq",)) == [(1, 2)]


def test_rows_have_exactly_the_requested_polarity():
    vs = [(0, 1, 2), (0, 1, 2), (0, 1, 2)]
    rels = [
        Relation("lt", 0, 1, lambda a, b: a < b),
        Relation("ne", 1, 2, lambda a, b: a != b),
    ]
    for row in relation_rows(vs, rels, violate=2, per_pattern=3, rng=random.Random(1)):
        assert polarity(row, rels) in {(), ("lt",), ("ne",), ("lt", "ne")}
    rows = relation_rows(vs, rels, violate=2, per_pattern=1)
    assert {polarity(r, rels) for r in rows} == {(), ("lt",), ("ne",), ("lt", "ne")}


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        solve([()], [])
    with pytest.raises(ValueError):
        solve([(1,), (2,)], [Relation("x", 0, 0, lambda a, b: True)])
    with pytest.raises(ValueError):
        solve([(1,), (2,)], [Relation("x", 0, 1, lambda a, b: True)], ("nope",))
    with pytest.raises(ValueError):
        solve([(1,), (2,)], [Relation("x", 0, 1, lambda a, b: True)] * 2)


def test_seeded_rows_are_reproducible():
    vs = [tuple(range(6))] * 4
    rels = [Relation(f"r{i}", i, i + 1, lambda a, b: a < b) for i in range(3)]
    a = relation_rows(vs, rels, rng=random.Random(9))
    b = relation_rows(vs, rels, rng=random.Random(9))
    assert a == b and a


def test_module_exports_stay_in_sync():
    assert callable(fc.solve) and callable(fc.relation_rows) and callable(fc.polarity)


# ── operators ───────────────────────────────────────────────────────


def _zip(**over: int) -> bytes:
    h = bytearray(b"PK\x03\x04" + bytes(26))
    for off, size, val in ((4, 2, 20), (8, 2, 8), (18, 4, 7), (22, 4, 7)):
        h[off : off + size] = val.to_bytes(size, "little")
    return bytes(h) + b"x" * 20


def _webp() -> bytes:
    h = bytearray(b"RIFF" + bytes(4) + b"WEBP" + b"VP8X" + bytes(16))
    h[16:20] = (10).to_bytes(4, "little")
    return bytes(h) + b"y" * 10


def _bmff() -> bytes:
    return (24).to_bytes(4, "big") + b"ftyp" + b"isom" + bytes(4) + b"isom" + b"mp42"


@pytest.mark.parametrize(
    ("cls", "relations", "fields", "sample"),
    [
        (fr.ZipRelationMutator, fr.ZIP_RELATIONS, fr.ZIP_FIELDS, _zip),
        (fr.WebpRelationMutator, fr.WEBP_RELATIONS, fr.WEBP_FIELDS, _webp),
        (fr.IsobmffRelationMutator, fr.ISOBMFF_RELATIONS, fr.ISOBMFF_FIELDS, _bmff),
    ],
)
def test_operator_rows_cover_every_single_violation(cls, relations, fields, sample):
    from fuzzer_tool.core import field_spec

    m = cls()
    data = sample()
    assert m.is_available(None, data)
    assert m.mutate(b"\x00" * 64, random.Random(1)) is None  # gate
    rng = random.Random(3)
    first = m.mutate(data, rng)
    assert m._rows
    seen = set()
    for _ in range(len(m._rows)):
        out = m.mutate(data, rng) or data
        row = field_spec.read_row(out, fields)
        seen.add(polarity(row, relations))
    for r in relations:
        assert (r.name,) in seen, f"{cls.name}: never violates {r.name} alone"
    assert () in seen
    assert first is None or len(first) == len(data)


@pytest.mark.parametrize(
    "name", ["field_relation_zip", "field_relation_webp", "field_relation_isobmff"]
)
def test_operators_registered(name):
    assert name in REGISTRY.names()
