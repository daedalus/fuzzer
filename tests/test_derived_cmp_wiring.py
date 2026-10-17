"""--derived-cmp: GF(2), additive and byte-map families in the --affine-mod search.

Targets are Python models of targets/compcov_gates.c and targets/gf2_gates.c;
each probe reports compares as cmplog does, ``(v, t)`` little-endian, behind
the constant pairs the shim logs on every run.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services.affine_mod import DERIVED_SCREEN_BYTES, AffineMod, Families
from fuzzer_tool.services.fuzzer import Fuzzer
from tests import test_commands_extended

SEED = bytes(range(1, 65))

# What the shim logs before any target compare (seen on compcov_gates_cmp.so).
NOISE = [
    (b"\x00\x00\x00\x00", b"\x01\x00\x00\x00"),
    (b"\x00\x00\x00\x00", b"\x02\x00\x00\x00"),
    (b"\x40" + bytes(7), b"\x1d" + bytes(7)),
    (b"\x05\x00\x00\x00", b"\x09\x00\x00\x00"),
    (b"\x07\x00", b"\x08\x00"),
]

SYNDROME = (
    0x7DE9FC02, 0x7EE9B8C9, 0x47682BEB, 0x0A7DC563,
    0x7ADD219D, 0xB07E1AF5, 0xB4D4D8FB, 0x3798A188,
    0x78D9D9B6, 0x00EE6FC7, 0xF9728652, 0x268251C9,
    0x7EEAACFD, 0x9C27CC92, 0x0423E37D, 0x45E9DA34,
)  # fmt: skip
G3_MASK, G3_TARGET = 0x2B91, 0xC3A5
SUM_TARGET = 0x0A57
HELLO = b"HelloWorld!!"
STR_OFF = 33


def _syndrome(data: bytes) -> int:
    x = int.from_bytes(data[7:11], "little")
    s = 0
    for j, row in enumerate(SYNDROME):
        s |= (bin(x & row).count("1") & 1) << j
    return s ^ G3_MASK


def _sum16(data: bytes) -> int:
    return sum(data[13:29]) & 0xFFFF


def _rot13(c: int) -> int:
    for lo in (0x41, 0x61):
        if lo <= c < lo + 26:
            return (c - lo + 13) % 26 + lo
    return c


def _rot(data: bytes) -> bytes:
    return bytes(_rot13(c) for c in data[STR_OFF : STR_OFF + len(HELLO)])


def _avalanche(data: bytes) -> int:
    h = int.from_bytes(data[29:33], "little") * 0x85EBCA6B & 0xFFFFFFFF
    h ^= h >> 15
    h = h * 0xC2B2AE35 & 0xFFFFFFFF
    return h ^ h >> 13


def _poly(data: bytes) -> int:
    x = int.from_bytes(data[41:45], "little")
    return (x * x + 3 * x + 7) & 0xFFFFFFFF


def _quad16(x: int) -> int:
    r = (x << 3 | x >> 13) & 0xFFFF
    return (x & r) ^ (x >> 5) ^ 0xA5C3


def _quad(data: bytes) -> int:
    return _quad16(int.from_bytes(data[46:48], "little"))


def _cubic(data: bytes) -> int:
    x = int.from_bytes(data[46:48], "little")
    return x & (x >> 1) & (x >> 2) ^ 0x1234


POLY_TARGET = (0x13572468**2 + 3 * 0x13572468 + 7) & 0xFFFFFFFF
QUAD_TARGET = _quad16(0xBEEF)


def _u32(v: int) -> bytes:
    return v.to_bytes(4, "little")


def _probe(*gates):
    """Probe logging NOISE, then ``(v(data), t)`` for each gate."""

    def probe(data: bytes):
        return NOISE + [(fn(data), t) for fn, t in gates]

    return probe


SYND_GATE = (lambda d: _u32(_syndrome(d)), _u32(G3_TARGET))
SUM_GATE = (lambda d: _u32(_sum16(d)), _u32(SUM_TARGET))
ROT_GATE = (_rot, HELLO)
HASH_GATE = (lambda d: _u32(_avalanche(d)), _u32(0x46044640))
POLY_GATE = (lambda d: _u32(_poly(d)), _u32(POLY_TARGET))
QUAD_GATE = (lambda d: _quad(d).to_bytes(2, "little"), QUAD_TARGET.to_bytes(2, "little"))
CUBIC_GATE = (lambda d: _cubic(d).to_bytes(2, "little"), (0x1234 ^ 0x0101).to_bytes(2, "little"))


def _search(gate, families, max_execs=400):
    svc = AffineMod(_probe(gate), max_execs, families)
    svc.search(SEED)
    out = []
    while (cand := svc.pop()) is not None:
        out.append(cand)
    return svc, out


# ── families ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("gate", "holds"),
    [
        (SYND_GATE, lambda d: _syndrome(d) == G3_TARGET),
        (SUM_GATE, lambda d: _sum16(d) == SUM_TARGET),
        (ROT_GATE, lambda d: _rot(d) == HELLO),
        (POLY_GATE, lambda d: _poly(d) == POLY_TARGET),
        (QUAD_GATE, lambda d: _quad(d) == QUAD_TARGET),
    ],
    ids=["gf2_syndrome", "additive_sum", "bytemap_rot13", "poly_mod", "gf2_quadratic"],
)
def test_derived_family_queues_a_satisfying_input(gate, holds):
    svc, cands = _search(gate, Families.DERIVED)

    assert any(holds(c) for c in cands)
    assert svc.stats["solved"] >= 1
    assert all(len(c) == len(SEED) for c in cands)


@pytest.mark.parametrize(
    "gate",
    [SYND_GATE, SUM_GATE, ROT_GATE, POLY_GATE, QUAD_GATE],
    ids=["gf2", "sum", "rot13", "poly", "quad"],
)
def test_modular_family_alone_does_not_solve_derived_gates(gate):
    """Falsification: without --derived-cmp these gates stay closed."""
    svc, cands = _search(gate, Families.MODULAR)
    assert "solved" not in svc.stats


def test_avalanche_hash_queues_nothing():
    """Adversarial: compcov G4 is no family; fail closed within budget."""
    svc, cands = _search(HASH_GATE, Families.DERIVED, max_execs=400)
    assert "solved" not in svc.stats
    assert svc.stats["execs"] <= 400


def test_poly_and_quad_families_are_counted():
    svc, _ = _search(POLY_GATE, Families.DERIVED)
    assert svc.stats["poly"] >= 1
    svc, _ = _search(QUAD_GATE, Families.DERIVED)
    assert svc.stats["gf2q"] >= 1


def test_cubic_gf2_gate_is_not_fitted_as_quadratic():
    """Adversarial: a degree-3 map must fail the quadratic confirms."""
    svc, _ = _search(CUBIC_GATE, Families.DERIVED)
    assert "gf2q" not in svc.stats
    assert svc.stats["execs"] <= 400


def test_derived_search_respects_its_exec_budget():
    svc, _ = _search(ROT_GATE, Families.DERIVED, max_execs=20)
    assert svc.stats["execs"] <= 20


def test_solution_keeps_bytes_outside_the_field():
    _, cands = _search(SYND_GATE, Families.DERIVED)
    good = [c for c in cands if _syndrome(c) == G3_TARGET]
    assert good[0][:7] == SEED[:7]
    assert good[0][11:] == SEED[11:]


def test_regression_noise_pairs_do_not_crowd_out_the_gate():
    """MAX_PAIRS capped candidates before the screen: five constant shim pairs
    filled it and the real compare was never screened."""

    def mod_gate(d):
        x = int.from_bytes(d[3:5], "little")
        return ((37 * x + 11) % 65521).to_bytes(2, "little")

    svc, cands = _search((mod_gate, (12345).to_bytes(2, "little")), Families.MODULAR)

    assert svc.stats["solved"] >= 1
    assert any(mod_gate(c) == (12345).to_bytes(2, "little") for c in cands)


# ── Fuzzer / CLI ─────────────────────────────────────────────────────


def _fuzzer(**kw):
    with (
        tempfile.TemporaryDirectory(prefix="derived_cmp_") as tmp,
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(Path(tmp) / "corpus"),
            crashes_dir=str(Path(tmp) / "crashes"),
            max_len=256,
            timeout=1,
            **kw,
        )


def test_fuzzer_builds_derived_service():
    f = _fuzzer(derived_cmp=True)
    assert isinstance(f._affine_mod, AffineMod)
    assert f._affine_mod.families is Families.DERIVED


def test_affine_mod_alone_stays_modular():
    f = _fuzzer(affine_mod=True)
    assert f._affine_mod.families is Families.MODULAR


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.derived_cmp = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["derived_cmp"] is flag


def test_hail_mary_arms_the_flag():
    from fuzzer_tool.cli.commands import _HAIL_MARY_FLAGS

    assert "derived_cmp" in _HAIL_MARY_FLAGS


def test_regression_satisfied_string_compare_logs_no_record():
    """strncmp that holds writes no cmplog record: the last guess round read
    as blind and the byte map dropped a solved operand (compcov_gates G5)."""

    def probe(data: bytes):
        v = _rot(data)
        return NOISE if v == HELLO else NOISE + [(v, HELLO)]

    svc = AffineMod(probe, 400, Families.DERIVED)
    svc.search(SEED)
    cands = []
    while (cand := svc.pop()) is not None:
        cands.append(cand)

    assert any(_rot(c) == HELLO for c in cands)
    assert svc.stats["vanished"] >= 1


# ── review follow-ups (PR #144) ──────────────────────────────────────


def _collect(svc):
    out = []
    while (cand := svc.pop()) is not None:
        out.append(cand)
    return out


def test_regression_weight_of_a_saturated_byte_in_a_narrow_sum():
    """A byte at 0xFF wraps on the +1 screen; its 8-bit weight was rejected as
    not divisible by 255 (signed -1), so a sum of two 0xFF bytes was unsolvable."""
    seed = SEED[:3] + b"\xff\xff" + SEED[5:]

    def probe(data):
        return NOISE + [(bytes([(data[3] + data[4]) & 0xFF]), b"\x10")]

    svc = AffineMod(probe, 400, Families.DERIVED)
    svc.search(seed)

    assert any((c[3] + c[4]) & 0xFF == 0x10 for c in _collect(svc))


def test_regression_failed_additive_guess_falls_through_to_bytemap():
    """A per-byte map that is x + 0x60 below 0x80 looks additive on the screen;
    the additive guess (0x85) lands in the x ^ 0x55 half and fails verification,
    and that used to skip the byte map (which finds 0xB0)."""

    def bytewise(d):
        return bytes(c + 0x60 if c < 0x80 else c ^ 0x55 for c in d[3:7])

    want = bytes([0xE5]) + bytewise(SEED)[1:]
    svc = AffineMod(lambda d: NOISE + [(bytewise(d), want)], 600, Families.DERIVED)
    svc.search(SEED)

    assert any(bytewise(c) == want for c in _collect(svc))


def test_regression_string_search_does_not_starve_additive_fits():
    """An early byte-map search spent the budget before a later 1-exec sum fit."""
    svc = AffineMod(_probe(ROT_GATE, SUM_GATE), 140, Families.DERIVED)
    svc.search(SEED)

    assert any(_sum16(c) == SUM_TARGET for c in _collect(svc))


def test_regression_short_drain_is_not_read_as_satisfied():
    """A FIFO drain can come back without a record; one missing read marked
    every open byte-map position solved and queued a wrong candidate."""
    calls = [0]
    first_round = 1 + DERIVED_SCREEN_BYTES + 1

    def probe(data):
        calls[0] += 1
        v = _rot(data)
        if calls[0] == first_round or v == HELLO:
            return NOISE
        return NOISE + [(v, HELLO)]

    svc = AffineMod(probe, 400, Families.DERIVED)
    svc.search(SEED)

    assert any(_rot(c) == HELLO for c in _collect(svc))
