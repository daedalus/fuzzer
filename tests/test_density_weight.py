"""``--density-weight``: scale seed weight by the rank of its edges per bit."""

from __future__ import annotations

import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fuzzer_tool.core.edge_density import (
    BITS_PER_BYTE,
    DENSITY_LOG2_SPAN,
    density_factors,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker


def _reference(edges: list[int], nbytes: list[int]) -> list[float]:
    """Scalar oracle: mid-rank of edges/bit over ranked seeds, others 1.0."""
    ranked = [i for i in range(len(edges)) if edges[i] > 0 and nbytes[i] > 0]
    out = [1.0] * len(edges)
    if len(ranked) < 2:
        return out

    dens = {i: edges[i] / (nbytes[i] * BITS_PER_BYTE) for i in ranked}
    for i in ranked:
        below = sum(1 for j in ranked if dens[j] < dens[i])
        same = sum(1 for j in ranked if dens[j] == dens[i])
        t = (below + (same - 1) / 2) / (len(ranked) - 1)
        out[i] = 2.0 ** (DENSITY_LOG2_SPAN * (2 * t - 1))
    return out


def _factors(edges: list[int], nbytes: list[int]) -> list[float]:
    return density_factors(np.array(edges), np.array(nbytes)).tolist()


def test_factors_span_and_median():
    """Example from the docstring: densities 0.1/0.2/0.4 -> 0.5x/1x/2x."""
    got = _factors([8, 16, 32], [10, 10, 10])
    assert got == pytest.approx([2.0**-DENSITY_LOG2_SPAN, 1.0, 2.0**DENSITY_LOG2_SPAN])


def test_tiny_seed_bounded():
    """Falsification of ratio weighting: a 1-byte outlier gets the top factor, no more."""
    got = _factors([3, 300, 200], [1, 4096, 4096])
    assert max(got) == pytest.approx(2.0**DENSITY_LOG2_SPAN)


@pytest.mark.parametrize(
    ("edges", "nbytes"),
    [([], []), ([5], [1]), ([0, 0], [1, 1]), ([5, 5], [0, 0]), ([-1, 4], [2, -2])],
)
def test_degenerate_inputs_neutral(edges, nbytes):
    """Adversarial: empty, single, edgeless, sizeless, negative -> all 1.0."""
    assert _factors(edges, nbytes) == [1.0] * len(edges)


def test_ties_share_mid_rank():
    """Adversarial: equal densities get equal factors (no order dependence)."""
    got = _factors([8, 8, 8, 1], [1, 1, 1, 1])
    assert got[0] == got[1] == got[2]
    assert got == pytest.approx(_reference([8, 8, 8, 1], [1, 1, 1, 1]))


def _random_case(seed: int) -> tuple[list[int], list[int]]:
    rng = RandPool(seed=seed)
    n = 40
    return rng.randint_list(0, 12, n), rng.randint_list(0, 6, n)


def test_reference_control():
    """Control (Hard Rule 46): the oracle agrees with a second run of itself."""
    edges, nbytes = _random_case(3)
    assert _reference(edges, nbytes) == _reference(list(edges), list(nbytes))


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference(seed):
    edges, nbytes = _random_case(seed)
    assert _factors(edges, nbytes) == pytest.approx(_reference(edges, nbytes))


def _picker(corpus: list[bytes], hits: list[set[int]], on: Switch) -> SeedPicker:
    f = SimpleNamespace(
        _density_weight=on,
        _edge_tracker=SimpleNamespace(
            seed_edges={s.hex(): h for s, h in zip(corpus, hits, strict=True)}
        ),
        _seed_key=bytes.hex,
    )
    p = SeedPicker.__new__(SeedPicker)
    p.f = f
    return p


def test_picker_table_keys_by_seed():
    corpus = [b"a" * 10, b"b" * 10, b"c" * 10]
    hits = [set(range(8)), set(range(16)), set(range(32))]
    table = _picker(corpus, hits, Switch.ON)._density_table(corpus, [None] * 3)

    want = _factors([len(h) for h in hits], [len(s) for s in corpus])
    assert [table[s.hex()] for s in corpus] == pytest.approx(want)


def test_picker_table_off():
    corpus = [b"a", b"bb"]
    p = _picker(corpus, [{1}, {1, 2}], Switch.OFF)
    assert p._density_table(corpus, [None, None]) == {}


@pytest.fixture
def fuzzer():
    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            density_weight=Switch.ON,
        )
    yield f
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_fuzzer_wires_flag(fuzzer):
    assert fuzzer._density_weight is Switch.ON


def test_compute_weights_applies_factor(fuzzer):
    """Wiring: on/off weight ratio per seed equals its density factor."""
    corpus = [b"a" * 4, b"b" * 40, b"c" * 400]
    fuzzer.corpus = list(corpus)
    for s in corpus:
        fuzzer.seed_meta[s] = {
            "fuzz_count": 1,
            "coverage_edges": 1,
            "added_at": 0.0,
            "lineage_depth": 0,
        }
        fuzzer._edge_tracker.record_edges(fuzzer._seed_key(s), set(range(1, 9)))

    picker = fuzzer._seed_picker
    on = picker._compute_weights(0.0)
    fuzzer._density_weight = Switch.OFF
    off = picker._compute_weights(0.0)

    want = _factors([8, 8, 8], [len(s) for s in corpus])
    assert [a / b for a, b in zip(on, off, strict=True)] == pytest.approx(want)
