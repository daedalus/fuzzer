"""Power Doppler seed energy (core/power_doppler.py).

Slow time = successive mutants of one seed; pixel = edge; signal = log hit
count.  SVD drops spatially coherent components (clutter: the shared path,
or an early-reject "flash" moving the whole path at once); CFAR keeps edges
whose residual power beats the noise floor (flow: input-sensitive edges).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core.power_doppler import (
    STALE_FRAMES,
    PowerDoppler,
    _components,
    doppler_power,
)
from fuzzer_tool.core.schedules import SeedScorer

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

N = 16  # ensemble length used by the unit tests
PATH = 20  # edges on the seed's fixed path


def _static(n=N, m=PATH, level=3.0):
    """Ensemble where no mutant moves anything: pure clutter."""
    return np.full((n, m), level)


def _with_flow(x, pattern):
    """Append one edge whose value follows *pattern* (local flow)."""
    return np.column_stack([x, np.asarray(pattern, dtype=float)])


def _toggle(n=N, period=3):
    """Edge hit in 1 of every *period* mutants (slow flow)."""
    return [1.0 if i % period == 0 else 0.0 for i in range(n)]


def _feed(pd, key, rows, ids):
    for row in rows:
        pd.observe(key, {e: int(c) for e, c in zip(ids, row, strict=True)})


class TestDopplerPower:
    def test_falsify_static_ensemble_has_no_flow(self):
        power, flow, rank = doppler_power(_static())

        assert power == 0.0
        assert not flow.any()
        assert rank == 0

    def test_single_toggling_edge_is_the_only_flow(self):
        x = _with_flow(_static(), _toggle())

        power, flow, _ = doppler_power(x)

        assert power > 0.0
        assert flow.tolist() == [False] * PATH + [True]

    def test_adversarial_flash_is_clutter_not_flow(self):
        # Early reject: every path edge drops to 0 together in half the mutants.
        x = _static()
        x[::2] = 0.0

        power, flow, rank = doppler_power(x)

        assert rank >= 1
        assert not flow.any()
        assert power == 0.0

    def test_adversarial_flow_survives_flash(self):
        # Flash and local flow are not orthogonal: leakage must not hide flow.
        x = _static()
        x[::2] = 0.0
        x = _with_flow(x, _toggle(period=4))

        _, flow, rank = doppler_power(x)

        assert rank >= 1
        assert flow[PATH]
        assert not flow[:PATH].any()

    def test_gram_components_match_lapack_svd(self):
        # Reference: LAPACK SVD energies s^2 and per-edge (s v)^2 per component.
        y = np.random.default_rng(7).normal(size=(N, 50))
        y -= y.mean(axis=0)
        _, s, vt = np.linalg.svd(y, full_matrices=False)

        p, energy = _components(y)

        np.testing.assert_allclose(energy, s**2, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(p * p, (s[:, None] * vt) ** 2, rtol=1e-6, atol=1e-9)

    def test_degenerate_shapes(self):
        assert doppler_power(np.zeros((1, 5)))[0] == 0.0
        assert doppler_power(np.zeros((N, 0)))[0] == 0.0
        assert doppler_power(np.zeros((N, 3)))[0] == 0.0


class TestPowerDoppler:
    def test_energy_zero_until_ensemble_closes(self):
        pd = PowerDoppler(ensemble=N)
        rows = _with_flow(_static(), _toggle())
        ids = list(range(PATH + 1))

        _feed(pd, "s", rows[:-1], ids)
        assert pd.energy("s") == 0.0

        _feed(pd, "s", rows[-1:], ids)
        assert pd.energy("s") == 1.0
        assert pd.flow_edges("s") == frozenset({PATH})

    def test_falsify_static_seed_gets_no_energy(self):
        pd = PowerDoppler(ensemble=N)
        ids = list(range(PATH + 1))
        _feed(pd, "flow", _with_flow(_static(), _toggle()), ids)
        _feed(pd, "static", _with_flow(_static(), [0.0] * N), ids)

        assert pd.energy("static") == 0.0
        assert pd.energy("flow") == 1.0

    def test_adversarial_flash_seed_ranks_below_flow_seed(self):
        pd = PowerDoppler(ensemble=N)
        ids = list(range(PATH))
        flash = _static(level=200.0)
        flash[::2] = 0.0
        _feed(pd, "flash", flash, ids)
        _feed(pd, "flow", _with_flow(_static(), _toggle()), ids + [PATH])

        assert pd.energy("flash") < pd.energy("flow")

    def test_edge_first_seen_late_counts_as_flow(self):
        # Absent edges read as 0: an edge only the last mutant reaches is flow.
        pd = PowerDoppler(ensemble=N)
        big = 10**9
        ids = list(range(PATH))
        _feed(pd, "s", _static()[:-1], ids)
        _feed(pd, "s", [[3] * PATH + [1]], ids + [big])

        assert big in pd.flow_edges("s")

    def test_unknown_seed_is_neutral(self):
        pd = PowerDoppler(ensemble=N)
        assert pd.energy("nope") == 0.0
        assert pd.flow_edges("nope") == frozenset()

    def test_saturated_counts_do_not_overflow(self):
        pd = PowerDoppler(ensemble=N)
        rows = [[0xFFFFFF if i % 2 else 0, 1] for i in range(N)]
        _feed(pd, "s", rows, [1, 2])

        assert np.isfinite(pd.energy("s"))

    def test_memory_bounded_by_seed_cap(self):
        cap = 4
        pd = PowerDoppler(ensemble=N, max_seeds=cap)
        for k in range(cap * 3):
            _feed(pd, f"s{k}", _static(n=2), list(range(PATH)))

        assert pd.stats()["open"] <= cap

    def test_memory_bounded_by_edge_cap(self):
        cap = 8
        pd = PowerDoppler(ensemble=N, max_edges=cap)
        _feed(pd, "s", _static(n=1, m=PATH), list(range(PATH)))

        assert pd.stats()["dropped_edges"] == PATH - cap

    def test_regression_cyclic_corpus_beyond_seed_cap_closes_frames(self):
        # Falsification: 3x more seeds than open slots, one mutant per pick.
        # LRU eviction of partial frames left every seed unscored forever.
        cap, n_seeds = 4, 12
        pd = PowerDoppler(ensemble=N, max_seeds=cap)
        ids = list(range(PATH))
        for _ in range(N):
            for k in range(n_seeds):
                _feed(pd, f"s{k}", _static(n=1), ids)

        assert pd.stats()["ensembles"] >= cap
        assert pd.stats()["open"] <= cap

    def test_regression_abandoned_frame_yields_its_slot(self):
        # Adversarial: a seed never picked again must not pin its slot; its
        # partial frame (>= 3 samples) is scored on the way out.
        pd = PowerDoppler(ensemble=N, max_seeds=1)
        ids = list(range(PATH + 1))
        flow = _with_flow(_static(), _toggle())
        _feed(pd, "gone", flow[:3], ids)
        stale = N * STALE_FRAMES

        _feed(pd, "next", _static(n=stale + N), ids[:PATH])

        assert pd.flow_edges("gone") == frozenset({PATH})
        assert pd.stats()["ensembles"] == 2

    def test_adversarial_tiny_partial_frame_not_scored(self):
        # Fewer samples than a valid ensemble: dropped, never scored.
        pd = PowerDoppler(ensemble=N, max_seeds=1)
        ids = list(range(PATH))
        _feed(pd, "gone", _static(n=2), ids)

        _feed(pd, "next", _static(n=N * STALE_FRAMES + N), ids)

        assert pd.stats()["ensembles"] == 1
        assert pd.energy("gone") == 0.0

    def test_regression_flow_ids_bounded_in_total(self):
        # Falsification: retained flow-edge ids stay under the global budget,
        # however many seeds are scored.
        budget = 3
        pd = PowerDoppler(ensemble=N, max_flow_ids=budget)
        rows = np.column_stack([_static(m=PATH)] + [_toggle()] * 2)
        ids = list(range(PATH + 2))
        for k in range(5):
            _feed(pd, f"s{k}", rows, ids)

        assert pd.stats()["flow_ids"] <= budget
        assert pd.flow_edges("s4") == frozenset({PATH, PATH + 1})

    def test_adversarial_rescore_does_not_leak_flow_budget(self):
        # Re-closing the same seed replaces, not adds to, its retained ids.
        pd = PowerDoppler(ensemble=N)
        rows = _with_flow(_static(), _toggle())
        ids = list(range(PATH + 1))
        for _ in range(4):
            _feed(pd, "s", rows, ids)

        assert pd.stats()["flow_ids"] == 1

    def test_bad_parameters_rejected(self):
        with pytest.raises(ValueError):
            PowerDoppler(ensemble=2)
        with pytest.raises(ValueError):
            PowerDoppler(max_seeds=0)
        with pytest.raises(ValueError):
            PowerDoppler(max_edges=0)
        with pytest.raises(ValueError):
            PowerDoppler(max_flow_ids=-1)


class TestDopplerSchedule:
    def _score(self, energy):
        return SeedScorer("doppler").score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=10,
            avg_bitmap_size=10,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1,
            doppler_energy=energy,
        )

    def test_falsify_zero_energy_is_neutral(self):
        assert self._score(0.0) == SeedScorer("base").score(100, 100, 10, 10, 0, 0, 0, 0, 1)

    def test_full_energy_hits_max_mult(self):
        sc = SeedScorer("doppler")
        assert self._score(1.0) == self._score(0.0) * sc.max_mult

    def test_adversarial_energy_out_of_range_is_clamped(self):
        assert self._score(-5.0) == self._score(0.0)
        assert self._score(50.0) == self._score(1.0)


_DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) return 0;
    unsigned char buf[64]; size_t n = fread(buf, 1, sizeof buf, f); fclose(f);
    for (size_t i = 0; i < n; i++) {
        uint32_t reps = 1 + (buf[i] & 7);
        for (uint32_t r = 0; r < reps; r++) {
            uint32_t guard = 1 + (buf[i] >> 2);
            __sanitizer_cov_trace_pc_guard(&guard);
        }
    }
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")
def test_fuzz_loop_feeds_doppler(tmp_path):
    """--schedule doppler wiring: executions reach the ensembles."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    src, exe = tmp_path / "drv.c", tmp_path / "drv"
    src.write_text(_DRIVER)
    r = subprocess.run(
        ["clang", "-O1", "-include", str(SHIM), "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    # clang presence is the skipif above; a failed build is a shim regression.
    assert r.returncode == 0, r.stderr[:300]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"hello world")
    f = Fuzzer(
        target=str(exe),
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
        timeout=1,
        mutations_per_input=2,
        quiet_stats=True,
        use_coverage=True,
        file_mode=True,
        schedule="doppler",
    )
    assert f.shm_cov is not None
    # Short frames so ensembles close within the run and energy is read back.
    f._doppler = PowerDoppler(ensemble=3)
    f.run(iterations=100_000, max_execs=600)

    stats = f._doppler.stats()
    # Every execution reaches Doppler: accepted, or waiting for a free slot.
    assert stats["samples"] + stats["refused"] >= 500
    assert stats["ensembles"] > 0


def _fuzzer_no_target(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=64,
        schedule="doppler",
        **kw,
    )


def test_regression_doppler_without_shm_is_disabled(tmp_path, capsys):
    """Falsification: no SHM means no samples; disable and say so."""
    f = _fuzzer_no_target(tmp_path, use_coverage=False)

    assert f._doppler is None
    assert f._seed_scorer.schedule == "base"
    assert "doppler" in capsys.readouterr().out.lower()


def test_doppler_with_shm_stays_enabled(tmp_path, monkeypatch):
    """Adversarial: the gate must not disable Doppler when SHM is live."""
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)
    f = _fuzzer_no_target(tmp_path, use_coverage=True)

    assert f.shm_cov is not None
    assert f._doppler is not None
    assert f._seed_scorer.schedule == "doppler"


def _key_of(target, multi):
    from types import SimpleNamespace

    from fuzzer_tool.services.fuzzer import Fuzzer

    owner = SimpleNamespace(target=target, multi_targets=multi)
    return Fuzzer._doppler_key(owner, "seed")


def test_regression_doppler_key_namespaced_per_target():
    """Falsification: SHM edge ids are per target; one seed, two targets, two frames."""
    multi = ["a.bin", "b.bin"]

    assert _key_of("a.bin", multi) != _key_of("b.bin", multi)
    assert _key_of("a.bin", multi) == _key_of("a.bin", multi)


def test_doppler_key_single_target_is_seed_key():
    """Adversarial: single-target runs keep the plain seed key."""
    assert _key_of("a.bin", None) == "seed"
