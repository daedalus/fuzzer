"""--te-correction: TransferEntropy binds its bias correction once in __init__.

SHUFFLE (default) keeps the surrogate-shuffle estimator bit-identical;
ANALYTIC routes transfer_entropy() to pt_corrected_te (Panzeri-Treves).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.analyzers.analyzer_transfer_entropy import TECorrection, TransferEntropy
from fuzzer_tool.core.rand_pool import RandPool

# (seed, n, alphabet max) -> float.hex of the master shuffle TE: pins bit-identity.
_GOLDEN = [
    (5, 60, 3, "0x1.9114b5fd30b13p-1"),
    (11, 200, 7, "0x1.89cf808e99c8ap-1"),
    (23, 500, 3, "0x1.b4c325c81556ep-1"),
]


def _coupled(seed: int, n: int, hi: int) -> tuple[list[int], list[int]]:
    """Target copies source lagged by one on 2 of 3 steps, else noise."""
    rng = RandPool(seed=seed)
    source = rng.randint_list(0, hi, n)
    lag = [0, *source[:-1]]
    noise = rng.randint_list(0, hi, n)
    target = [a if i % 3 else b for i, (a, b) in enumerate(zip(lag, noise, strict=True))]
    return source, target


@pytest.mark.parametrize(("seed", "n", "hi", "golden"), _GOLDEN)
def test_default_shuffle_bit_identical(seed, n, hi, golden):
    source, target = _coupled(seed, n, hi)

    default = TransferEntropy(history_length=1).transfer_entropy(source, target)
    explicit = TransferEntropy(history_length=1, correction=TECorrection.SHUFFLE)

    assert default.hex() == golden
    assert explicit.transfer_entropy(source, target).hex() == golden


def test_control_shuffle_against_itself():
    """Rule 46 control: two fresh instances agree before comparing modes."""
    source, target = _coupled(11, 200, 7)
    a = TransferEntropy(history_length=1).transfer_entropy(source, target)
    b = TransferEntropy(history_length=1).transfer_entropy(source, target)

    assert a == b


@pytest.mark.parametrize(("seed", "n", "hi"), [g[:3] for g in _GOLDEN])
def test_analytic_routes_to_pt(seed, n, hi):
    source, target = _coupled(seed, n, hi)
    te = TransferEntropy(history_length=1, correction=TECorrection.ANALYTIC)

    assert te.transfer_entropy(source, target) == te.pt_corrected_te(source, target)


def test_falsify_modes_differ():
    """Falsification: a binding that ignored the mode would return equal values."""
    source, target = _coupled(11, 200, 7)
    shuffle = TransferEntropy(correction=TECorrection.SHUFFLE).transfer_entropy(source, target)
    analytic = TransferEntropy(correction=TECorrection.ANALYTIC).transfer_entropy(source, target)

    assert shuffle != analytic


def test_analytic_accepts_n_surrogates():
    """Drop-in: shuffle callers may pass n_surrogates; analytic ignores it."""
    source, target = _coupled(5, 60, 3)
    te = TransferEntropy(correction=TECorrection.ANALYTIC)

    assert te.transfer_entropy(source, target, n_surrogates=40) == te.pt_corrected_te(
        source, target
    )


def test_analytic_reaches_internal_callers():
    """edge_to_edge_flow calls self.transfer_entropy: it must see the binding."""
    source, target = _coupled(23, 500, 3)
    te = TransferEntropy(correction=TECorrection.ANALYTIC)

    flow = te.transfer_entropy_matrix({"x": source, "y": target})

    assert flow[("x", "y")] == te.pt_corrected_te(source, target)


def test_adversarial_string_mode_rejected():
    """A raw string is not a mode: fail loudly, never fall back to a default."""
    with pytest.raises(KeyError):
        TransferEntropy(correction="analytic")


def test_adversarial_analytic_short_series():
    te = TransferEntropy(correction=TECorrection.ANALYTIC)

    assert te.transfer_entropy([], []) == 0.0
    assert te.transfer_entropy([1, 2], [1, 2]) == 0.0


# ── wiring ──────────────────────────────────────────────────────────


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


def test_cli_default_shuffle(monkeypatch, tmp_path):
    assert _fuzz_kwargs(monkeypatch, tmp_path, [])["te_correction"] is TECorrection.SHUFFLE


def test_cli_analytic_reaches_fuzzer(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--te-correction", "analytic"])

    assert kw["te_correction"] is TECorrection.ANALYTIC


def test_cli_rejects_unknown_mode(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _fuzz_kwargs(monkeypatch, tmp_path, ["--te-correction", "banana"])


def test_registry_builds_selected_mode():
    from fuzzer_tool.core.analyzer_registry import _activate_transfer_entropy

    f = MagicMock(_use_transfer_entropy=True, _te_correction=TECorrection.ANALYTIC)
    _activate_transfer_entropy(f)

    assert f._te.correction is TECorrection.ANALYTIC


def test_report_shows_mode():
    from fuzzer_tool.services.report import _misc_strategy_lines

    f = MagicMock(_use_transfer_entropy=True, _use_mi=False, _secretary=None, _anneal_budget=0)
    f._te = TransferEntropy(correction=TECorrection.ANALYTIC)
    lines: list[str] = []
    _misc_strategy_lines(f, lines)

    assert "  Transfer entropy: history=1, correction=analytic" in lines
