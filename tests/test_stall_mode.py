"""--stall-mode random|exploit: what stall recovery does to op choice.

AFL++ `-P` switches to its exploitation op set when finds dry up and back on
new coverage. `exploit` here: on stall, draw only from the precise bands
(adaptive: redqueen/I2S/colorization..., dict); `random` keeps uniform.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import EXPLOIT_BANDS, OperatorEngine, StallMode
from tests.support.scripted_rng import ScriptedRng

EXPLOIT = sorted(n for b in EXPLOIT_BANDS for n in REGISTRY.categories()[b])
OTHER = sorted(REGISTRY.categories()["bit"])


class _Engine(OperatorEngine):
    ctx = None  # plain attribute: the real property reads a full Fuzzer


def _engine(mode, choices):
    f = SimpleNamespace(_stall_recovery_active=True, _meta_strategy=None, _op_selector=None)
    eng = _Engine.__new__(_Engine)
    eng.f = f
    eng.ctx = SimpleNamespace(_rng=ScriptedRng(choice_idxs=choices))
    eng.bind_stall_mode(mode)
    return eng, f


def test_exploit_draws_only_precise_ops():
    ops = OTHER[:3] + EXPLOIT[:2]
    eng, f = _engine(StallMode.EXPLOIT, choices=[1])
    pick = eng.select_op(ops)
    assert pick == sorted(set(ops) & set(EXPLOIT))[1]
    assert f._meta_strategy == "exploit_stall"


def test_random_draws_from_all_ops():
    """Control: random mode sees the full list (index 0 is a bit op)."""
    ops = OTHER[:3] + EXPLOIT[:2]
    eng, f = _engine(StallMode.RANDOM, choices=[0])
    assert eng.select_op(ops) == ops[0]
    assert f._meta_strategy == "random_stall"


def test_exploit_falls_back_without_precise_ops():
    """Adversarial: no adaptive/dict op available -> uniform over what there is."""
    ops = OTHER[:4]
    eng, _ = _engine(StallMode.EXPLOIT, choices=[2])
    assert eng.select_op(ops) == ops[2]


def test_bands_exist():
    assert all(b in REGISTRY.categories() for b in EXPLOIT_BANDS)


def test_cli(monkeypatch):
    from tests.test_lightweight_mode import _cli

    args, _ = _cli(monkeypatch, "--stall-mode", "exploit")
    assert args.stall_mode == "exploit"
    args, _ = _cli(monkeypatch)
    assert args.stall_mode == "random"
    with pytest.raises(SystemExit):
        _cli(monkeypatch, "--stall-mode", "maybe")


def test_exploit_marker_not_dressed_as_scheduler():
    from fuzzer_tool.core.analyzers.analyzer_elo import strategy_display_name

    assert strategy_display_name("exploit_stall") == "exploit_stall"
