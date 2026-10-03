"""Regression: startup reads branch density from the cached profile.

Fuzzer.run decoded the whole .text in pure Python on every start (66 s on
ffmpeg) for one printed line, although TargetProfile is cached per binary.
"""

from unittest.mock import patch

from fuzzer_tool.core.target_profiler import TargetProfile
from fuzzer_tool.services.fuzzer import Fuzzer


def _stub(density: float | None) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f.target = "/nonexistent/target"
    f.multi_targets = None
    f._profile = TargetProfile(text_branch_density=density)
    return f


def test_regression_single_target_uses_cached_profile(capsys):
    with patch("fuzzer_tool.core.elf.branch_density") as decode:
        _stub(22.9)._print_branch_density()
    decode.assert_not_called()
    assert "Branch density: 22.9 cond branches/KB" in capsys.readouterr().out


def test_undecodable_text_prints_nothing(capsys):
    """Adversarial: a None density stays silent and does not re-decode."""
    with patch("fuzzer_tool.core.elf.branch_density") as decode:
        _stub(None)._print_branch_density()
    decode.assert_not_called()
    assert "Branch density" not in capsys.readouterr().out
