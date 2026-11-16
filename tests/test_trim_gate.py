"""New-coverage trim is off unless --trim-new-coverage asks for it.

The AFL chunk walk costs up to 512 execs per admission; on FFmpeg it took
~98% of campaign time. It now runs only behind the flag, and
--no-auto-minimize still turns it off.
"""

from types import SimpleNamespace

from fuzzer_tool.cli import commands
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.services.fuzz_round import FuzzRound
from tests import test_commands_extended

LONG = b"x" * 64


def _round(trim, auto=Switch.ON, new_cov=True, data=LONG):
    f = SimpleNamespace(_trim_new_coverage_on=trim, _auto_minimize=auto)
    r = FuzzRound(f, b"parent")
    r._has_new_coverage = new_cov
    r._mutated = data
    return r


def test_off_by_default_never_trims():
    """Falsification: the default switch must not trim a new-coverage seed."""
    assert not _round(Switch.OFF)._trims()


def test_flag_trims_new_coverage():
    assert _round(Switch.ON)._trims()


def test_no_auto_minimize_wins():
    """Adversarial: both flags set, --no-auto-minimize keeps trim off."""
    assert not _round(Switch.ON, auto=Switch.OFF)._trims()


def test_only_new_coverage_and_long_inputs():
    assert not _round(Switch.ON, new_cov=False)._trims()
    assert not _round(Switch.ON, data=b"x" * 10)._trims()


def test_fuzzer_default_is_off(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"seed")
    f = Fuzzer(target="/bin/true", corpus_dir=str(corpus), crashes_dir=str(tmp_path / "k"))
    assert f._trim_new_coverage_on is Switch.OFF


def _cli_kwargs(monkeypatch, tmp_path, **flags):
    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    for name, value in flags.items():
        setattr(args, name, value)
    captured = {}

    def fake_fuzzer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda max_execs=0: 0, aborted=False)

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake_fuzzer)
    commands.cmd_fuzz(args)
    return captured


def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path):
    on = _cli_kwargs(monkeypatch, tmp_path, trim_new_coverage=True)
    assert on["trim_new_coverage"] is Switch.ON

    (tmp_path / "off").mkdir()
    off = _cli_kwargs(monkeypatch, tmp_path / "off")
    assert off["trim_new_coverage"] is Switch.OFF
