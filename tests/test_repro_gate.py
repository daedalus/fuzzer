"""--repro-gate N: novel crashes must reproduce before they are saved (OneFuzz port P1).

A novel crash is replayed up to N times; one replay of the same bug class
passes it (OneFuzz ``check_retry_count``). None → ``crashes/no_repro/``,
no alert, own signature map.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from fuzzer_tool.adapters.filesystem import NO_REPRO_DIR
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer

_UAF = (
    "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "    #0 0x401111 in json_parse_ex\n    #1 0x402222 in json_parse\n"
)
_HBO = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000010\n"
    "    #0 0x401111 in json_parse_ex\n    #1 0x402222 in json_parse\n"
)
_CLEAN = (0, "")
_CRASH = (1, _UAF)


class _Script:
    """Runner returning scripted (rc, stderr) results in order; counts calls."""

    def __init__(self, results: list[tuple[int, str]]):
        self._results = list(results)
        self.calls = 0

    def run_target(self, data: bytes) -> tuple[int, str]:
        self.calls += 1
        return self._results.pop(0)


def _setup(tmp_path, monkeypatch, n: int, results: list[tuple[int, str]]):
    f = _Fuzzer(tmp_path)
    f.repro_gate = n
    f.no_repro_hashes = set()
    f.no_repro_sigs = {}
    f._runner = _Script(results)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    alerts: list[object] = []
    f._push = object()
    monkeypatch.setattr(cm, "_push_crash", lambda *a, **k: alerts.append(a))
    return f, cm, alerts


def _saved(f) -> int:
    return len(list(f.crashes_dir.glob("crash_*.json")))


def _no_repro(f) -> int:
    return len(list((f.crashes_dir / NO_REPRO_DIR).glob("crash_*.json")))


def test_clean_replays_go_to_no_repro(tmp_path, monkeypatch) -> None:
    # Falsification: a crash that never reproduces is not a saved crash.
    f, cm, alerts = _setup(tmp_path, monkeypatch, 2, [_CLEAN, _CLEAN])

    assert not cm.save_crash(b"in", 1, _UAF)
    assert (_saved(f), _no_repro(f), alerts) == (0, 1, [])
    assert f._last_crash_signature is None


def test_other_bug_class_is_not_a_repro(tmp_path, monkeypatch) -> None:
    # Adversarial: a replay that crashes as another bug must not pass.
    f, cm, _ = _setup(tmp_path, monkeypatch, 1, [(1, _HBO)])

    assert not cm.save_crash(b"in", 1, _UAF)
    assert (_saved(f), _no_repro(f)) == (0, 1)


def test_first_match_passes(tmp_path, monkeypatch) -> None:
    f, cm, alerts = _setup(tmp_path, monkeypatch, 3, [_CLEAN, _CLEAN, _CRASH])

    assert cm.save_crash(b"in", 1, _UAF)
    assert (_saved(f), _no_repro(f), len(alerts)) == (1, 0, 1)
    assert f._runner.calls == 3


def test_stops_at_first_match(tmp_path, monkeypatch) -> None:
    f, cm, _ = _setup(tmp_path, monkeypatch, 3, [_CRASH, _CLEAN, _CLEAN])

    assert cm.save_crash(b"in", 1, _UAF)
    assert f._runner.calls == 1


def test_budget_exhausted_fails(tmp_path, monkeypatch) -> None:
    # Edge: the match would come on replay 3, N=2 never reaches it.
    f, cm, _ = _setup(tmp_path, monkeypatch, 2, [_CLEAN, _CLEAN, _CRASH])

    assert not cm.save_crash(b"in", 1, _UAF)
    assert f._runner.calls == 2


def test_off_never_replays(tmp_path, monkeypatch) -> None:
    f, cm, _ = _setup(tmp_path, monkeypatch, 0, [])

    assert cm.save_crash(b"in", 1, _UAF)
    assert f._runner.calls == 0


def test_repeat_signature_not_replayed(tmp_path, monkeypatch) -> None:
    # Speed: only novel crashes pay the gate.
    f, cm, _ = _setup(tmp_path, monkeypatch, 1, [_CRASH])
    assert cm.save_crash(b"a", 1, _UAF)

    assert not cm.save_crash(b"b", 1, _UAF)
    assert f._runner.calls == 1


def test_no_repro_sig_later_reproducing_is_saved(tmp_path, monkeypatch) -> None:
    # Edge: a no_repro signature does not shadow a reproducing instance.
    f, cm, _ = _setup(tmp_path, monkeypatch, 1, [_CLEAN, _CRASH])
    assert not cm.save_crash(b"a", 1, _UAF)

    assert cm.save_crash(b"b", 1, _UAF)
    assert (_saved(f), _no_repro(f)) == (1, 1)


def test_no_repro_dedups_itself(tmp_path, monkeypatch) -> None:
    f, cm, _ = _setup(tmp_path, monkeypatch, 1, [_CLEAN, _CLEAN])
    cm.save_crash(b"a", 1, _UAF)
    cm.save_crash(b"b", 1, _UAF)

    assert _no_repro(f) == 1


def test_replay_state_restored(tmp_path, monkeypatch) -> None:
    # Replays reset per-run crash state; the novel crash's must survive, and
    # each replay's shim record must be drained so it cannot leak forward.
    f, cm, _ = _setup(tmp_path, monkeypatch, 2, [])
    f._last_regs = {"rip": 0x1234}
    f._last_fault_addr = 0xDEAD
    drains: list[int] = []
    f._crash_sym_sink = types.SimpleNamespace(drain=lambda: drains.append(1))

    def _run(data: bytes) -> tuple[int, str]:
        f._last_regs, f._last_fault_addr = {}, None
        return _CLEAN

    f._runner = types.SimpleNamespace(run_target=_run)
    seen: dict = {}
    monkeypatch.setattr(
        cm,
        "_save_no_repro",
        lambda *a, **k: seen.update(regs=f._last_regs, fault=f._last_fault_addr),
    )
    cm.save_crash(b"in", 1, _UAF)

    assert seen == {"regs": {"rip": 0x1234}, "fault": 0xDEAD}
    assert len(drains) == 1 + 2  # the crash's own record, then one per replay


def test_resync_no_repro(tmp_path, monkeypatch) -> None:
    # --resume: no_repro signatures come back from their sidecars.
    f, cm, _ = _setup(tmp_path, monkeypatch, 1, [_CLEAN])
    cm.save_crash(b"a", 1, _UAF)
    sigs = dict(f.no_repro_sigs)
    f.no_repro_sigs = {}

    cm.resync_no_repro()

    assert sigs and set(f.no_repro_sigs) == set(sigs)


def test_cli_parses_repro_gate(monkeypatch) -> None:
    from fuzzer_tool.cli import commands
    from fuzzer_tool.services.fuzzer import Fuzzer

    captured = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.update(args=a) or 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--repro-gate", "3"])

    assert commands.main() == 0
    assert captured["args"].repro_gate == 3
    assert "repro_gate" in inspect.signature(Fuzzer.__init__).parameters


def test_cli_rejects_negative(monkeypatch) -> None:
    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--repro-gate", "-1"])

    with pytest.raises(SystemExit):
        commands.main()


def test_no_repro_maps_bounded(monkeypatch) -> None:
    # Hard Rule 54: flaky targets cannot grow the no_repro maps forever.
    from fuzzer_tool.services import fuzzer as fuzzer_mod

    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_HASHES", 2)
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_SIGS", 2)
    f = types.SimpleNamespace(
        crash_hashes=set(),
        crash_sigs={},
        no_repro_hashes={"a", "b", "c"},
        no_repro_sigs={"s1": 1, "s2": 1, "s3": 1},
    )

    fuzzer_mod.Fuzzer._prune_crash_data(f)

    assert len(f.no_repro_hashes) <= 2 and len(f.no_repro_sigs) <= 2
