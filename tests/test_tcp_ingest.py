"""Tests for services/tcp_ingest.py: --tcp-mode direct.

Received seeds skip the fuzz round: run as is, edges recorded, admitted to the
corpus only on new coverage. The coverage seams (_unmutated_coverage,
_record_direct_edges) are patched: no instrumented target is built here.
"""

from __future__ import annotations

import socket
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.adapters.tcp_seed import DEFAULT_HOST
from fuzzer_tool.services.tcp_ingest import TcpMode

WAIT_S = 5.0
NEW_EDGES = [11, 22, 33]


def _send(port: int, payload: bytes) -> None:
    with socket.create_connection((DEFAULT_HOST, port), timeout=WAIT_S) as s:
        s.sendall(payload)


def _wait_received(inbox, n: int) -> None:
    deadline = time.monotonic() + WAIT_S
    while inbox.received + inbox.dropped < n:
        assert time.monotonic() < deadline, "listener never handled the message"
        time.sleep(0.01)


@pytest.fixture
def direct(tmp_path):
    from tests.test_regression_end_of_run_persistence import _make_fuzzer

    f = _make_fuzzer(tmp_path, tcp_listen_port=0, tcp_mode=TcpMode.DIRECT)
    yield f
    f._tcp_inbox.close()


def _feed(f, *payloads: bytes) -> None:
    for p in payloads:
        _send(f._tcp_inbox.port, p)
    _wait_received(f._tcp_inbox, len(payloads))


def _coverage(f, verdicts: dict[bytes, bool]):
    """Patch the coverage seams: new edges per payload as scripted."""
    unmutated = patch.object(
        f,
        "_unmutated_coverage",
        side_effect=lambda d: (verdicts[bytes(d)], NEW_EDGES if verdicts[bytes(d)] else []),
    )
    record = patch.object(f, "_record_direct_edges", return_value=len(NEW_EDGES))
    return unmutated, record


class TestDrain:
    def test_new_coverage_admitted_boring_not(self, direct, capsys):
        _feed(direct, b"NEWCOV", b"BORING")
        unmutated, record = _coverage(direct, {b"NEWCOV": True, b"BORING": False})
        with patch.object(direct, "_run_target", return_value=(0, "")), unmutated, record as rec:
            assert direct._tcp_direct.drain() == 2

        assert b"NEWCOV" in direct.corpus
        assert b"BORING" not in direct.corpus
        rec.assert_called_once_with(b"NEWCOV", NEW_EDGES)
        out = capsys.readouterr().out
        assert f"Ran seed id_{hash_data(b'NEWCOV')} (direct): rc=0, +3 edges, admitted" in out
        assert f"Ran seed id_{hash_data(b'BORING')} (direct): rc=0, +0 edges, boring" in out

    def test_runs_the_exact_bytes(self, direct):
        _feed(direct, b"\x00EXACT\xff")
        ran = []
        unmutated, record = _coverage(direct, {b"\x00EXACT\xff": False})
        with (
            patch.object(
                direct, "_run_target", side_effect=lambda d: ran.append(bytes(d)) or (0, "")
            ),
            unmutated,
            record,
        ):
            direct._tcp_direct.drain()
        assert ran == [b"\x00EXACT\xff"]

    def test_drains_everything_queued(self, direct):
        payloads = [b"S%02d" % i for i in range(12)]
        _feed(direct, *payloads)
        unmutated, record = _coverage(direct, dict.fromkeys(payloads, False))
        with patch.object(direct, "_run_target", return_value=(0, "")), unmutated, record:
            assert direct._tcp_direct.drain() == 12
        assert direct._tcp_inbox.pending == 0
        assert direct._tcp_ran == 12

    def test_exec_count_advances(self, direct):
        _feed(direct, b"A1", b"A2")
        before = direct.exec_count
        unmutated, record = _coverage(direct, {b"A1": False, b"A2": False})
        with patch.object(direct, "_run_target", return_value=(0, "")), unmutated, record:
            direct._tcp_direct.drain()
        assert direct.exec_count == before + 2

    def test_empty_queue_runs_nothing(self, direct):
        # Falsification: no seed, no execution.
        with patch.object(direct, "_run_target") as run:
            assert direct._tcp_direct.drain() == 0
        run.assert_not_called()


class TestAdversarial:
    def test_crash_saved_not_admitted(self, direct, capsys):
        _feed(direct, b"CRASHME")
        with (
            patch.object(direct, "_run_target", return_value=(-11, "")),
            patch.object(direct, "_is_crash", return_value=True),
            patch.object(direct, "save_crash") as save_crash,
            patch.object(direct, "_unmutated_coverage") as cov,
        ):
            direct._tcp_direct.drain()
        save_crash.assert_called_once_with(b"CRASHME", -11, "")
        cov.assert_not_called()
        assert b"CRASHME" not in direct.corpus
        assert "crash" in capsys.readouterr().out

    def test_timeout_saved_as_timeout(self, direct, capsys):
        _feed(direct, b"SLOW")
        with (
            patch.object(direct, "_run_target", return_value=(-1, "")),
            patch.object(direct, "_is_crash", return_value=False),
            patch.object(direct._corpus_manager, "save_timeout") as save_timeout,
        ):
            direct._tcp_direct.drain()
        save_timeout.assert_called_once_with(b"SLOW")
        assert b"SLOW" not in direct.corpus
        assert "timeout" in capsys.readouterr().out

    def test_no_coverage_target_never_admits(self, direct):
        # /bin/true has no SHM: nothing can show new coverage, nothing is admitted.
        assert direct._unmutated_coverage(b"anything") == (False, [])

    def test_mutate_never_consumes_direct_seeds(self, direct):
        # The round path must not race the direct path for the same queue.
        _feed(direct, b"MINE")
        out = bytes(direct.mutate(b"parent"))
        assert out != b"MINE"
        assert direct._tcp_inbox.pending == 1


class TestWiring:
    def test_default_mode_is_round(self, tmp_path):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        try:
            assert f._tcp_direct is None
        finally:
            f._tcp_inbox.close()

    def test_run_drains_before_fuzzing(self, direct, capsys):
        _feed(direct, b"R1", b"R2")
        unmutated, record = _coverage(direct, {b"R1": True, b"R2": False})
        with patch.object(direct, "_run_target", return_value=(0, "")), unmutated, record:
            direct.run(max_execs=5)
        out = capsys.readouterr().out
        assert out.count("(direct)") == 2
        assert "TCP seeds: received 2, ran 2, dropped 0, unrun 0" in out
        assert b"R1" in direct.corpus

    def test_cli_flag_parses(self, monkeypatch):
        from fuzzer_tool.cli import commands

        captured = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
        argv = [
            "fuzzer-tool",
            "fuzz",
            "/bin/true",
            "--tcp-listen-port",
            "4444",
            "--tcp-mode",
            "direct",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        commands.main()
        assert captured["a"].tcp_mode == "direct"

    @pytest.mark.parametrize(
        ("flag", "mode"), [("round", TcpMode.ROUND), ("direct", TcpMode.DIRECT)]
    )
    def test_flag_reaches_fuzzer(self, monkeypatch, tmp_path, flag, mode):
        from fuzzer_tool.cli.commands import cmd_fuzz
        from tests import test_commands_extended

        args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
        args.tcp_listen_port = 4444
        args.tcp_mode = flag
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
        assert cmd_fuzz(args) == 0
        assert seen["tcp_mode"] is mode
