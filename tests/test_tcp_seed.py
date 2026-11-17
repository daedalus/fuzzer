"""Tests for adapters/tcp_seed.py — TCP seed ingestion (--tcp-listen-port)."""

from __future__ import annotations

import socket
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.adapters.tcp_seed import DEFAULT_HOST, TcpSeedInbox

WAIT_S = 5.0


def _send(port: int, payload: bytes) -> None:
    """One connection = one message: write, then close."""
    with socket.create_connection((DEFAULT_HOST, port), timeout=WAIT_S) as s:
        s.sendall(payload)


def _wait_received(inbox: TcpSeedInbox, n: int) -> None:
    """Block until the listener thread has handled *n* messages."""
    deadline = time.monotonic() + WAIT_S
    while inbox.received + inbox.dropped < n:
        assert time.monotonic() < deadline, "listener never handled the message"
        time.sleep(0.01)


@pytest.fixture
def inbox():
    box = TcpSeedInbox(0, max_bytes=64)
    yield box
    box.close()


class TestInbox:
    def test_messages_queued_in_order(self, inbox):
        _send(inbox.port, b"first")
        _wait_received(inbox, 1)
        _send(inbox.port, b"second")
        _wait_received(inbox, 2)

        assert inbox.pop() == b"first"
        assert inbox.pop() == b"second"
        assert inbox.pop() is None

    def test_empty_connection_is_not_a_seed(self, inbox):
        # Falsification: a bare connect/close (port scan, health check) adds nothing.
        socket.create_connection((DEFAULT_HOST, inbox.port), timeout=WAIT_S).close()
        _send(inbox.port, b"x")
        _wait_received(inbox, 1)

        assert inbox.pop() == b"x"
        assert inbox.pop() is None
        assert inbox.received == 1

    def test_oversized_message_truncated(self, inbox):
        # Adversarial: memory stays bounded by max_bytes per message.
        _send(inbox.port, b"A" * (64 * 3))
        _wait_received(inbox, 1)

        assert inbox.pop() == b"A" * 64

    def test_full_inbox_drops_newest(self):
        # Adversarial: a flood cannot grow the queue past capacity.
        box = TcpSeedInbox(0, max_bytes=16, capacity=2)
        try:
            for i in range(4):
                _send(box.port, bytes([0x30 + i]))
                _wait_received(box, i + 1)

            assert (box.received, box.dropped) == (2, 2)
            assert [box.pop(), box.pop(), box.pop()] == [b"0", b"1", None]
        finally:
            box.close()

    def test_stalled_client_does_not_block_others(self):
        # Adversarial: a client that never closes is cut off at conn_timeout,
        # its partial bytes kept. Readers run concurrently, so "next" usually
        # lands first: order is not part of the contract.
        box = TcpSeedInbox(0, max_bytes=16, conn_timeout=0.2)
        try:
            with socket.create_connection((DEFAULT_HOST, box.port), timeout=WAIT_S) as stall:
                stall.sendall(b"partial")
                _send(box.port, b"next")
                _wait_received(box, 2)

            assert sorted([box.pop(), box.pop()]) == [b"next", b"partial"]
        finally:
            box.close()

    def test_receive_logged_with_corpus_id_size_and_peer(self, inbox, capsys):
        # The id matches the corpus file name the seed gets if admitted.
        _send(inbox.port, b"hello")
        _wait_received(inbox, 1)

        out = capsys.readouterr().out
        assert f"Received seed id_{hash_data(b'hello')} with 5 bytes from {DEFAULT_HOST}:" in out

    def test_empty_connection_not_logged(self, inbox, capsys):
        # Falsification: a probe that sends nothing is not announced as a seed.
        socket.create_connection((DEFAULT_HOST, inbox.port), timeout=WAIT_S).close()
        _send(inbox.port, b"x")
        _wait_received(inbox, 1)

        assert capsys.readouterr().out.count("Received seed") == 1

    def test_drop_logged_with_peer(self, capsys):
        # Adversarial: a full inbox says which peer lost a seed, not "Received".
        box = TcpSeedInbox(0, capacity=1)
        try:
            _send(box.port, b"a")
            _wait_received(box, 1)
            _send(box.port, b"b")
            _wait_received(box, 2)

            out = capsys.readouterr().out
            assert out.count("Received seed") == 1
            assert f"Dropped seed id_{hash_data(b'b')} with 1 bytes from {DEFAULT_HOST}:" in out
        finally:
            box.close()

    def test_close_stops_thread_and_frees_port(self):
        box = TcpSeedInbox(0)
        port = box.port
        box.close()

        assert not box._thread.is_alive()
        with pytest.raises(OSError):
            _send(port, b"late")

    def test_port_in_use_raises(self, inbox):
        with pytest.raises(OSError):
            TcpSeedInbox(inbox.port)


class TestFuzzerWiring:
    def test_off_by_default(self, tmp_path):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        assert _make_fuzzer(tmp_path)._tcp_inbox is None

    def test_mutate_returns_tcp_seed_verbatim(self, tmp_path):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        try:
            _send(f._tcp_inbox.port, b"TCPSEED")
            _wait_received(f._tcp_inbox, 1)

            assert bytes(f.mutate(b"parent")) == b"TCPSEED"
            assert f._tcp_inbox.pop() is None
        finally:
            f._tcp_inbox.close()

    def test_tcp_seed_truncated_to_max_len(self, tmp_path):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0, max_len=4)
        try:
            _send(f._tcp_inbox.port, b"0123456789")
            _wait_received(f._tcp_inbox, 1)

            assert f._tcp_inbox.pop() == b"0123"
        finally:
            f._tcp_inbox.close()

    def test_run_executes_seed_and_closes_inbox(self, tmp_path):
        # The message goes through the normal exec path, then teardown closes.
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        _send(f._tcp_inbox.port, b"TCP-EXEC-ME")
        _wait_received(f._tcp_inbox, 1)
        ran = []

        def run_target(data, *a, **k):
            ran.append(bytes(data))
            return 0, ""

        with patch.object(f, "_run_target", side_effect=run_target):
            f.run(max_execs=3)

        assert b"TCP-EXEC-ME" in ran
        assert not f._tcp_inbox._thread.is_alive()


class TestCli:
    def test_flags_parse(self, monkeypatch):
        from fuzzer_tool.cli import commands

        captured = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
        argv = ["fuzzer-tool", "fuzz", "/bin/true", "--tcp-listen-port", "4444"]
        monkeypatch.setattr(sys, "argv", [*argv, "--tcp-listen-host", "0.0.0.0"])
        commands.main()

        assert captured["a"].tcp_listen_port == 4444
        assert captured["a"].tcp_listen_host == "0.0.0.0"

    @pytest.mark.parametrize("port", ["0", "65536", "-1"])
    def test_bad_port_rejected(self, monkeypatch, port):
        from fuzzer_tool.cli import commands

        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: 0)
        monkeypatch.setattr(
            sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--tcp-listen-port", port]
        )
        with pytest.raises(SystemExit):
            commands.main()

    @pytest.mark.parametrize("port", [None, 4444])
    def test_flag_reaches_fuzzer(self, monkeypatch, tmp_path, port):
        from fuzzer_tool.cli.commands import cmd_fuzz
        from tests import test_commands_extended

        args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
        args.tcp_listen_port = port
        args.tcp_listen_host = DEFAULT_HOST
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
        assert cmd_fuzz(args) == 0
        assert seen["tcp_listen_port"] == port
        assert seen["tcp_listen_host"] == DEFAULT_HOST


def _conn_threads() -> list:
    return [t for t in threading.enumerate() if t.name == "tcp-seed-conn" and t.is_alive()]


class TestConcurrency:
    def test_stalled_client_does_not_delay_others(self):
        # A client that never closes held the single listener for the whole
        # conn_timeout; now others are served while it stalls.
        box = TcpSeedInbox(0, max_bytes=16, conn_timeout=30.0)
        stall = socket.create_connection((DEFAULT_HOST, box.port), timeout=WAIT_S)
        try:
            stall.sendall(b"partial")
            _send(box.port, b"quick")
            _wait_received(box, 1)  # within WAIT_S (5 s), far below conn_timeout (30 s)
            assert box.pop() == b"quick"
        finally:
            stall.close()
            box.close()

    def test_parallel_clients_all_delivered(self):
        box = TcpSeedInbox(0)
        payloads = [f"seed-{i:03d}".encode() for i in range(50)]
        try:
            senders = [threading.Thread(target=_send, args=(box.port, p)) for p in payloads]
            for t in senders:
                t.start()
            for t in senders:
                t.join(WAIT_S)
            _wait_received(box, len(payloads))

            got = []
            while (seed := box.pop()) is not None:
                got.append(seed)
            assert sorted(got) == payloads
        finally:
            box.close()

    def test_capacity_exact_under_concurrency(self):
        # Adversarial: racing producers must not overfill the queue.
        box = TcpSeedInbox(0, capacity=5)
        try:
            senders = [
                threading.Thread(target=_send, args=(box.port, f"s{i}".encode())) for i in range(30)
            ]
            for t in senders:
                t.start()
            for t in senders:
                t.join(WAIT_S)
            _wait_received(box, 30)
            assert (box.received, box.dropped) == (5, 25)
        finally:
            box.close()

    def test_connection_cap_holds(self):
        # Adversarial: at most max_connections readers; the next client waits
        # in the backlog until a slot frees, then is served.
        box = TcpSeedInbox(0, conn_timeout=30.0, max_connections=2)
        stalls = [
            socket.create_connection((DEFAULT_HOST, box.port), timeout=WAIT_S) for _ in range(2)
        ]
        try:
            deadline = time.monotonic() + WAIT_S
            while len(_conn_threads()) < 2:
                assert time.monotonic() < deadline, "stalled clients never accepted"
                time.sleep(0.01)

            _send(box.port, b"queued")  # connects via the backlog, not accepted yet
            time.sleep(0.5)
            assert box.received == 0
            assert len(_conn_threads()) == 2

            stalls[0].close()  # frees a slot: empty message, then "queued" is read
            _wait_received(box, 1)
            assert box.pop() == b"queued"
        finally:
            for s in stalls:
                s.close()
            box.close()

    def test_close_ends_open_connections_promptly(self):
        # A stalled client must not hold close() for conn_timeout.
        box = TcpSeedInbox(0, conn_timeout=30.0)
        stall = socket.create_connection((DEFAULT_HOST, box.port), timeout=WAIT_S)
        try:
            deadline = time.monotonic() + WAIT_S
            while not _conn_threads():
                assert time.monotonic() < deadline, "client never accepted"
                time.sleep(0.01)

            t0 = time.monotonic()
            box.close()
            assert time.monotonic() - t0 < WAIT_S
            assert _conn_threads() == []
        finally:
            stall.close()

    def test_bad_max_connections_rejected(self):
        with pytest.raises(ValueError):
            TcpSeedInbox(0, max_connections=0)


class TestProcessingVisible:
    """Every received seed runs once and says so: "Ran seed id_<h>: ..."."""

    def test_pop_seed_carries_the_received_id(self, inbox):
        _send(inbox.port, b"hello")
        _wait_received(inbox, 1)
        assert inbox.pending == 1

        seed = inbox.pop_seed()
        assert (seed.data, seed.seed_id) == (b"hello", hash_data(b"hello"))
        assert inbox.pending == 0 and inbox.pop_seed() is None

    def test_regression_seed_equal_to_parent_not_rerolled(self, tmp_path):
        # The dedup gate re-rolled a TCP seed identical to the fuzzed parent:
        # received, logged, never run.
        from tests.test_regression_end_of_run_persistence import SEED, _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        try:
            _send(f._tcp_inbox.port, SEED)
            _wait_received(f._tcp_inbox, 1)
            assert bytes(f._dedup_mutate(SEED)) == SEED
        finally:
            f._tcp_inbox.close()

    def test_regression_already_executed_seed_runs_again(self, tmp_path):
        # Sent twice: the exec bloom flagged the second copy and re-rolled it.
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        try:
            for _ in range(2):
                _send(f._tcp_inbox.port, b"TWICE")
            _wait_received(f._tcp_inbox, 2)
            assert [bytes(f._dedup_mutate(b"parent")) for _ in range(2)] == [b"TWICE", b"TWICE"]
        finally:
            f._tcp_inbox.close()

    def test_run_logs_ran_line_and_summary(self, tmp_path, capsys):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        for payload in (b"TCP-ONE", b"TCP-TWO"):
            _send(f._tcp_inbox.port, payload)
        _wait_received(f._tcp_inbox, 2)

        with patch.object(f, "_run_target", return_value=(0, "")):
            f.run(max_execs=5)

        out = capsys.readouterr().out
        for payload in (b"TCP-ONE", b"TCP-TWO"):
            line = next(ln for ln in out.splitlines() if f"Ran seed id_{hash_data(payload)}" in ln)
            assert "rc=0" in line and "edges" in line
            assert any(word in line for word in ("admitted", "crash", "boring"))
        assert "TCP seeds: received 2, ran 2, dropped 0, unrun 0" in out

    def test_non_tcp_rounds_not_logged(self, tmp_path, capsys):
        # Falsification: ordinary mutants never print "Ran seed".
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        with patch.object(f, "_run_target", return_value=(0, "")):
            f.run(max_execs=5)
        out = capsys.readouterr().out
        assert "Ran seed" not in out
        assert "TCP seeds: received 0, ran 0, dropped 0, unrun 0" in out

    def test_unrun_counted_at_exit(self, tmp_path, capsys):
        # Adversarial: seeds still queued when the run stops are reported, not hidden.
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, tcp_listen_port=0)
        for i in range(3):
            _send(f._tcp_inbox.port, b"Q%d" % i)
        _wait_received(f._tcp_inbox, 3)
        f._tcp_inbox.close()
        f._report_tcp_seeds()
        assert "TCP seeds: received 3, ran 0, dropped 0, unrun 3" in capsys.readouterr().out


class TestOutputAtomic:
    def test_regression_received_line_is_one_write(self, inbox, monkeypatch):
        # print() writes text and newline separately: the main thread's
        # "Ran seed" line landed between them ("...:54218[*] Ran seed ...").
        writes = []

        class Recorder:
            def write(self, s):
                writes.append(s)
                return len(s)

            def flush(self):
                pass

        monkeypatch.setattr(sys, "stdout", Recorder())
        _send(inbox.port, b"hello")
        _wait_received(inbox, 1)

        line = [w for w in writes if "Received seed" in w]
        assert len(line) == 1 and line[0].endswith("\n")
        assert "\n" not in [w for w in writes]  # no lone newline write
