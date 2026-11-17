"""Tests for tools/tcp_corpus.py: TCP messages -> fuzzer corpus, nothing sent anywhere."""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import chrome_cache_seeds as ccs  # noqa: E402
import tcp_corpus as tc  # noqa: E402

from fuzzer_tool.adapters.filesystem import hash_data  # noqa: E402
from fuzzer_tool.adapters.tcp_seed import DEFAULT_HOST, TcpSeed  # noqa: E402

WAIT_S = 5.0


def _seed(data: bytes, seed_id: str | None = None) -> TcpSeed:
    return TcpSeed(data, seed_id or hash_data(data))


def _sink(tmp_path, save_every=tc.SAVE_EVERY):
    log = ccs.SentLog(tmp_path / "bloom.pkl.gz", ccs.Persist.WRITE)
    return tc.CorpusSink(tmp_path / "corpus", log, save_every), log


def _seed_path(corpus: Path, data: bytes) -> Path:
    # The fuzzer's layout: seeds/<h[:2]>/id_<h> (import_corpus, save_to_corpus).
    h = hash_data(data)
    return corpus / "seeds" / h[:2] / f"id_{h}"


class TestSink:
    def test_writes_fuzzer_corpus_layout(self, tmp_path):
        sink, _ = _sink(tmp_path)
        assert sink.take(_seed(b"hello")) is tc.Taken.SAVED
        assert _seed_path(tmp_path / "corpus", b"hello").read_bytes() == b"hello"

    def test_repeat_in_run_not_rewritten(self, tmp_path):
        sink, _ = _sink(tmp_path)
        sink.take(_seed(b"x"))
        assert sink.take(_seed(b"x")) is tc.Taken.DUPLICATE
        assert sink.saved == 1 and sink.duplicates == 1

    def test_seed_already_on_disk_recorded_not_rewritten(self, tmp_path):
        # Falsification: an existing corpus file is a duplicate even with an empty bloom.
        path = _seed_path(tmp_path / "corpus", b"old")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old")
        before = path.stat().st_mtime_ns

        sink, log = _sink(tmp_path)
        assert sink.take(_seed(b"old")) is tc.Taken.DUPLICATE
        assert path.stat().st_mtime_ns == before
        assert log.seen(b"old")  # remembered for the next run

    def test_hash_twin_kept_beside(self, tmp_path):
        # Adversarial: other bytes under the same id (forged XXH64 collision)
        # are kept as id_<h>~1, never overwriting the first.
        sink, _ = _sink(tmp_path)
        sink.take(_seed(b"first", seed_id="00aa"))
        assert sink.take(_seed(b"second", seed_id="00aa")) is tc.Taken.SAVED
        sub = tmp_path / "corpus" / "seeds" / "00"
        assert (sub / "id_00aa").read_bytes() == b"first"
        assert (sub / "id_00aa~1").read_bytes() == b"second"

    def test_no_temp_files_left(self, tmp_path):
        sink, _ = _sink(tmp_path)
        sink.take(_seed(b"atomic"))
        assert not list((tmp_path / "corpus").rglob("*.tmp"))

    def test_bloom_saved_every_n_and_survives_restart(self, tmp_path):
        sink, _ = _sink(tmp_path, save_every=2)
        sink.take(_seed(b"a"))
        assert not (tmp_path / "bloom.pkl.gz").exists()
        sink.take(_seed(b"b"))
        assert (tmp_path / "bloom.pkl.gz").exists()

        # A new run sees both as already ingested, even if the files were pruned.
        for p in (tmp_path / "corpus").rglob("id_*"):
            p.unlink()
        again, _ = _sink(tmp_path)
        assert again.take(_seed(b"a")) is tc.Taken.DUPLICATE


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


def _send(port: int, payload: bytes) -> None:
    deadline = time.monotonic() + WAIT_S
    while True:
        try:
            with socket.create_connection((DEFAULT_HOST, port), timeout=WAIT_S) as s:
                s.sendall(payload)
            return
        except ConnectionRefusedError:
            # The listener thread may not be bound yet: wait for it, bounded.
            assert time.monotonic() < deadline, "listener never came up"
            time.sleep(0.05)


class TestMain:
    def test_listen_ingest_and_stop(self, tmp_path, capsys):
        port = _free_port()
        corpus, bloom = tmp_path / "corpus", tmp_path / "bloom.pkl.gz"
        stop = threading.Event()
        argv = ["--port", str(port), "--corpus", str(corpus), "--bloom", str(bloom)]
        runner = threading.Thread(target=tc.main, args=(argv, stop))
        runner.start()
        try:
            for payload in (b"one", b"two", b"one"):
                _send(port, payload)
            deadline = time.monotonic() + WAIT_S
            while len(list(corpus.rglob("id_*"))) < 2:
                assert time.monotonic() < deadline, "seeds never written"
                time.sleep(0.02)
        finally:
            stop.set()
            runner.join(WAIT_S)

        assert not runner.is_alive()
        assert sorted(p.read_bytes() for p in corpus.rglob("id_*")) == [b"one", b"two"]
        assert bloom.exists()
        out = capsys.readouterr().out
        assert "saved 2, duplicate 1" in out

    def test_queued_seeds_written_on_stop(self, tmp_path):
        # Adversarial: seeds received but not yet written when stop comes are not lost.
        corpus = tmp_path / "corpus"
        sink, log = _sink(tmp_path)
        inbox = tc.TcpSeedInbox(0)
        try:
            for p in (b"q1", b"q2", b"q3"):
                _send(inbox.port, p)
            deadline = time.monotonic() + WAIT_S
            while inbox.received < 3:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            stop = threading.Event()
            stop.set()  # stop before the loop takes anything
            tc.serve(inbox, sink, stop)
        finally:
            inbox.close()
        assert sorted(p.read_bytes() for p in corpus.rglob("id_*")) == [b"q1", b"q2", b"q3"]

    def test_corpus_path_is_a_file_fails(self, tmp_path, capsys):
        f = tmp_path / "corpus"
        f.write_bytes(b"not a dir")
        argv = ["--port", str(_free_port()), "--corpus", str(f), "--bloom", str(tmp_path / "b")]
        assert tc.main(argv, threading.Event()) == 1
        assert "not a directory" in capsys.readouterr().err

    @pytest.mark.parametrize("missing", ["--port", "--corpus", "--bloom"])
    def test_required_args(self, tmp_path, missing):
        args = {"--port": "4444", "--corpus": str(tmp_path / "c"), "--bloom": str(tmp_path / "b")}
        del args[missing]
        argv = [x for kv in args.items() for x in kv]
        with pytest.raises(SystemExit):
            tc.main(argv, threading.Event())
