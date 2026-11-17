"""Tests for tools/dir_seeds.py: walk directories, send every file as a TCP seed."""

import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import chrome_cache_seeds as ccs  # noqa: E402
import dir_seeds as ds  # noqa: E402

from fuzzer_tool.adapters.tcp_seed import TcpSeedInbox  # noqa: E402

Media = ccs.Media
PNG = b"\x89PNG\r\n\x1a\n" + b"P" * 20
JPG = b"\xff\xd8\xff\xe0" + b"J" * 20
TXT = b"plain text seed"
WAIT_S = 5.0


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def _recorder(monkeypatch):
    """Patch tcp_sink with a sender that records every seed."""
    sent = []
    monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)
    return sent


class TestWalk:
    def test_recursive_order_files_before_subdirs(self, tmp_path, monkeypatch):
        # Each directory: its own files (sorted), then its subdirectories (sorted).
        root = _tree(
            tmp_path / "c", {"b/2": b"b2", "a/1": b"a1", "z": b"z", "y": b"y", "a/q/0": b"aq0"}
        )
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [b"y", b"z", b"a1", b"aq0", b"b2"]

    def test_every_file_sent_by_default_typed_or_not(self, tmp_path, monkeypatch, capsys):
        root = _tree(tmp_path / "c", {"1": PNG, "2": JPG, "3": TXT})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [PNG, JPG, TXT]
        assert "sent 3 seeds (jpg: 1, other: 1, png: 1)" in capsys.readouterr().out

    def test_types_filter_by_leading_magic(self, tmp_path, monkeypatch):
        # A PNG magic later in a file does not make it a PNG.
        root = _tree(tmp_path / "c", {"1": PNG, "2": JPG, "3": TXT + PNG})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root), "--types", "png"]) == 0
        assert sent == [PNG]

    def test_other_type_selects_untyped_files(self, tmp_path, monkeypatch):
        root = _tree(tmp_path / "c", {"1": PNG, "2": TXT})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root), "--types", "other"]) == 0
        assert sent == [TXT]

    def test_max_bytes_reads_only_the_cap(self, tmp_path, monkeypatch):
        root = _tree(tmp_path / "c", {"big": b"B" * 5000})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root), "--max-bytes", "10"]) == 0
        assert sent == [b"B" * 10]

    def test_empty_and_duplicate_files_skipped(self, tmp_path, monkeypatch, capsys):
        root = _tree(tmp_path / "c", {"1": TXT, "2": b"", "3": TXT})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [TXT]
        assert "dup 1" in capsys.readouterr().out

    def test_file_argument_sent_directly(self, tmp_path, monkeypatch):
        f = _tree(tmp_path, {"one": TXT}) / "one"
        sent = _recorder(monkeypatch)

        assert ds.main([str(f)]) == 0
        assert sent == [TXT]


class TestAdversarial:
    def test_symlink_loop_does_not_hang(self, tmp_path, monkeypatch):
        # A directory link back to its parent is not followed.
        root = _tree(tmp_path / "c", {"1": TXT})
        (root / "loop").symlink_to(root, target_is_directory=True)
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [TXT]

    def test_dangling_link_counted_unreadable(self, tmp_path, monkeypatch, capsys):
        root = _tree(tmp_path / "c", {"1": TXT})
        (root / "0dead").symlink_to(tmp_path / "gone")
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [TXT]
        assert "unreadable 1" in capsys.readouterr().out

    def test_fifo_skipped_not_blocking(self, tmp_path, monkeypatch, capsys):
        # Opening a FIFO with no writer blocks forever; pytest's timeout catches a regression.
        root = _tree(tmp_path / "c", {"1": TXT})
        os.mkfifo(root / "0pipe")
        sent = _recorder(monkeypatch)

        assert ds.main([str(root)]) == 0
        assert sent == [TXT]
        assert "unreadable 1" in capsys.readouterr().out

    def test_sent_bloom_inside_walked_dir_not_sent(self, tmp_path, monkeypatch):
        # The state file must never be fed to the target, on any run.
        root = _tree(tmp_path / "c", {"1": TXT})
        sent = _recorder(monkeypatch)
        argv = [str(root), "--sent-bloom", str(root / "sent.pkl.gz")]

        assert ds.main(argv) == 0
        (root / "2").write_bytes(PNG)
        assert ds.main(argv) == 0
        assert sent == [TXT, PNG]

    def test_missing_path_fails(self, tmp_path, capsys):
        # Falsification: a typo'd path is an error, not "sent 0".
        assert ds.main([str(tmp_path / "nope"), "--dry-run"]) == 1
        assert "nope" in capsys.readouterr().err


class TestMain:
    def test_live_inbox_receives_every_file(self, tmp_path, capsys):
        root = _tree(tmp_path / "c", {"a": PNG, "sub/b": TXT})
        inbox = TcpSeedInbox(0)
        try:
            assert ds.main([str(root), "--port", str(inbox.port), "--delay", "0"]) == 0

            deadline = time.monotonic() + WAIT_S
            while inbox.received < 2:
                assert time.monotonic() < deadline, "seeds never arrived"
                time.sleep(0.01)
            assert [inbox.pop(), inbox.pop()] == [PNG, TXT]
        finally:
            inbox.close()

    def test_sent_bloom_second_run_sends_nothing(self, tmp_path, monkeypatch, capsys):
        root = _tree(tmp_path / "c", {"1": PNG, "2": TXT})
        sent = _recorder(monkeypatch)
        argv = [str(root), "--sent-bloom", str(tmp_path / "sent.pkl.gz")]

        assert ds.main(argv) == 0
        assert ds.main(argv) == 0
        assert sent == [PNG, TXT]
        assert "already sent 2" in capsys.readouterr().out

    def test_dry_run_sends_nothing(self, tmp_path, capsys):
        root = _tree(tmp_path / "c", {"1": PNG, "2": TXT})
        # Port 1 never listens here: a send attempt would fail the run.
        assert ds.main([str(root), "--port", "1", "--dry-run"]) == 0
        assert "would send 2 seeds" in capsys.readouterr().out

    def test_shared_flags_accepted(self, tmp_path):
        # Same parameters as chrome_cache_seeds.py / blockfile_seeds.py.
        root = _tree(tmp_path / "c", {})
        root.mkdir(exist_ok=True)
        argv = [
            str(root),
            "--dry-run",
            "--types",
            "png,other",
            "--host",
            "127.0.0.1",
            "--port",
            "1",
        ]
        argv += ["--delay", "0", "--max-bytes", "10", "--retries", "0", "--retry-backoff", "0"]
        argv += ["--root", str(tmp_path)]
        assert ds.main(argv) == 0


class TestMaxSize:
    def test_larger_files_skipped_and_counted(self, tmp_path, monkeypatch, capsys):
        root = _tree(tmp_path / "c", {"1": TXT, "2": b"x" * 500})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root), "--max-size", str(len(TXT))]) == 0
        assert sent == [TXT]
        assert "too big 1" in capsys.readouterr().out

    def test_file_size_not_truncated_size(self, tmp_path, monkeypatch):
        # Adversarial: --max-bytes 4 reads 4 B, but the 500 B file is too big.
        root = _tree(tmp_path / "c", {"big": b"x" * 500})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--max-bytes", "4", "--max-size", "100"]) == 0
        assert sent == []
