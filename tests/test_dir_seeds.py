"""Tests for tools/dir_seeds.py: walk directories, send every file as a TCP seed."""

import os
import re
import sys
import time
from pathlib import Path

import pytest

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


class TestWalkOrder:
    TREE = {"y": b"y", "z": b"z", "a/1": b"a1", "a/q/0": b"aq0", "b/2": b"b2"}

    def test_arranger_applied_to_files_and_dirs_at_every_level(self, tmp_path):
        # Deterministic stand-in for the shuffle (Rule 39): reverse each list.
        root = _tree(tmp_path / "c", self.TREE)

        def backwards(items):
            items.sort(reverse=True)

        got = [p.relative_to(root).as_posix() for p in ds.walk([root], arrange=backwards)]
        assert got == ["z", "y", "b/2", "a/1", "a/q/0"]

    def test_default_order_unchanged(self, tmp_path):
        # Falsification: without --order random the walk stays sorted.
        root = _tree(tmp_path / "c", self.TREE)
        got = [p.relative_to(root).as_posix() for p in ds.walk([root])]
        assert got == ["y", "z", "a/1", "a/q/0", "b/2"]

    def test_same_seed_same_order(self, tmp_path, monkeypatch):
        # Control (Rule 46): one seed, two runs, identical order.
        root = _tree(tmp_path / "c", {f"d{i % 4}/f{i:02d}": b"%d" % i for i in range(40)})
        runs = []
        for _ in range(2):
            sent = _recorder(monkeypatch)
            assert ds.main([str(root), "--order", "random", "--seed", "7"]) == 0
            runs.append(sent)
        assert runs[0] == runs[1]
        assert sorted(runs[0]) == sorted(b"%d" % i for i in range(40))

    def test_seed_changes_order(self, tmp_path, monkeypatch):
        # 40 files: two seeds agreeing by chance is ~1/40!.
        root = _tree(tmp_path / "c", {f"f{i:02d}": b"%d" % i for i in range(40)})
        runs = []
        for seed in ("1", "2"):
            sent = _recorder(monkeypatch)
            assert ds.main([str(root), "--order", "random", "--seed", seed]) == 0
            runs.append(sent)
        assert runs[0] != runs[1]
        assert runs[0] != [b"%d" % i for i in range(40)]  # not the sorted order

    def test_top_level_paths_shuffled_too(self, tmp_path, monkeypatch):
        dirs = [_tree(tmp_path / f"p{i:02d}", {"f": b"%d" % i}) for i in range(30)]
        sent = _recorder(monkeypatch)
        assert ds.main([*map(str, dirs), "--order", "random", "--seed", "3"]) == 0
        assert sorted(sent) == sorted(b"%d" % i for i in range(30))
        assert sent != [b"%d" % i for i in range(30)]

    def test_unseeded_run_prints_reproducible_seed(self, tmp_path, monkeypatch, capsys):
        # Adversarial: a random run must be repeatable from its printed seed.
        root = _tree(tmp_path / "c", {f"f{i:02d}": b"%d" % i for i in range(20)})
        first = _recorder(monkeypatch)
        assert ds.main([str(root), "--order", "random"]) == 0
        seed = re.search(r"walk order: random \(seed (\d+)\)", capsys.readouterr().out).group(1)

        again = _recorder(monkeypatch)
        assert ds.main([str(root), "--order", "random", "--seed", seed]) == 0
        assert again == first

    def test_seed_without_random_order_rejected(self, tmp_path):
        root = _tree(tmp_path / "c", {"f": b"x"})
        with pytest.raises(SystemExit):
            ds.main([str(root), "--seed", "5"])


def _tar(members: dict[str, bytes], mode: str = "w") -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        d = tarfile.TarInfo("adir")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("alink")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    return buf.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sub/", b"")  # a directory entry: skipped
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestDecompress:
    @pytest.mark.parametrize("codec", ["gzip", "bz2", "lzma"])
    def test_single_stream_sent_decompressed(self, tmp_path, monkeypatch, codec):
        import importlib

        packed = importlib.import_module(codec).compress(PNG)
        root = _tree(tmp_path / "c", {"img.bin": packed})
        sent = _recorder(monkeypatch)

        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == [PNG]

    def test_zip_members_in_order_dirs_skipped(self, tmp_path, monkeypatch):
        root = _tree(tmp_path / "c", {"a.zip": _zip({"one.png": PNG, "sub/two.txt": TXT})})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == [PNG, TXT]

    @pytest.mark.parametrize("mode", ["w", "w:gz", "w:bz2", "w:xz"])
    def test_tar_members_dirs_and_links_skipped(self, tmp_path, monkeypatch, mode):
        root = _tree(tmp_path / "c", {"a.tar": _tar({"x/one": JPG, "two": TXT}, mode)})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == [JPG, TXT]

    def test_without_flag_archive_sent_as_is(self, tmp_path, monkeypatch):
        # Falsification: the option is off by default.
        import gzip

        packed = gzip.compress(PNG)
        root = _tree(tmp_path / "c", {"img.gz": packed})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root)]) == 0
        assert sent == [packed]

    def test_plain_files_unchanged(self, tmp_path, monkeypatch):
        root = _tree(tmp_path / "c", {"1": PNG, "2": TXT})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == [PNG, TXT]

    def test_types_filter_applies_to_members(self, tmp_path, monkeypatch):
        root = _tree(tmp_path / "c", {"a.zip": _zip({"one.png": PNG, "t.txt": TXT})})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress", "--types", "png"]) == 0
        assert sent == [PNG]

    def test_corrupt_archive_counted_others_sent(self, tmp_path, monkeypatch, capsys):
        # Adversarial: gzip magic, then garbage.
        root = _tree(tmp_path / "c", {"0bad.gz": b"\x1f\x8b\x08\x00" + b"\xff" * 40, "1": TXT})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == [TXT]
        assert "unreadable 1" in capsys.readouterr().out

    def test_bomb_read_only_up_to_max_bytes(self, tmp_path, monkeypatch):
        # Adversarial: 64 MiB of zeros in ~64 KiB; only --max-bytes is kept.
        import gzip

        root = _tree(tmp_path / "c", {"bomb.gz": gzip.compress(bytes(64 << 20))})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress", "--max-bytes", "10"]) == 0
        assert sent == [bytes(10)]

    def test_max_size_judged_on_decompressed_size(self, tmp_path, monkeypatch, capsys):
        import gzip

        root = _tree(
            tmp_path / "c", {"big.gz": gzip.compress(bytes(5000)), "s.gz": gzip.compress(TXT)}
        )
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress", "--max-size", "1000"]) == 0
        assert sent == [TXT]
        assert "too big 1" in capsys.readouterr().out

    def test_stream_over_hard_cap_is_too_big(self, tmp_path, monkeypatch, capsys):
        # A stream larger than MAX_UNPACKED is cut off and judged too big.
        import gzip

        import unpack

        monkeypatch.setattr(unpack, "MAX_UNPACKED", 1000)
        root = _tree(tmp_path / "c", {"z.gz": gzip.compress(bytes(5000))})
        sent = _recorder(monkeypatch)
        assert ds.main([str(root), "--decompress"]) == 0
        assert sent == []
        assert "too big 1" in capsys.readouterr().out
