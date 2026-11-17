"""Tests for tools/chrome_cache_seeds.py: carve media from Chrome/Chromium caches, send over TCP."""

import struct
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import chrome_cache_seeds as ccs  # noqa: E402

from fuzzer_tool.adapters.tcp_seed import TcpSeedInbox  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"P" * 20
JPG = b"\xff\xd8\xff\xe0" + b"J" * 20
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"W" * 8
MP4 = b"\x00\x00\x00\x18ftypisom" + b"M" * 12
WAIT_S = 5.0


def _entry(body: bytes, key: bytes = b"1/0/https://example.com/x") -> bytes:
    """A simple-cache entry file: header, key, body, EOF trailer, HTTP headers."""
    header = struct.pack("<QIIII", ccs.ENTRY_MAGIC, 5, len(key), 0, 0)
    eof = ccs.TRAILER + struct.pack("<III", 1, 0, len(body)) + b"\x00" * 4
    return header + key + body + eof + b"HTTP/1.1 200 OK\x00content-type: x\x00\x00"


def _profile(root: Path, name: str, files: dict[str, bytes]) -> Path:
    cache = root / name / "Cache" / "Cache_Data"
    cache.mkdir(parents=True)
    (cache / "index").write_bytes(b"")
    for fname, data in files.items():
        (cache / fname).write_bytes(data)
    return cache


class TestSniff:
    @pytest.mark.parametrize(
        ("body", "media"),
        [(PNG, ccs.Media.PNG), (JPG, ccs.Media.JPG), (WEBP, ccs.Media.WEBP), (MP4, ccs.Media.MP4)],
    )
    def test_known_types(self, body, media):
        assert ccs.sniff(b"hdr" + body) == (media, len(b"hdr"))

    def test_earliest_signature_wins(self):
        # A JPEG body that happens to contain a PNG magic later is still a JPEG.
        assert ccs.sniff(JPG + PNG) == (ccs.Media.JPG, 0)

    def test_no_signature(self):
        # Falsification: text/HTML bodies are not seeds.
        assert ccs.sniff(b"<html>hello</html>") is None

    def test_riff_without_webp_is_not_webp(self):
        # Adversarial: a WAV/AVI RIFF must not be taken as WebP.
        assert ccs.sniff(b"RIFF\x10\x00\x00\x00WAVEfmt ") is None

    def test_ftyp_needs_room_for_box_size(self):
        # Adversarial: "ftyp" at offset < 4 has no size field before it.
        assert ccs.sniff(b"ftypisom") is None


class TestCarve:
    def test_body_cut_at_trailer(self, tmp_path):
        p = tmp_path / "a_0"
        p.write_bytes(_entry(PNG))
        assert ccs.carve(p) == (ccs.Media.PNG, PNG)

    def test_magic_in_url_key_ignored(self, tmp_path):
        # Adversarial: the key (URL) is skipped, so "GIF89a" in it is not a body.
        p = tmp_path / "k_0"
        p.write_bytes(_entry(PNG, key=b"1/0/https://x.test/GIF89a.gif"))
        assert ccs.carve(p) == (ccs.Media.PNG, PNG)

    def test_max_bytes_bounds_read(self, tmp_path):
        p = tmp_path / "m_0"
        p.write_bytes(_entry(JPG))
        assert ccs.carve(p, max_bytes=5) == (ccs.Media.JPG, JPG[:5])

    def test_no_trailer_reads_to_eof(self, tmp_path):
        # Blockfile f_* files carry no trailer.
        p = tmp_path / "f_000001"
        p.write_bytes(JPG)
        assert ccs.carve(p) == (ccs.Media.JPG, JPG)

    def test_signature_past_header_window_ignored(self, tmp_path):
        # Adversarial: a magic deep inside a non-media body is not a file start.
        p = tmp_path / "b_0"
        p.write_bytes(_entry(b"x" * (ccs.SNIFF_WINDOW + 10) + PNG))
        assert ccs.carve(p) is None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e_0"
        p.write_bytes(b"")
        assert ccs.carve(p) is None


class TestDiscovery:
    def test_every_profile_found(self, tmp_path):
        root = tmp_path / "google-chrome"
        a = _profile(root, "Default", {})
        b = _profile(root, "Profile 1", {})
        c = _profile(root, "Profile 7", {})
        (root / "System Profile").mkdir()  # no cache: skipped

        assert ccs.find_caches([root]) == [a, b, c]

    def test_legacy_cache_layout(self, tmp_path):
        # Older builds keep entries directly in <profile>/Cache.
        legacy = tmp_path / "chromium" / "Default" / "Cache"
        legacy.mkdir(parents=True)
        (legacy / "index").write_bytes(b"")
        assert ccs.find_caches([tmp_path / "chromium"]) == [legacy]

    def test_missing_root_and_file_named_cache(self, tmp_path):
        # Adversarial: absent roots and a regular file called "Cache" are skipped.
        root = tmp_path / "chromium"
        (root / "Default").mkdir(parents=True)
        (root / "Default" / "Cache").write_bytes(b"not a dir")
        assert ccs.find_caches([root, tmp_path / "nope"]) == []

    def test_linux_roots_cover_both_browsers_and_sandboxes(self, tmp_path):
        roots = ccs.browser_roots("linux", tmp_path, {})
        cache = tmp_path / ".cache"
        for want in (
            cache / "google-chrome",
            cache / "chromium",
            tmp_path / "snap" / "chromium" / "common" / ".cache" / "chromium",
            tmp_path / ".var" / "app" / "org.chromium.Chromium" / "cache" / "chromium",
            tmp_path / ".var" / "app" / "com.google.Chrome" / "cache" / "google-chrome",
        ):
            assert want in roots

    def test_linux_honours_xdg_cache_home(self, tmp_path):
        roots = ccs.browser_roots("linux", tmp_path, {"XDG_CACHE_HOME": "/xdg"})
        assert Path("/xdg/google-chrome") in roots
        assert Path("/xdg/chromium") in roots

    def test_macos_and_windows_roots(self, tmp_path):
        mac = ccs.browser_roots("darwin", tmp_path, {})
        assert tmp_path / "Library/Caches/Google/Chrome" in mac
        assert tmp_path / "Library/Caches/Chromium" in mac

        win = ccs.browser_roots("win32", tmp_path, {"LOCALAPPDATA": str(tmp_path / "L")})
        assert tmp_path / "L/Google/Chrome/User Data" in win
        assert tmp_path / "L/Chromium/User Data" in win


class TestSeeds:
    def test_filter_dedup_and_truncate(self, tmp_path):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG), "3_0": b"<html>"})
        _profile(root, "Profile 1", {"1_0": _entry(PNG)})  # same body: sent once
        caches = ccs.find_caches([root])

        got = list(ccs.iter_seeds(caches, {ccs.Media.PNG}, max_bytes=8))
        assert got == [(ccs.Media.PNG, PNG[:8])]

    def test_unreadable_file_counted_not_fatal(self, tmp_path):
        # Adversarial: a dangling entry (evicted mid-scan) is skipped.
        cache = _profile(tmp_path / "chromium", "Default", {"1_0": _entry(JPG)})
        (cache / "0_0").symlink_to(tmp_path / "gone")
        stats = ccs.ScanStats()

        got = list(ccs.iter_seeds([cache], set(ccs.Media), stats=stats))
        assert got == [(ccs.Media.JPG, JPG)]
        assert stats.unreadable == 1


class TestMain:
    def test_sends_every_profile_to_inbox(self, tmp_path, capsys):
        root = tmp_path / "google-chrome"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        _profile(root, "Profile 2", {"1_0": _entry(WEBP)})
        inbox = TcpSeedInbox(0)
        try:
            argv = ["--root", str(root), "--port", str(inbox.port), "--delay", "0"]
            assert ccs.main(argv) == 0

            deadline = time.monotonic() + WAIT_S
            while inbox.received < 2:
                assert time.monotonic() < deadline, "seeds never arrived"
                time.sleep(0.01)
            assert [inbox.pop(), inbox.pop()] == [PNG, WEBP]
        finally:
            inbox.close()

    def test_dry_run_sends_nothing(self, tmp_path, capsys):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(MP4)})
        # Port 1 is never listening here: a send attempt would fail the run.
        assert ccs.main(["--root", str(root), "--port", "1", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "png: 1" in out and "mp4: 1" in out

    def test_no_cache_found_fails(self, tmp_path, capsys):
        # Falsification: nothing to send is an error, not a silent success.
        assert ccs.main(["--root", str(tmp_path / "none"), "--dry-run"]) == 1

    def test_failure_reports_count_sent(self, tmp_path, capsys, monkeypatch):
        # Adversarial: the fuzzer dies mid-run; the user learns how many got through.
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG), "3_0": _entry(WEBP)})
        calls = []

        def sink(_host, _port, _delay):
            def send(data):
                calls.append(data)
                if len(calls) == 2:
                    raise ConnectionRefusedError(111, "Connection refused")

            return send

        monkeypatch.setattr(ccs, "tcp_sink", sink)
        assert ccs.main(["--root", str(root), "--retries", "0"]) == 1
        assert "sent 1 seeds before failure" in capsys.readouterr().err

    def test_progress_line_every_n(self, tmp_path, capsys, monkeypatch):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG), "3_0": _entry(WEBP)})
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: lambda _d: None)
        monkeypatch.setattr(ccs, "PROGRESS_EVERY", 2)

        assert ccs.main(["--root", str(root)]) == 0
        out = capsys.readouterr().out
        assert out.count("[*] sent 2 seeds so far") == 1
        assert "[*] sent 3 seeds (" in out

    def test_refused_connection_fails(self, tmp_path, capsys):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        inbox = TcpSeedInbox(0)
        port = inbox.port
        inbox.close()  # nothing listens on this port any more
        argv = ["--root", str(root), "--port", str(port), "--delay", "0", "--retries", "0"]
        assert ccs.main(argv) == 1


class TestRetry:
    @staticmethod
    def _flaky(fail_times: int):
        """A sender that refuses *fail_times* calls, then accepts; logs every call."""
        calls = []

        def send(data):
            calls.append(data)
            if len(calls) <= fail_times:
                raise ConnectionRefusedError(111, "Connection refused")

        return send, calls

    def test_recovers_after_failures_with_backoff(self, capsys):
        send, calls = self._flaky(2)
        naps = []
        retrying = ccs.with_retry(send, retries=3, backoff=0.5, sleep=naps.append)

        retrying(b"seed")
        assert calls == [b"seed"] * 3  # same bytes resent, not skipped
        assert naps == [0.5, 1.0]  # backoff doubles per attempt
        assert capsys.readouterr().err.count("retry") == 2

    def test_gives_up_after_retries(self):
        # Falsification: a dead fuzzer still ends the run, after retries+1 attempts.
        send, calls = self._flaky(99)
        retrying = ccs.with_retry(send, retries=2, backoff=0.1, sleep=lambda _s: None)

        with pytest.raises(ConnectionRefusedError):
            retrying(b"seed")
        assert len(calls) == 3

    def test_backoff_capped(self):
        # Adversarial: many retries never wait longer than MAX_BACKOFF_S per attempt.
        send, _ = self._flaky(99)
        naps = []
        retrying = ccs.with_retry(send, retries=12, backoff=1.0, sleep=naps.append)

        with pytest.raises(ConnectionRefusedError):
            retrying(b"seed")
        assert max(naps) == ccs.MAX_BACKOFF_S
        assert naps[:3] == [1.0, 2.0, 4.0]

    def test_non_network_error_not_retried(self):
        # Adversarial: a bug (TypeError) surfaces at once instead of looping.
        calls = []

        def send(data):
            calls.append(data)
            raise TypeError("bug")

        retrying = ccs.with_retry(send, retries=5, backoff=0.1, sleep=lambda _s: None)
        with pytest.raises(TypeError):
            retrying(b"seed")
        assert len(calls) == 1

    def test_main_survives_fuzzer_restart(self, tmp_path, capsys, monkeypatch):
        # The fuzzer is down for one attempt; every seed still goes out once.
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG)})
        send, calls = self._flaky(1)
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: send)

        argv = ["--root", str(root), "--retries", "2", "--retry-backoff", "0"]
        assert ccs.main(argv) == 0
        assert calls == [PNG, PNG, JPG]
        assert "[*] sent 2 seeds (" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["--retries", "--retry-backoff"])
    def test_negative_values_rejected(self, tmp_path, flag):
        with pytest.raises(SystemExit):
            ccs.main(["--root", str(tmp_path), flag, "-1"])


class TestSentBloom:
    @staticmethod
    def _recorder(monkeypatch, fail_on: bytes | None = None):
        """Patch tcp_sink with a sender that records seeds; refuses *fail_on*."""
        sent = []

        def send(data):
            if data == fail_on:
                raise ConnectionRefusedError(111, "Connection refused")
            sent.append(data)

        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: send)
        return sent

    def test_second_run_skips_seeds_already_sent(self, tmp_path, capsys, monkeypatch):
        root = tmp_path / "chromium"
        cache = _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG)})
        bloom = tmp_path / "state" / "sent.pkl.gz"  # parent created on save
        argv = ["--root", str(root), "--sent-bloom", str(bloom)]
        sent = self._recorder(monkeypatch)

        assert ccs.main(argv) == 0
        assert sent == [PNG, JPG]

        # Second run: only the newly cached body goes out.
        (cache / "3_0").write_bytes(_entry(WEBP))
        assert ccs.main(argv) == 0
        assert sent == [PNG, JPG, WEBP]
        assert "already sent 2" in capsys.readouterr().out

    def test_without_bloom_everything_resent(self, tmp_path, monkeypatch):
        # Falsification: the skip comes from the file, not from in-run dedup.
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        sent = self._recorder(monkeypatch)

        for _ in range(2):
            assert ccs.main(["--root", str(root)]) == 0
        assert sent == [PNG, PNG]

    def test_failed_send_not_recorded(self, tmp_path, monkeypatch):
        # Adversarial: a seed the fuzzer never got must be sent on the next run;
        # seeds sent before the failure must not be.
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG), "2_0": _entry(JPG)})
        bloom = tmp_path / "sent.pkl.gz"
        argv = ["--root", str(root), "--sent-bloom", str(bloom), "--retries", "0"]

        self._recorder(monkeypatch, fail_on=JPG)
        assert ccs.main(argv) == 1

        sent = self._recorder(monkeypatch)
        assert ccs.main(argv) == 0
        assert sent == [JPG]

    def test_dry_run_leaves_bloom_untouched(self, tmp_path, capsys, monkeypatch):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        bloom = tmp_path / "sent.pkl.gz"

        assert ccs.main(["--root", str(root), "--sent-bloom", str(bloom), "--dry-run"]) == 0
        assert not bloom.exists()

        # A real run after a dry run still sends the seed.
        sent = self._recorder(monkeypatch)
        assert ccs.main(["--root", str(root), "--sent-bloom", str(bloom)]) == 0
        assert sent == [PNG]

    def test_corrupt_bloom_fails_and_is_kept(self, tmp_path, capsys, monkeypatch):
        # Adversarial: a damaged file must not be silently replaced by an empty
        # filter (that would resend everything and lose the history).
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        bloom = tmp_path / "sent.pkl.gz"
        bloom.write_bytes(b"not a gzip pickle")
        sent = self._recorder(monkeypatch)

        assert ccs.main(["--root", str(root), "--sent-bloom", str(bloom)]) == 1
        assert sent == []
        assert bloom.read_bytes() == b"not a gzip pickle"
        assert "sent-bloom" in capsys.readouterr().err


class TestMaxSize:
    def test_larger_bodies_skipped_and_counted(self, tmp_path, capsys, monkeypatch):
        root = tmp_path / "chromium"
        small, big = PNG, JPG + b"J" * 100
        _profile(root, "Default", {"1_0": _entry(small), "2_0": _entry(big)})
        sent = []
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)

        assert ccs.main(["--root", str(root), "--max-size", str(len(small))]) == 0
        assert sent == [small]
        assert "too big 1" in capsys.readouterr().out

    def test_full_size_judged_not_truncated_size(self, tmp_path, monkeypatch):
        # Adversarial: --max-bytes cuts to 8 B, but the 124 B body is still too big.
        root = tmp_path / "chromium"
        big = JPG + b"J" * 100
        _profile(root, "Default", {"1_0": _entry(big)})
        sent = []
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)

        assert ccs.main(["--root", str(root), "--max-bytes", "8", "--max-size", "50"]) == 0
        assert sent == []

    def test_body_past_read_window_sized_by_trailer(self, tmp_path):
        # The trailer sits beyond max_bytes: the full size still comes from it.
        p = tmp_path / "a_0"
        body = JPG + b"J" * 5000
        p.write_bytes(_entry(body))
        assert ccs.carve_full(p, 16) == (ccs.Media.JPG, body[:16], len(body))

    def test_at_limit_is_sent(self, tmp_path, monkeypatch):
        # Falsification: exactly max-size is not "larger".
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        sent = []
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)
        assert ccs.main(["--root", str(root), "--max-size", str(len(PNG))]) == 0
        assert sent == [PNG]

    @pytest.mark.parametrize("value", ["0", "-5", "x"])
    def test_bad_values_rejected(self, tmp_path, value):
        with pytest.raises(SystemExit):
            ccs.main(["--root", str(tmp_path), "--max-size", value])


class TestSentBloomMerge:
    """Saving merges into the file on disk instead of overwriting it."""

    @staticmethod
    def _log(path):
        return ccs.SentLog(path, ccs.Persist.WRITE)

    def test_concurrent_writers_keep_both_records(self, tmp_path):
        # Two feeders on one file: the second save must not erase the first's seeds.
        path = tmp_path / "sent.pkl.gz"
        a, b = self._log(path), self._log(path)
        a.record(PNG)
        b.record(JPG)
        a.save()
        b.save()

        merged = self._log(path)
        assert merged.seen(PNG) and merged.seen(JPG)

    def test_no_change_leaves_file_untouched(self, tmp_path, capsys):
        # Same seeds already on disk: the file is not rewritten (same inode, mtime).
        path = tmp_path / "sent.pkl.gz"
        a, b = self._log(path), self._log(path)
        a.record(PNG)
        b.record(PNG)
        a.save()
        before = path.stat()

        b.save()
        after = path.stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert "unchanged" in capsys.readouterr().out

    def test_new_records_counted_on_top_of_disk(self, tmp_path):
        path = tmp_path / "sent.pkl.gz"
        a, b = self._log(path), self._log(path)
        a.record(PNG)
        b.record(JPG)
        b.record(WEBP)
        a.save()
        b.save()
        assert ccs.negative_bloom.load(path).n_added == 3

    def test_file_corrupted_meanwhile_kept(self, tmp_path, capsys):
        # Adversarial: the file went bad after load; don't replace it blindly.
        path = tmp_path / "sent.pkl.gz"
        log = self._log(path)
        log.record(PNG)
        path.write_bytes(b"garbage")

        log.save()
        assert path.read_bytes() == b"garbage"
        assert "--sent-bloom" in capsys.readouterr().err

    def test_layout_mismatch_kept(self, tmp_path, capsys):
        # Adversarial: a file from another capacity cannot be ORed; keep it.
        path = tmp_path / "sent.pkl.gz"
        log = self._log(path)
        log.record(PNG)
        other = ccs.BloomFilter(capacity=10, error_rate=0.01)
        other.update_bytes(JPG)
        ccs.negative_bloom.save(path, other)
        before = path.read_bytes()

        log.save()
        assert path.read_bytes() == before
        assert "--sent-bloom" in capsys.readouterr().err
