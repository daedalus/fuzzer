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

    def test_refused_connection_fails(self, tmp_path, capsys):
        root = tmp_path / "chromium"
        _profile(root, "Default", {"1_0": _entry(PNG)})
        inbox = TcpSeedInbox(0)
        port = inbox.port
        inbox.close()  # nothing listens on this port any more
        assert ccs.main(["--root", str(root), "--port", str(port), "--delay", "0"]) == 1
