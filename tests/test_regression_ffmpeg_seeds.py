"""Regression tests for extract_ffmpeg_seeds.py max_size download limit.

Ensures the --max-size option (default 4096 bytes) skips files that
exceed the limit and allows files within it, including the Content-Length
guard that avoids downloading oversized bodies in the first place.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tools.extract_ffmpeg_seeds import download


def _mock_response(data: bytes, content_length: str | None = None):
    """Build a mock urlopen response with optional Content-Length header."""
    headers = {}
    if content_length is not None:
        headers["Content-Length"] = content_length
    resp = MagicMock()
    resp.read.return_value = data
    resp.headers = headers
    resp.__enter__ = lambda self: self
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestDownloadMaxSize:
    def test_skips_file_exceeding_max_size(self, tmp_path):
        """Adversarial: a file bigger than max_size must be rejected and not saved."""
        dest = tmp_path / "seed.bin"
        big = b"x" * (4096 + 1)
        with patch("urllib.request.urlopen", return_value=_mock_response(big)):
            result = download("http://example.com/seed", str(dest), max_size=4096)
        assert result is False
        assert not dest.exists()

    def test_accepts_file_within_max_size(self, tmp_path):
        """Happy path: a file at or under max_size is downloaded."""
        dest = tmp_path / "seed.bin"
        data = b"x" * 4096
        with patch("urllib.request.urlopen", return_value=_mock_response(data)):
            result = download("http://example.com/seed", str(dest), max_size=4096)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == data

    def test_content_length_guard_skips_oversized(self, tmp_path):
        """Falsification: Content-Length must be checked before reading the body."""
        dest = tmp_path / "seed.bin"
        with patch("urllib.request.urlopen") as mock_urlopen:
            resp = _mock_response(b"x" * 100, content_length="5000")
            mock_urlopen.return_value = resp
            result = download("http://example.com/seed", str(dest), max_size=4096)
        assert result is False
        mock_urlopen.return_value.read.assert_not_called()
        assert not dest.exists()

    def test_max_size_zero_is_unlimited(self, tmp_path):
        """Edge case: max_size=0 disables the size limit entirely."""
        dest = tmp_path / "seed.bin"
        data = b"x" * (4096 * 10)
        with patch("urllib.request.urlopen", return_value=_mock_response(data)):
            result = download("http://example.com/seed", str(dest), max_size=0)
        assert result is True
        assert dest.read_bytes() == data

    def test_default_max_size_is_4096(self):
        """Default max_size must be 4096."""
        assert download.__defaults__ is not None
        # Defaults tuple: retries=4, backoff=1.0, max_size=4096
        defaults = download.__defaults__
        assert defaults[-1] == 4096


def test_cli_default_max_size_is_4096(monkeypatch):
    """The --max-size flag defaults to 4096 and is forwarded to download()."""
    from tools.extract_ffmpeg_seeds import main

    seen = {}

    def _fake_download(url, dest, retries=4, backoff=1.0, max_size=4096):
        seen["max_size"] = max_size
        return True

    monkeypatch.setattr("tools.extract_ffmpeg_seeds.download", _fake_download)
    monkeypatch.setattr("tools.extract_ffmpeg_seeds.load_cache", lambda: {})
    monkeypatch.setattr(
        "sys.argv",
        ["extract_ffmpeg_seeds.py", "--source", "oss-fuzz", "--out", "out", "--max", "1"],
    )
    rc = main()
    assert rc == 0
    assert seen.get("max_size") == 4096


def test_cli_max_size_zero_is_unlimited(monkeypatch):
    """--max-size 0 disables the limit."""
    from tools.extract_ffmpeg_seeds import main

    seen = {}

    def _fake_download(url, dest, retries=4, backoff=1.0, max_size=4096):
        seen["max_size"] = max_size
        return True

    monkeypatch.setattr("tools.extract_ffmpeg_seeds.download", _fake_download)
    monkeypatch.setattr("tools.extract_ffmpeg_seeds.load_cache", lambda: {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_ffmpeg_seeds.py",
            "--source",
            "oss-fuzz",
            "--out",
            "out",
            "--max",
            "1",
            "--max-size",
            "0",
        ],
    )
    rc = main()
    assert rc == 0
    assert seen.get("max_size") == 0
