"""Tests for tools/blockfile_seeds.py: sequential carve of concatenated media, sent over TCP."""

import struct
import sys
import time
import zlib
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import blockfile_seeds as bfs  # noqa: E402
import chrome_cache_seeds as ccs  # noqa: E402

from fuzzer_tool.adapters.tcp_seed import TcpSeedInbox  # noqa: E402

Media = ccs.Media
ALL = set(Media)
JUNK = b"\x00junk-between-entries\x00"
WAIT_S = 5.0


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"\x00" * 14))
        + _chunk(b"IEND", b"")
    )


def _seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _jpg() -> bytes:
    # EXIF thumbnail with its own SOI..EOI must not end the outer image.
    thumb = b"\xff\xd8\xff\xe0\x00\x04ab\xff\xd9"
    app1 = _seg(0xE1, b"Exif\x00\x00" + thumb)
    dqt = _seg(0xDB, bytes(65))
    sos = _seg(0xDA, bytes(6))
    # Entropy data: stuffed 0xFF00 and a restart marker are not the end.
    entropy = b"\x12\xff\x00\x34\xff\xd0\x56"
    return b"\xff\xd8" + app1 + dqt + sos + entropy + b"\xff\xd9"


def _gif() -> bytes:
    lsd = struct.pack("<HHBBB", 1, 1, 0x80, 0, 0)  # global color table, 2 entries
    gct = bytes(6)
    gce = b"\x21\xf9\x04\x00\x00\x00\x00\x00"
    image = b"\x2c" + struct.pack("<HHHHB", 0, 0, 1, 1, 0) + b"\x02" + b"\x02ab\x00"
    return b"GIF89a" + lsd + gct + gce + image + b"\x3b"


def _webp() -> bytes:
    body = b"WEBP" + b"VP8 " + struct.pack("<I", 6) + b"vp8dat"
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _mp4(mdat: bytes = b"frames") -> bytes:
    return (
        _box(b"ftyp", b"isom\x00\x00\x00\x00isom") + _box(b"mdat", mdat) + _box(b"moov", bytes(8))
    )


def _ebml(elem_id: bytes, payload: bytes) -> bytes:
    assert len(payload) < 0x7F
    return elem_id + bytes([0x80 | len(payload)]) + payload


EBML_HEADER = _ebml(b"\x1a\x45\xdf\xa3", _ebml(b"\x42\x82", b"webm"))
INFO = _ebml(b"\x15\x49\xa9\x66", b"info")
CLUSTER = _ebml(b"\x1f\x43\xb6\x75", b"clusterdata")
UNKNOWN_SIZE = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def _webm() -> bytes:
    return EBML_HEADER + _ebml(b"\x18\x53\x80\x67", INFO + CLUSTER)


SAMPLES = {
    Media.PNG: _png(),
    Media.JPG: _jpg(),
    Media.GIF: _gif(),
    Media.WEBP: _webp(),
    Media.MP4: _mp4(),
    Media.WEBM: _webm(),
}


def _blob(*parts: bytes) -> bytes:
    return JUNK + JUNK.join(parts) + JUNK


def _scan(data: bytes, wanted=ALL, max_bytes=ccs.DEFAULT_MAX_BYTES):
    return list(bfs.scan(data, wanted, max_bytes))


class TestScan:
    def test_every_type_carved_exactly_in_order(self):
        assert _scan(_blob(*SAMPLES.values())) == list(SAMPLES.items())

    def test_back_to_back_without_gap(self):
        # The next message starts right where the previous one ended.
        png, jpg = SAMPLES[Media.PNG], SAMPLES[Media.JPG]
        assert _scan(png + jpg + png) == [(Media.PNG, png), (Media.JPG, jpg), (Media.PNG, png)]

    def test_no_media(self):
        # Falsification: plain bytes yield nothing.
        assert _scan(JUNK * 50) == []

    def test_signature_inside_body_not_split_out(self):
        # Adversarial: a PNG inside an MP4 mdat is part of the MP4, even when
        # only PNGs are wanted (the scan still skips the whole MP4).
        png = SAMPLES[Media.PNG]
        mp4 = _mp4(mdat=png)
        assert _scan(_blob(mp4)) == [(Media.MP4, mp4)]
        assert _scan(_blob(mp4), wanted={Media.PNG}) == []

    def test_type_filter_keeps_order(self):
        data = _blob(SAMPLES[Media.GIF], SAMPLES[Media.PNG], SAMPLES[Media.GIF])
        assert _scan(data, wanted={Media.GIF}) == [(Media.GIF, SAMPLES[Media.GIF])] * 2

    def test_bad_jpeg_marker_skipped(self):
        # Adversarial: FF D8 FF followed by an invalid marker is not a JPEG.
        png = SAMPLES[Media.PNG]
        assert _scan(_blob(b"\xff\xd8\xff\x00garbage", png)) == [(Media.PNG, png)]

    def test_png_with_bad_chunk_type_skipped(self):
        bad = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"\x00\x01\x02\x03" + bytes(17)
        gif = SAMPLES[Media.GIF]
        assert _scan(_blob(bad, gif)) == [(Media.GIF, gif)]

    def test_riff_wave_is_not_webp(self):
        wav = b"RIFF" + struct.pack("<I", 8) + b"WAVEfmt "
        assert _scan(_blob(wav)) == []

    def test_truncated_at_end_of_file_sent_partial(self):
        # The last message of a file may be cut; what is there is still a seed.
        png = SAMPLES[Media.PNG]
        cut = png[: len(png) - 10]
        assert _scan(JUNK + cut) == [(Media.PNG, cut)]

    def test_max_bytes_caps_seed_and_scan_resumes_after_full_body(self):
        # The PNG past the cap is still inside the MP4: resuming at the cap
        # instead of the box end would emit it.
        mp4 = _mp4(mdat=b"M" * 200 + SAMPLES[Media.PNG])
        gif = SAMPLES[Media.GIF]
        got = _scan(_blob(mp4, gif), max_bytes=20)
        assert got == [(Media.MP4, mp4[:20]), (Media.GIF, gif[:20])]

    def test_webm_unknown_size_segment_ends_at_last_known_child(self):
        # Live-stream WebM: segment size unknown, so children are walked.
        webm = EBML_HEADER + b"\x18\x53\x80\x67" + UNKNOWN_SIZE + INFO + CLUSTER
        gif = SAMPLES[Media.GIF]
        assert _scan(_blob(webm, gif)) == [(Media.WEBM, webm), (Media.GIF, gif)]

    def test_mp4_ends_at_first_non_box(self):
        # Adversarial: junk after the last box is not swallowed.
        mp4 = SAMPLES[Media.MP4]
        assert _scan(mp4 + b"\x00\x00\x00\x10zzzz" + bytes(8)) == [(Media.MP4, mp4)]


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


class TestMain:
    def test_sends_every_message_in_file_order(self, tmp_path, capsys):
        f1 = _write(tmp_path, "data_1", _blob(SAMPLES[Media.PNG], SAMPLES[Media.JPG]))
        f2 = _write(tmp_path, "data_2", _blob(SAMPLES[Media.WEBP]))
        inbox = TcpSeedInbox(0)
        try:
            argv = [str(f1), str(f2), "--port", str(inbox.port), "--delay", "0"]
            assert bfs.main(argv) == 0

            deadline = time.monotonic() + WAIT_S
            while inbox.received < 3:
                assert time.monotonic() < deadline, "seeds never arrived"
                time.sleep(0.01)
            got = [inbox.pop() for _ in range(3)]
            assert got == [SAMPLES[Media.PNG], SAMPLES[Media.JPG], SAMPLES[Media.WEBP]]
            assert "[*] sent 3 seeds (" in capsys.readouterr().out
        finally:
            inbox.close()

    def test_directory_argument_scans_its_files(self, tmp_path, capsys):
        d = tmp_path / "dump"
        d.mkdir()
        _write(d, "a", _blob(SAMPLES[Media.GIF]))
        _write(d, "b", b"")  # empty file: nothing, no error
        assert bfs.main([str(d), "--dry-run"]) == 0
        assert "would send 1 seeds (gif: 1)" in capsys.readouterr().out

    def test_no_path_discovers_browser_caches(self, tmp_path, capsys):
        cache = tmp_path / "chromium" / "Default" / "Cache" / "Cache_Data"
        cache.mkdir(parents=True)
        _write(cache, "data_1", _blob(SAMPLES[Media.PNG], SAMPLES[Media.GIF]))
        assert bfs.main(["--root", str(tmp_path / "chromium"), "--dry-run"]) == 0
        assert "would send 2 seeds (gif: 1, png: 1)" in capsys.readouterr().out

    def test_sent_bloom_second_run_sends_nothing(self, tmp_path, capsys, monkeypatch):
        f = _write(tmp_path, "data_1", _blob(SAMPLES[Media.PNG], SAMPLES[Media.GIF]))
        sent = []
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)
        argv = [str(f), "--sent-bloom", str(tmp_path / "sent.pkl.gz")]

        assert bfs.main(argv) == 0
        assert bfs.main(argv) == 0
        assert sent == [SAMPLES[Media.PNG], SAMPLES[Media.GIF]]
        assert "already sent 2" in capsys.readouterr().out

    def test_duplicate_messages_sent_once(self, tmp_path, capsys, monkeypatch):
        png = SAMPLES[Media.PNG]
        f = _write(tmp_path, "data_1", _blob(png, png))
        sent = []
        monkeypatch.setattr(ccs, "tcp_sink", lambda *_a: sent.append)
        assert bfs.main([str(f)]) == 0
        assert sent == [png]

    def test_missing_path_fails(self, tmp_path, capsys):
        # Falsification: a typo'd path is an error, not "sent 0".
        assert bfs.main([str(tmp_path / "nope"), "--dry-run"]) == 1
        assert "nope" in capsys.readouterr().err

    def test_shared_flags_accepted(self, tmp_path):
        # Same parameters as chrome_cache_seeds.py.
        f = _write(tmp_path, "data_1", b"")
        argv = [str(f), "--dry-run", "--types", "png", "--host", "127.0.0.1", "--port", "1"]
        argv += ["--delay", "0", "--max-bytes", "10", "--retries", "0", "--retry-backoff", "0"]
        assert bfs.main(argv) == 0
