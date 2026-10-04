"""Checksum site locator + post-mutation repair (TaintScope-style).

Covers core/checksum_sites.py and its hook at the end of the mutation
round (``--checksum-sites``). The locator finds, on a valid seed, which
bytes are a checksum and which region they cover; the repair recomputes
them after a mutation so the target's integrity check passes and the
mutant reaches the code behind it.
"""

import binascii
import struct
import zlib

import numpy as np
import pytest

from fuzzer_tool.core import checksum_sites as cs
from fuzzer_tool.core.checksum_sites import (
    Algo,
    Anchor,
    Endian,
    Site,
    SiteBook,
    Span,
    checksum,
    locate,
    repair,
)
from tests.support.scripted_rng import ScriptedRng

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _crc32_be(data):
    return zlib.crc32(data).to_bytes(4, "big")


def _body(n, salt=1):
    return bytes((i * 31 + salt * 7 + (i >> 3)) & 0xFF for i in range(n))


def _png_like():
    ihdr = b"IHDR" + _body(13)
    idat = b"IDAT" + _body(40, 2)
    out = PNG_MAGIC
    for chunk in (ihdr, idat):
        out += struct.pack(">I", len(chunk) - 4) + chunk + _crc32_be(chunk)
    return out


def _naive_fletcher16(data, mod=255):
    s1 = s2 = 0
    for b in data:
        s1 = (s1 + b) % mod
        s2 = (s2 + s1) % mod
    return (s2 << 8) | s1


def _naive_crc16(data, init, poly=0x1021):
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class TestAlgorithms:
    def test_known_check_values(self):
        assert checksum(Algo.CRC32, b"123456789") == 0xCBF43926
        assert checksum(Algo.ADLER32, b"Wikipedia") == 0x11E60398
        assert checksum(Algo.FLETCHER16, b"abcde") == 0xC8F0
        assert checksum(Algo.CRC16, b"123456789") == 0x29B1
        assert checksum(Algo.CRC16_XMODEM, b"123456789") == 0x31C3
        assert checksum(Algo.SUM16, b"\xff" * 300) == (255 * 300) & 0xFFFF

    @pytest.mark.parametrize("n", [0, 1, 2, 7, 255, 256, 1000])
    def test_matches_naive_references(self, n):
        data = _body(n, 3)
        assert checksum(Algo.FLETCHER16, data) == _naive_fletcher16(data)
        assert checksum(Algo.CRC16, data) == _naive_crc16(data, 0xFFFF)
        assert checksum(Algo.CRC16_XMODEM, data) == _naive_crc16(data, 0)
        assert checksum(Algo.CRC32, data) == binascii.crc32(data)

    def test_control_the_oracle_rejects_a_wrong_reference(self):
        data = _body(300, 5)
        assert checksum(Algo.FLETCHER16, data) != _naive_fletcher16(data, mod=256)
        assert checksum(Algo.CRC16, data) != _naive_crc16(data, 0)

    def test_widths(self):
        assert cs.WIDTH[Algo.CRC32] == cs.WIDTH[Algo.ADLER32] == 4
        assert cs.WIDTH[Algo.FLETCHER16] == cs.WIDTH[Algo.SUM16] == 2


class TestLocateTrailer:
    @pytest.mark.parametrize(
        ("algo", "width"),
        [
            (Algo.CRC32, 4),
            (Algo.ADLER32, 4),
            (Algo.CRC16, 2),
            (Algo.CRC16_XMODEM, 2),
            (Algo.FLETCHER16, 2),
            (Algo.SUM16, 2),
        ],
    )
    @pytest.mark.parametrize("endian", [Endian.BIG, Endian.LITTLE])
    def test_trailer(self, algo, width, endian):
        body = _body(48, 4)
        field = checksum(algo, body).to_bytes(width, endian.value)
        sites = locate(body + field)
        want = Site(algo, endian, Span.PREFIX, Anchor.TAIL, len(body), 0)
        assert want in sites

    def test_site_width_follows_the_algo(self):
        s = Site(Algo.ADLER32, Endian.BIG, Span.PREFIX, Anchor.TAIL, 10, 0)
        assert s.width == 4


class TestLocateStructured:
    def test_header_checksum_covering_the_rest(self):
        rest = _body(60, 6)
        data = b"HD" + _crc32_be(rest) + rest
        sites = locate(data)
        assert Site(Algo.CRC32, Endian.BIG, Span.SUFFIX, Anchor.HEAD, 2, 0) in sites

    def test_png_like_chunks_both_found(self):
        data = _png_like()
        pos_ihdr = len(PNG_MAGIC) + 4 + 17
        pos_idat = len(data) - 4
        found = {(s.pos, s.start) for s in locate(data) if s.algo is Algo.CRC32}
        assert (pos_ihdr, 12) in found
        assert (pos_idat, pos_ihdr + 4 + 4) in found

    def test_mid_file_16bit_field_needs_a_hint(self):
        body = _body(120, 7)
        mid = 70
        field = checksum(Algo.CRC16, body[8:mid]).to_bytes(2, "big")
        data = body[:mid] + field + body[mid:]
        assert not [s for s in locate(data) if s.pos == mid]
        hinted = locate(data, hints=[field])
        assert Site(Algo.CRC16, Endian.BIG, Span.PREFIX, Anchor.HEAD, mid, 8) in hinted

    def test_max_sites_bounds_the_result(self):
        assert len(locate(_png_like(), max_sites=1)) == 1


class TestLocateFalsification:
    def test_random_buffers_yield_no_32bit_sites(self):
        rng = np.random.default_rng(1234)
        hits = 0
        for n in rng.integers(8, 400, size=300):
            data = rng.integers(0, 256, size=int(n), dtype=np.uint8).tobytes()
            hits += sum(s.width == 4 for s in locate(data))
        assert hits == 0

    def test_random_buffers_rarely_yield_16bit_sites(self):
        rng = np.random.default_rng(99)
        samples = 300
        flagged = 0
        for n in rng.integers(16, 400, size=samples):
            data = rng.integers(0, 256, size=int(n), dtype=np.uint8).tobytes()
            flagged += bool(locate(data))
        assert flagged / samples < 0.08

    def test_corrupted_checksum_is_not_a_site(self):
        body = _body(48, 8)
        bad = bytes([body[0] ^ 1]) + body[1:]
        data = bad + _crc32_be(body)
        assert not [s for s in locate(data) if s.pos == len(body)]

    def test_empty_region_is_never_a_site(self):
        # crc32 of nothing is 0; four zero bytes must not "verify".
        assert locate(b"\x00" * 4) == []

    def test_tiny_and_empty_inputs(self):
        assert locate(b"") == []
        assert locate(b"\x01") == []
        assert locate(b"abcd") == []


class TestLocateAdversarial:
    def test_field_beyond_the_scan_cap_is_ignored(self):
        body = _body(cs.LOCATE_MAX_LEN + 64, 9)
        assert locate(body + _crc32_be(body)) == []

    def test_all_zero_and_all_ff_buffers(self):
        assert locate(b"\x00" * 200) == []
        assert locate(b"\xff" * 200) == []

    def test_hints_of_odd_sizes_are_ignored(self):
        body = _body(40, 10)
        locate(body, hints=[b"", b"\x01", b"\x01\x02\x03", b"x" * 99])


class TestRepair:
    def _trailer(self, body):
        return Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.TAIL, len(body), 0)

    def test_tail_site_after_body_mutation(self):
        body = _body(40, 11)
        site = self._trailer(body)
        buf = bytearray(body + _crc32_be(body))
        buf[5] ^= 0xFF
        assert repair(buf, [site], parent_len=len(buf)) == 1
        assert bytes(buf[-4:]) == _crc32_be(bytes(buf[:-4]))

    def test_tail_site_follows_a_length_change(self):
        body = _body(40, 12)
        site = self._trailer(body)
        buf = bytearray(body + _crc32_be(body))
        buf[10:10] = b"inserted"
        assert repair(buf, [site], parent_len=len(body) + 4) == 1
        assert bytes(buf[-4:]) == _crc32_be(bytes(buf[:-4]))

    def test_suffix_head_site_follows_body_growth(self):
        rest = _body(50, 13)
        site = Site(Algo.CRC32, Endian.BIG, Span.SUFFIX, Anchor.HEAD, 2, 0)
        buf = bytearray(b"HD" + _crc32_be(rest) + rest + b"more")
        assert repair(buf, [site], parent_len=len(buf) - 4) == 1
        assert bytes(buf[2:6]) == _crc32_be(bytes(buf[6:]))

    def test_falsification_head_prefix_site_skipped_on_length_change(self):
        body = _body(40, 14)
        site = Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.HEAD, 20, 0)
        buf = bytearray(body[:20] + _crc32_be(body[:20]) + body[20:])
        buf[3:3] = b"zz"
        before = bytes(buf)
        assert repair(buf, [site], parent_len=len(before) - 2) == 0
        assert bytes(buf) == before

    def test_png_like_both_chunks_repaired(self):
        data = bytearray(_png_like())
        sites = locate(bytes(data))
        data[len(PNG_MAGIC) + 8] ^= 0x55
        data[-10] ^= 0xAA
        assert repair(data, sites, parent_len=len(data)) >= 2
        assert not repair_is_needed(data, sites)

    def test_idempotent(self):
        body = _body(40, 15)
        site = self._trailer(body)
        buf = bytearray(body + b"\x00" * 4)
        repair(buf, [site], parent_len=len(buf))
        once = bytes(buf)
        repair(buf, [site], parent_len=len(buf))
        assert bytes(buf) == once

    def test_little_endian_field_written_little(self):
        body = _body(30, 16)
        site = Site(Algo.CRC16, Endian.LITTLE, Span.PREFIX, Anchor.TAIL, len(body), 0)
        buf = bytearray(body + b"\x00\x00")
        repair(buf, [site], parent_len=len(buf))
        assert bytes(buf[-2:]) == checksum(Algo.CRC16, body).to_bytes(2, "little")


class TestRepairAdversarial:
    def test_empty_buffer_and_no_sites(self):
        buf = bytearray()
        assert repair(buf, [], parent_len=0) == 0
        site = Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.TAIL, 10, 0)
        assert repair(buf, [site], parent_len=0) == 0
        assert buf == bytearray()

    def test_buffer_shrunk_below_the_field(self):
        site = Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.HEAD, 50, 0)
        buf = bytearray(b"x" * 20)
        assert repair(buf, [site], parent_len=20) == 0
        assert bytes(buf) == b"x" * 20

    def test_region_below_minimum_is_skipped(self):
        site = Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.TAIL, 0, 0)
        buf = bytearray(b"abcdefgh")
        before = bytes(buf)
        # tail field of 4 leaves a 4-byte region only when MIN_REGION allows.
        n = repair(bytearray(b"ab" + b"\x00" * 4), [site], parent_len=6)
        assert n == 0
        assert bytes(buf) == before

    def test_start_beyond_the_buffer(self):
        site = Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.TAIL, 0, 900)
        buf = bytearray(b"x" * 30)
        assert repair(buf, [site], parent_len=30) == 0

    def test_never_changes_the_buffer_length(self):
        body = _body(40, 17)
        buf = bytearray(body + b"\x00" * 4)
        sites = [Site(Algo.CRC32, Endian.BIG, Span.PREFIX, Anchor.TAIL, 40, 0)]
        repair(buf, sites, parent_len=len(buf))
        assert len(buf) == 44


def repair_is_needed(data, sites):
    """True when re-running repair on *data* would change anything."""
    probe = bytearray(data)
    repair(probe, sites, parent_len=len(data))
    return bytes(probe) != bytes(data)


class TestSiteBook:
    def _seed(self):
        body = _body(48, 18)
        return body + _crc32_be(body)

    def test_caches_per_parent(self, monkeypatch):
        calls = []
        real = cs.locate
        monkeypatch.setattr(cs, "locate", lambda d, **kw: calls.append(1) or real(d, **kw))
        book = SiteBook()
        seed = self._seed()
        book.sites_for(seed)
        book.sites_for(seed)
        assert len(calls) == 1

    def test_negative_result_is_cached_too(self, monkeypatch):
        calls = []
        real = cs.locate
        monkeypatch.setattr(cs, "locate", lambda d, **kw: calls.append(1) or real(d, **kw))
        book = SiteBook()
        book.sites_for(b"\x00" * 40)
        book.sites_for(b"\x00" * 40)
        assert len(calls) == 1

    def test_seed_count_is_bounded(self):
        book = SiteBook()
        for i in range(cs.MAX_SEEDS + 9):
            book.sites_for(i.to_bytes(4, "big") * 4)
        assert book.seeds_tracked() == cs.MAX_SEEDS

    def test_apply_repairs_the_mutant(self):
        book = SiteBook()
        seed = self._seed()
        buf = bytearray(seed)
        buf[7] ^= 0x10
        rng = ScriptedRng(randoms=[0.0])
        assert book.apply(seed, buf, rng) == 1
        assert bytes(buf[-4:]) == _crc32_be(bytes(buf[:-4]))

    def test_apply_skips_above_the_repair_rate(self):
        book = SiteBook()
        seed = self._seed()
        buf = bytearray(seed)
        buf[7] ^= 0x10
        before = bytes(buf)
        rng = ScriptedRng(randoms=[cs.REPAIR_P + 0.01])
        assert book.apply(seed, buf, rng) == 0
        assert bytes(buf) == before

    def test_apply_on_a_seed_without_sites_is_free(self):
        book = SiteBook()
        buf = bytearray(b"\x00" * 40)
        assert book.apply(bytes(buf), buf, ScriptedRng(randoms=[])) == 0

    def test_counters(self):
        book = SiteBook()
        seed = self._seed()
        buf = bytearray(seed)
        buf[1] ^= 1
        book.apply(seed, buf, ScriptedRng(randoms=[0.0]))
        assert book.repairs == 1
        assert book.located >= 1


class TestEngineHook:
    """The mutation round ends with the repair (``--checksum-sites``)."""

    @staticmethod
    def _fuzzer(tmp_path, seed, on):
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus = tmp_path / "corpus"
        (corpus / "seeds").mkdir(parents=True)
        (tmp_path / "crashes").mkdir()
        (corpus / "seeds" / "seed1").write_bytes(seed)
        return Fuzzer(
            "/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            checksum_sites=on,
        )

    @staticmethod
    def _flip_first(f):
        eng = f._operators

        def apply(op, buf, data, track_effect):
            buf[0] = 0xFF
            return buf

        eng._apply_op_once = apply

    @staticmethod
    def _valid(mutant):
        return mutant[-4:] == _crc32_be(mutant[:-4])

    def test_flag_on_every_mutant_keeps_a_valid_checksum(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "REPAIR_P", 1.0)
        body = _body(64, 19)
        seed = body + _crc32_be(body)
        f = self._fuzzer(tmp_path, seed, on=True)
        self._flip_first(f)
        mutants = [f._operators.mutate(seed) for _ in range(5)]
        assert all(m != seed and self._valid(m) for m in mutants)

    def test_falsification_flag_off_mutants_break_the_checksum(self, tmp_path):
        body = _body(64, 20)
        seed = body + _crc32_be(body)
        f = self._fuzzer(tmp_path, seed, on=False)
        self._flip_first(f)
        mutants = [f._operators.mutate(seed) for _ in range(5)]
        assert not any(self._valid(m) for m in mutants)
        assert f._cksum_sites is None

    def test_adversarial_seed_without_checksum_is_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "REPAIR_P", 1.0)
        seed = b"\x00" * 64
        f = self._fuzzer(tmp_path, seed, on=True)
        self._flip_first(f)
        mutant = f._operators.mutate(seed)
        assert mutant == b"\xff" + b"\x00" * 63
