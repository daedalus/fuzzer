"""Tests for core/mutations/text_codec: encoding_wrap, escape_mutate,
ascii_float, utf16_transcode.

Every exact-output test drives the draw sequence with ScriptedRng (Hard
Rule 39) and derives the expected bytes from the stdlib codec, not from the
module under test.
"""

import base64
import binascii
import html
import time
import urllib.parse

import pytest

from fuzzer_tool.core.mutations.text_codec import (
    CODECS,
    ESCAPE_MODES,
    ESCAPES,
    FLOAT_EDGES,
    UTF16_MODES,
    ascii_float,
    encoding_wrap,
    escape_mutate,
    utf16_transcode,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096


def _codec(name):
    return [c.__name__ for c in CODECS].index(name)


def _escape_mode(name):
    return [m.__name__ for m in ESCAPE_MODES].index(name)


def _utf16_mode(name):
    return [m.__name__ for m in UTF16_MODES].index(name)


# ── encoding_wrap ──────────────────────────────────────────────────────


class TestEncodingWrap:
    def test_b64_encode_span(self):
        rng = ScriptedRng(randints=[3], choice_idxs=[_codec("b64_encode")])
        out = encoding_wrap(b"xxABCyy", 2, rng, _MAX)
        assert out == b"xx" + base64.b64encode(b"ABC") + b"yy"

    def test_b64_decode_span(self):
        enc = base64.b64encode(b"hi!")
        rng = ScriptedRng(randints=[len(enc)], choice_idxs=[_codec("b64_decode")])
        assert encoding_wrap(b"<" + enc + b">", 1, rng, _MAX) == b"<hi!>"

    def test_hex_roundtrip_pair(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_codec("hex_encode")])
        out = encoding_wrap(b"\x00\xff", 0, rng, _MAX)
        assert out == b"\x00\xff".hex().encode()

        rng = ScriptedRng(randints=[len(out)], choice_idxs=[_codec("hex_decode")])
        assert encoding_wrap(out, 0, rng, _MAX) == b"\x00\xff"

    def test_pct_encode_encodes_every_byte(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_codec("pct_encode")])
        out = encoding_wrap(b"a/", 0, rng, _MAX)
        assert out == b"%61%2F"
        assert urllib.parse.unquote_to_bytes(out) == b"a/"

    def test_pct_double_decodes_twice(self):
        rng = ScriptedRng(randints=[1], choice_idxs=[_codec("pct_double")])
        out = encoding_wrap(b"/", 0, rng, _MAX)
        once = urllib.parse.unquote_to_bytes(out)
        assert once != b"/"
        assert urllib.parse.unquote_to_bytes(once) == b"/"

    def test_pct_decode_span(self):
        rng = ScriptedRng(randints=[6], choice_idxs=[_codec("pct_decode")])
        assert encoding_wrap(b"%41%42", 0, rng, _MAX) == b"AB"

    def test_byte_idx_wraps_into_buffer(self):
        rng = ScriptedRng(randints=[1], choice_idxs=[_codec("hex_encode")])
        assert encoding_wrap(b"ab", 3, rng, _MAX) == b"a" + b"b".hex().encode()

    # HTML5 maps NUL and C0/C1 controls to U+FFFD by spec, so its span avoids them.
    @pytest.mark.parametrize(
        ("enc", "dec", "span"),
        [
            ("b32_encode", base64.b32decode, b"\x00a=&\xff"),
            ("b85_encode", base64.b85decode, b"\x00a=&\xff"),
            ("a85_encode", lambda s: base64.a85decode(s, adobe=True), b"\x00a=&\xff"),
            ("qp_encode", binascii.a2b_qp, b"\x00a=&\xff"),
            ("html_encode", lambda s: html.unescape(s.decode()).encode("latin-1"), b"a=&<\xff"),
        ],
    )
    def test_new_encoder_roundtrips_via_stdlib(self, enc, dec, span):
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec(enc)])
        out = encoding_wrap(b"<" + span + b">", 1, rng, _MAX)
        assert out[0:1] == b"<" and out[-1:] == b">"
        assert dec(out[1:-1]) == span

    def test_qp_html_encode_every_byte(self):
        # Every byte escaped, printable ones too: the decoder path always runs.
        rng = ScriptedRng(randints=[2], choice_idxs=[_codec("qp_encode")])
        assert encoding_wrap(b"ab", 0, rng, _MAX) == b"=61=62"
        rng = ScriptedRng(randints=[2], choice_idxs=[_codec("html_encode")])
        assert encoding_wrap(b"ab", 0, rng, _MAX) == b"&#x61;&#x62;"

    @pytest.mark.parametrize(
        ("dec", "span", "want"),
        [
            ("b32_decode", base64.b32encode(b"hi!"), b"hi!"),
            ("b32_decode", base64.b32encode(b"hi!").rstrip(b"="), b"hi!"),
            ("a85_decode", base64.a85encode(b"hi!", adobe=True), b"hi!"),
            ("a85_decode", base64.a85encode(b"hi!"), b"hi!"),
            ("qp_decode", b"=41=\r\nB", b"AB"),
            ("html_decode", b"&lt;&#65;&#x42;", b"<AB"),
            ("html_decode", b"&#x1F600;", "\U0001f600".encode()),
        ],
    )
    def test_new_decoder_span(self, dec, span, want):
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec(dec)])
        assert encoding_wrap(span, 0, rng, _MAX) == want

    # Falsification: input the decoder does not recognise is left alone.
    @pytest.mark.parametrize(
        ("dec", "span"),
        [
            ("b32_decode", b"!!!!"),
            ("a85_decode", b"~~~~"),
            ("qp_decode", b"plain"),
            ("html_decode", b"a & b"),
        ],
    )
    def test_falsify_new_decoder_declines(self, dec, span):
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec(dec)])
        assert encoding_wrap(span, 0, rng, _MAX) is None

    def test_mutf8_encode_nul_and_astral(self):
        astral = "\U0001f600"
        units = astral.encode("utf-16-be")
        cesu = b"".join(
            chr(int.from_bytes(units[i : i + 2], "big")).encode("utf-8", "surrogatepass")
            for i in (0, 2)
        )
        span = b"a\x00" + astral.encode()
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec("mutf8_encode")])
        assert encoding_wrap(span, 0, rng, _MAX) == b"a\xc0\x80" + cesu

    def test_mutf8_decode_inverts_encode(self):
        span = b"x\x00\xf0\x9f\x98\x80\xc3\xa9"
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec("mutf8_encode")])
        enc = encoding_wrap(span, 0, rng, _MAX)
        rng = ScriptedRng(randints=[len(enc)], choice_idxs=[_codec("mutf8_decode")])
        assert encoding_wrap(enc, 0, rng, _MAX) == span

    # Falsification: plain UTF-8 has nothing MUTF-8 specific; invalid
    # 4-byte leads are not re-encoded as a pair.
    @pytest.mark.parametrize("codec", ["mutf8_encode", "mutf8_decode"])
    def test_falsify_mutf8_plain_declines(self, codec):
        span = "héllo".encode() + b"\xf4\x90\x80\x80"
        rng = ScriptedRng(randints=[len(span)], choice_idxs=[_codec(codec)])
        assert encoding_wrap(span, 0, rng, _MAX) is None

    # Falsification: a decoder that cannot decode must decline, not emit.
    def test_regression_undecodable_span_declines(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_codec("hex_decode")])
        assert encoding_wrap(b"zz", 0, rng, _MAX) is None

    def test_decline_when_output_exceeds_max_len(self):
        rng = ScriptedRng(randints=[4], choice_idxs=[_codec("pct_encode")])
        assert encoding_wrap(b"abcd", 0, rng, 8) is None

    def test_empty_declines(self):
        assert encoding_wrap(b"", 0, ScriptedRng(), _MAX) is None

    # Adversarial: hostile bytes, tiny budgets, every seed stays in bounds.
    @pytest.mark.parametrize("max_len", [1, 2, 8, 64])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        blobs = [
            bytes(range(256)),
            b"%" * 50,
            b"=" * 40,
            b"\xff" * 33,
            b"&#" * 30,
            b"<~" + b"z" * 40,
        ]
        for seed in range(40):
            rng = RandPool(seed=seed)
            for blob in blobs:
                out = encoding_wrap(blob[:max_len], seed, rng, max_len)
                assert out is None or len(out) <= max_len


# ── escape_mutate ──────────────────────────────────────────────────────


class TestEscapeMutate:
    def test_insert_escape(self):
        idx = ESCAPES.index(b"\\uD800")
        rng = ScriptedRng(choice_idxs=[_escape_mode("insert_escape"), idx])
        assert escape_mutate(b'"ab"', 2, rng, _MAX) == b'"a' + b"\\uD800" + b'b"'

    def test_drop_quote_leaves_string_unterminated(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("drop_quote")])
        assert escape_mutate(b'k="v"', 3, rng, _MAX) == b'k="v'

    def test_escape_quote(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("escape_quote")])
        assert escape_mutate(b"'a'", 1, rng, _MAX) == b"'a\\'"

    def test_quote_search_wraps(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("drop_quote")])
        assert escape_mutate(b'"abc', 2, rng, _MAX) == b"abc"

    def test_strip_escape(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("strip_escape")])
        assert escape_mutate(b"a\\nb", 0, rng, _MAX) == b"anb"

    def test_tail_backslash(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("tail_backslash")])
        assert escape_mutate(b"abc", 0, rng, _MAX) == b"abc\\"

    # Falsification: nothing to break means decline.
    def test_regression_no_quote_declines(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("drop_quote")])
        assert escape_mutate(b"plain", 0, rng, _MAX) is None

    def test_insert_over_budget_declines(self):
        rng = ScriptedRng(choice_idxs=[_escape_mode("tail_backslash")])
        assert escape_mutate(b"abcd", 0, rng, 4) is None

    @pytest.mark.parametrize("max_len", [1, 4, 16])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        blobs = [b'"' * 20, b"\\" * 20, bytes(range(256)), b"'\"'\"\\"]
        for seed in range(40):
            rng = RandPool(seed=seed)
            for blob in blobs:
                out = escape_mutate(blob[:max_len], seed, rng, max_len)
                assert out is None or len(out) <= max_len


# ── ascii_float ────────────────────────────────────────────────────────


class TestAsciiFloat:
    def test_replaces_covering_number(self):
        edge = FLOAT_EDGES.index(b"1e309")
        rng = ScriptedRng(choice_idxs=[edge])
        assert ascii_float(b'{"a": 12.5e3}', 8, rng, _MAX) == b'{"a": 1e309}'

    def test_takes_next_number_after_idx(self):
        edge = FLOAT_EDGES.index(b"-0.0")
        rng = ScriptedRng(choice_idxs=[edge])
        assert ascii_float(b"x=1, y=22", 3, rng, _MAX) == b"x=1, y=-0.0"

    def test_wraps_to_first_number(self):
        edge = FLOAT_EDGES.index(b"NaN")
        rng = ScriptedRng(choice_idxs=[edge])
        assert ascii_float(b"7 abc", 3, rng, _MAX) == b"NaN abc"

    def test_edges_parse_or_are_malformed_on_purpose(self):
        # Every edge is either a float Python accepts or a deliberately
        # malformed spelling; none is empty.
        assert all(FLOAT_EDGES)
        assert b"2.2250738585072011e-308" in FLOAT_EDGES

    # Falsification: no digits means no work.
    def test_regression_no_number_declines(self):
        assert ascii_float(b"abc", 0, ScriptedRng(), _MAX) is None

    def test_over_budget_declines(self):
        edge = FLOAT_EDGES.index(b"1.7976931348623157e308")
        assert ascii_float(b"1", 0, ScriptedRng(choice_idxs=[edge]), 4) is None

    def test_adversarial_long_digit_run_is_bounded(self):
        data = b"9" * 60000
        t0 = time.perf_counter()
        out = ascii_float(data, 30000, RandPool(seed=1), 65536)
        assert time.perf_counter() - t0 < 0.5
        assert out is None or len(out) <= 65536

    @pytest.mark.parametrize("max_len", [1, 2, 8])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        for seed in range(40):
            out = ascii_float(b"0.5e-1"[:max_len], seed, RandPool(seed=seed), max_len)
            assert out is None or len(out) <= max_len


# ── utf16_transcode ────────────────────────────────────────────────────


class TestUtf16Transcode:
    def test_span_le_has_bom(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_utf16_mode("span_le")])
        out = utf16_transcode(b"<ab>", 1, rng, _MAX)
        assert out == b"<" + b"\xff\xfe" + "ab".encode("utf-16-le") + b">"

    def test_span_be_has_bom(self):
        rng = ScriptedRng(randints=[1], choice_idxs=[_utf16_mode("span_be")])
        out = utf16_transcode(b"a", 0, rng, _MAX)
        assert out == b"\xfe\xff" + "a".encode("utf-16-be")

    def test_whole_le_decodes_back(self):
        rng = ScriptedRng(choice_idxs=[_utf16_mode("whole_le")])
        out = utf16_transcode("é!".encode(), 0, rng, _MAX)
        assert out.decode("utf-16-le") == "é!"

    def test_odd_trunc_is_odd(self):
        rng = ScriptedRng(choice_idxs=[_utf16_mode("odd_trunc")])
        out = utf16_transcode(b"abc", 0, rng, _MAX)
        assert len(out) % 2 == 1
        assert out == "abc".encode("utf-16-le")[:-1]

    def test_lone_surrogate(self):
        rng = ScriptedRng(choice_idxs=[_utf16_mode("lone_surrogate")])
        out = utf16_transcode(b"ab", 1, rng, _MAX)
        assert out == b"a\x00\xd8b"

    def test_invalid_utf8_survives_via_surrogates(self):
        rng = ScriptedRng(choice_idxs=[_utf16_mode("whole_le")])
        out = utf16_transcode(b"\xff", 0, rng, _MAX)
        # Undecodable byte becomes a lone low surrogate (U+DCFF).
        assert out == b"\xff\xdc"

    def test_utf32_span_le_has_bom(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_utf16_mode("utf32_span_le")])
        out = utf16_transcode(b"<ab>", 1, rng, _MAX)
        assert out == b"<" + b"\xff\xfe\x00\x00" + "ab".encode("utf-32-le") + b">"

    def test_utf32_span_be_has_bom(self):
        rng = ScriptedRng(randints=[2], choice_idxs=[_utf16_mode("utf32_span_be")])
        out = utf16_transcode("é".encode(), 0, rng, _MAX)
        assert out == b"\x00\x00\xfe\xff" + "é".encode("utf-32-be")

    def test_utf32_out_of_range(self):
        # U+110000: one past Unicode, a valid 32-bit unit a decoder must reject.
        rng = ScriptedRng(choice_idxs=[_utf16_mode("utf32_oob")])
        out = utf16_transcode(b"ab", 1, rng, _MAX)
        assert out == b"a" + (0x110000).to_bytes(4, "little") + b"b"
        with pytest.raises(UnicodeDecodeError):
            out[1:5].decode("utf-32-le")

    # Falsification: UTF-32 quadruples; over budget declines rather than truncates.
    def test_falsify_utf32_over_budget_declines(self):
        rng = ScriptedRng(randints=[3], choice_idxs=[_utf16_mode("utf32_span_le")])
        assert utf16_transcode(b"abc", 0, rng, 15) is None

    # Falsification: a transcoding that cannot fit must decline.
    def test_regression_over_budget_declines(self):
        rng = ScriptedRng(choice_idxs=[_utf16_mode("whole_le")])
        assert utf16_transcode(b"abcd", 0, rng, 6) is None

    def test_empty_declines(self):
        assert utf16_transcode(b"", 0, ScriptedRng(), _MAX) is None

    @pytest.mark.parametrize("max_len", [1, 2, 3, 8])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        blobs = [bytes(range(256)), "\U0010ffff".encode() * 4, b"\xed\xa0\x80" * 5]
        for seed in range(40):
            rng = RandPool(seed=seed)
            for blob in blobs:
                out = utf16_transcode(blob[:max_len], seed, rng, max_len)
                assert out is None or len(out) <= max_len
