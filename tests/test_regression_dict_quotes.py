"""Regression test: parse_dict_line strips the AFL enclosing quotes.

The AFL dictionary format puts the token in double quotes, optionally after a
``name=`` prefix. The quotes are delimiters. They were being kept, so every
token from a standard dictionary carried a spurious 0x22 on each end --
``b'"IDAT"'`` rather than ``b'IDAT'`` -- and matched nothing in the target.
12,169 of the 18,311 tokens in ``dictionaries/`` (66.5%) were affected.

This survived because the pre-existing tests only asserted the result was
non-None ``bytes``, never what the bytes WERE. Every assertion here checks a
value.

Three further defects fixed alongside it:

* splitting on the first ``=`` mangled tokens containing one (``"a=b"`` ->
  ``b'b"'``), and destroyed the bare unquoted tokens in ruby.dict/rar.dict
  (``!=`` -> ``b''``, ``==`` -> ``b'='``);
* a regex sweep for ``\\xNN`` matched inside an escaped backslash, so
  ``\\\\x41`` decoded as backslash + ``A`` instead of backslash + ``x41``;
* ``\\\\`` and ``\\"`` were not decoded at all.
"""

from __future__ import annotations

import glob
import os

from fuzzer_tool.core.mutations.generic import load_dictionary, parse_dict_line

_DICT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dictionaries"
)


class TestEnclosingQuotes:
    def test_bare_quoted_token(self):
        assert parse_dict_line('"IDAT"') == b"IDAT"

    def test_named_quoted_token(self):
        assert parse_dict_line('keyword_if="if"') == b"if"

    def test_name_with_level_suffix(self):
        """AFL allows name@level= to weight a token."""
        assert parse_dict_line('tag@1="x"') == b"x"

    def test_empty_token_is_dropped(self):
        assert parse_dict_line('NAME=""') is None

    def test_quotes_inside_are_content_not_delimiters(self):
        r"""jsonschema.dict tokens genuinely are quoted JSON keys."""
        assert parse_dict_line(r'"\"$schema\""') == b'"$schema"'


class TestTokensContainingEquals:
    def test_quoted_token_with_equals(self):
        assert parse_dict_line('"a=b"') == b"a=b"

    def test_bare_operator_tokens(self):
        """ruby.dict is a bare token list; these are Ruby operators."""
        assert parse_dict_line("!=") == b"!="
        assert parse_dict_line("==") == b"=="
        assert parse_dict_line("<=>") == b"<=>"

    def test_bare_setter_method_name(self):
        assert parse_dict_line("DEBUG=") == b"DEBUG="


class TestEscapes:
    def test_hex_escape(self):
        assert parse_dict_line(r'"\x89PNG"') == b"\x89PNG"

    def test_unquoted_hex_escape_still_works(self):
        assert parse_dict_line(r"\x00\xff") == b"\x00\xff"

    def test_escaped_backslash_is_not_a_hex_escape(self):
        r"""\\x41 is backslash + "x41", not backslash + "A"."""
        assert parse_dict_line(r'"\\x41"') == b"\\x41"

    def test_escaped_backslash(self):
        assert parse_dict_line(r'"a\\b"') == b"a\\b"

    def test_escaped_quote(self):
        assert parse_dict_line(r'"a\"b"') == b'a"b'

    def test_unknown_escape_keeps_its_backslash(self):
        r"""ass.dict carries \1a, \2c ... which are not escapes at all."""
        assert parse_dict_line(r'"\\1a"') == b"\\1a"

    def test_trailing_lone_backslash(self):
        assert parse_dict_line('"a\\"') == b"a\\"

    def test_short_hex_escape_is_not_consumed(self):
        r"""\x4 has only one hex digit, so it is not a hex escape."""
        assert parse_dict_line(r'"\x4"') == b"\\x4"


class TestUnchangedBehaviour:
    def test_blank_line(self):
        assert parse_dict_line("") is None
        assert parse_dict_line("   ") is None

    def test_comment(self):
        assert parse_dict_line("# comment") is None

    def test_utf8_is_encoded_raw(self):
        assert parse_dict_line('"café"') == "café".encode()


class TestShippedDictionaries:
    def test_png_magic_is_the_real_magic(self):
        d = load_dictionary(os.path.join(_DICT_DIR, "png.dict"))
        assert b"\x89PNG\r\n\x1a\n" in d
        assert b"IHDR" in d
        assert b"IDAT" in d

    def test_no_token_is_spuriously_quote_wrapped(self):
        """For a SIMPLE line -- quoted, no backslashes inside -- the quotes are
        pure delimiters and must not survive into the token.

        Lines with escapes are excluded because their quotes can be real
        content: hoextdown.dict line 32 is ``string_empty_dblquotes="\\"\\""``
        and b'""' is precisely the token it means to contribute.
        """
        for path in sorted(glob.glob(os.path.join(_DICT_DIR, "*.dict"))):
            with open(path, errors="replace") as fh:
                lines = fh.readlines()
            for raw in lines:
                s = raw.strip()
                if not s or s.startswith("#") or "\\" in s:
                    continue
                if not (s.startswith('"') and s.endswith('"') and len(s) > 2):
                    continue
                tok = parse_dict_line(raw)
                assert tok == s[1:-1].encode(), f"{os.path.basename(path)}: {s!r} -> {tok!r}"

    def test_bare_token_dictionaries_survive(self):
        d = load_dictionary(os.path.join(_DICT_DIR, "ruby.dict"))
        for tok in (b"!=", b"==", b"<=>", b"[]="):
            assert tok in d, f"{tok!r} lost from ruby.dict"

    def test_no_empty_tokens_anywhere(self):
        for path in sorted(glob.glob(os.path.join(_DICT_DIR, "*.dict"))):
            assert all(load_dictionary(path)), f"{path} yielded an empty token"
