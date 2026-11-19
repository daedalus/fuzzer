"""Regression: `\\x` escapes accepted non-hex-digit text.

Found by tools/fuzz.py (`grammar_escapes`): the decoder used
`int(text[i+2:i+4], 16)`, which accepts a sign, whitespace and Unicode
digits, so `\\x+1`, `\\x 1`, `\\x1 ` decoded to 0x01 and `\\x٣٣` to `3`.
Only two ASCII hex digits form an escape; anything else keeps its backslash.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.grammar import decode_quoted_literal

ARABIC_THREE = "٣"


@pytest.mark.parametrize(
    "text",
    [r"\x+1", r"\x 1", r"\x1 ", r"\x-1", "\\x" + ARABIC_THREE * 2, r"\x_1"],
)
def test_regression_non_hex_digits_keep_backslash(text):
    assert decode_quoted_literal(text) == text.encode("utf-8")


@pytest.mark.parametrize("text, want", [(r"\x41", b"A"), (r"\xff", b"\xff"), (r"\xFF", b"\xff")])
def test_falsify_ascii_hex_still_decodes(text, want):
    assert decode_quoted_literal(text) == want


def test_adversarial_escape_at_end_of_text():
    # `\x4` has one digit left: not an escape, kept verbatim.
    assert decode_quoted_literal("a\\x4") == b"a\\x4"
