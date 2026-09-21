"""Auto-populated dictionary merge from the target profile.

The fuzzer builds its startup dictionary out of four channels extracted from
the target: interesting strings, magic bytes, disassembly constants, literal
word constants from .rodata/.data, and Bison/Yacc parser token tables. This
test pins the channel order, the dedup-against-existing rule, and the
skip-short-constants rule through a bound `_merge_profile_dictionary` so the
logic is exercised without spinning up a full Fuzzer.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.services.fuzzer import Fuzzer


def _merge(profile, dictionary):
    fake = SimpleNamespace(_profile=profile, dictionary=dictionary)
    Fuzzer._merge_profile_dictionary(fake)
    return fake.dictionary


def _profile(**overrides):
    base = dict(
        interesting_strings=[],
        magic_bytes=[],
        extracted_constants=[],
        rodata_word_constants=[],
        parser_tokens=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_channels_merged_in_order():
    out = _merge(
        _profile(
            interesting_strings=["alpha", "beta"],
            magic_bytes=[b"\x89PNG\r\n\x1a\n"],
            extracted_constants=[b"\x0d\x0a\x1a\x0a", b"\x00\x00\x00\x00\x88\x77\x66\x55"],
            rodata_word_constants=[b"\x0d\x0a\x1a\x0a", b"\x88\x77\x66\x55"],
            parser_tokens=[b"\x89PNG\r\n\x1a\n", b"\xaa\xbb"],
        ),
        [],
    )
    assert out == [
        b"alpha",
        b"beta",
        b"\x89PNG\r\n\x1a\n",
        b"\x0d\x0a\x1a\x0a",
        b"\x00\x00\x00\x00\x88\x77\x66\x55",
        b"\x88\x77\x66\x55",
        b"\xaa\xbb",
    ]


def test_existing_entries_are_not_duplicated():
    existing = [b"keep", b"\x0d\x0a\x1a\x0a"]
    out = _merge(
        _profile(
            interesting_strings=["keep", "new"],
            magic_bytes=[b"\x0d\x0a\x1a\x0a"],
            extracted_constants=[b"\x0d\x0a\x1a\x0a", b"\x77\x66\x55\x44"],
            rodata_word_constants=[b"\x0d\x0a\x1a\x0a", b"\x11\x22\x33\x44"],
            parser_tokens=[b"new", b"\x0d\x0a\x1a\x0a"],
        ),
        existing,
    )
    assert out == [b"keep", b"\x0d\x0a\x1a\x0a", b"new", b"\x77\x66\x55\x44", b"\x11\x22\x33\x44"]
    assert out.count(b"\x0d\x0a\x1a\x0a") == 1


def test_short_binary_constants_skipped():
    """Strings and parser tokens admit single bytes; the extracted-constant
    channels must not seed the dictionary with sub-2-byte noise."""
    out = _merge(
        _profile(
            extracted_constants=[b"\x41", b"\x42\x43"],
            rodata_word_constants=[b"\x41", b"\x44\x45"],
        ),
        [],
    )
    assert out == [b"\x42\x43", b"\x44\x45"]


def test_interesting_strings_capped_at_200():
    profile = _profile(interesting_strings=[f"s{i}" for i in range(250)])
    out = _merge(profile, [])
    assert len(out) == 200


def test_empty_profile_merges_nothing():
    assert _merge(_profile(), []) == []
