# der-tlv-mutators: a parse "empty" vs "failed" conflation silently corrupts round-trips

**Date:** 2026-08-11
**Context:** fuzzer-new repo; new BER/DER op mutators (`core/mutations/der.py`, commit 471b9ef).

## Problem
The new DER TLV parser's round-trip test failed on deep nesting: `b"\x30" * 50` re-serialized as `b"\x30\x00"` — a SEQUENCE whose entire interior had silently collapsed to an empty value. The parser read fine; the serializer destroyed the data.

## Rejected
- None serious — this was a test-driven catch, not a fork. The tempting "fix the serializer" patch (special-case empty constructed nodes) would have been a symptom patch; the cause was upstream in the parser's return convention.

## Approach
`_parse_children` returned `[]` for BOTH "genuinely empty value" and "interior failed to parse (junk/deep)". The serializer then treated `[]` as "no children" and re-encoded a zero-length value, dropping the opaque bytes. Fix: the parse signal is tri-state — `list | None` — with `None` meaning "keep as an opaque leaf"; each node carries a `parsed_children` flag, and serialization emits the raw value slice verbatim for leaves. Byte-minimal re-serialization (untouched subtrees keep original tag+length bytes, ancestors re-derive lengths only when content changed) then holds for BER long-form and garbage interiors alike.

## Key insight
A parse function that reuses one sentinel for "valid but empty" and "couldn't parse" forces every consumer to guess, and a serializer guessing wrong destroys data silently — worse than failing loudly, because the round-trip test only caught it on deep nesting. Empty-vs-failed is a tri-state, not a boolean.

## Verification
Round-trip tests: canonical DER, BER long-form (byte-identical), 50-deep nesting (byte-identical), bare INTEGER; 25 behavior tests in `tests/test_mutations_der.py`; full suite 4213 passed.

## Generalizes to
Parsers that feed re-serializers/repairers must keep empty ≠ failed explicit (return `None` for failure, `[]` only for valid-empty). A parse→serialize round-trip test on adversarial inputs (deep nesting, non-minimal encodings, garbage interiors) is the cheap gate that catches sentinel conflation.
