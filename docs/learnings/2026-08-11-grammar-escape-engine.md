# grammar-escape-engine: a "loaded" grammar that generates garbage is a dead asset

**Date:** 2026-08-11
**Context:** fuzzer-new repo; `dictionaries/der.gram` wired while adding BER/DER ops (commit 471b9ef).

## Problem
`load_grammar("dictionaries/der.gram")` succeeded (9 rules), but `generate("der")` produced `3f3f3f3f3f` — every literal emitted `0x3F`. Even production `png.gram`'s `signature = \x89PNG\r\n\x1a\n` generated `?`×5. The grammar engine's tokenizer never expanded `\xNN` escapes: `\x30` tokenized as a ref to an undefined rule named `x30`, and undefined refs generate `b"?"` (0x3F).

## Rejected
- **Write der.gram with quoted ASCII literals only** — quoted literals take UTF-8 text verbatim and keep backslashes literal (pinned by `test_hex_escape`), so binary DER bytes are inexpressible that way.
- **Drop der.gram from the change** — the user explicitly approved the asset; shipping it broken was not an option.
- **Fix the engine** (chosen) — the module docstring documents `SP = \x20` as intended syntax, so the parser, not the grammar files, was wrong. Hard rule 1: fix in one place for everything (png/jpeg/rar.gram included).

## Approach
In `Grammar._parse_alternative`, pre-scan each alternative outside quotes for `\xNN` / `\t` / `\r` / `\n`, splitting it into literal-byte tokens; all remaining text flows through the existing regex tokenizer unchanged. Quoted literals keep backslash text verbatim (existing `test_hex_escape` preserved). 5 engine tests + a der.gram end-to-end test added.

## Key insight
"Grammar loaded: N rules" proves the loader, not generation — an asset can load cleanly and still be dead. The der.gram bug surfaced only because I validated through the full consumer path (load → generate → parse with the format's own parser) instead of stopping at "loads". Also: when the docstring documents a syntax the parser doesn't implement, the parser is wrong — don't rewrite the docs to match the bug.

## Verification
`load_grammar(der.gram).generate("der")` now yields `30 ...` SEQUENCE bytes, some fully parseable via `parse_der`; png.gram's `signature` still emits `?` for its bare-ASCII refs (`PNG`, `IHDR` — a separate pre-existing flaw, noted not fixed). Full suite 4213 passed.

## Generalizes to
For any format asset (dictionary, grammar, template), validate through the end-to-end consumer path — load → generate → feed the format's own parser — not just the loader. And when docs and code disagree about a syntax, the code is the bug: fix it once, in the shared parser, rather than per-file workarounds.
