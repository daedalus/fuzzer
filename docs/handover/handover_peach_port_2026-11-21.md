# Handover: Peach Fuzzer (community) port analysis, 2026-11-21

Status: **analysis only, nothing implemented, nothing measured.** Base: `master` @ `51043f37`.

Source: https://peachtech.gitlab.io/peach-fuzzer-community/WhatIsPeach.html and the
Peach 3 pages it links (DataModeling, StateModel, AgentsMonitors, Fixup, Transformer, minset).
The landing page is history only. The sub-pages are element lists with no algorithms, so
every item below is a **concept-level** port, not a code port. Peach 3 is a C# / Mono
program (MIT license); nothing here copies its code.

> Before starting any item: grep for it. Earlier surveys found most of a shortlist already
> in the tree (see `docs/port-backlog.md`).

## What Peach is

A "SmartFuzzer" doing generation and mutation fuzzing from a *Pit* file with two models:

* **DataModel**: typed element tree (Number, String, Blob, Block, Choice, Flags, Padding,
  XmlElement...) plus **Relations** (size / count / offset), **Fixups** (value computed from
  other elements, e.g. CRC32, hashes) and **Transformers** (encode/decode the parent element,
  e.g. Base64, gzip, UTF-16, so the *decoded* value is what gets fuzzed). **Placement**
  relocates an element after cracking (arrays + offset relations).
* **StateModel**: States containing Actions (`output`, `input`, `call`, `close`) that
  reference DataModels; Publishers (file, TCP...) carry the bytes.
* **Agents / Monitors**: debugger, memory, process killer, cleanup-folder, ping, pcap, ...
* **Minset**: trace each sample, then pick the fewest files covering the most code. Trace
  collection is distributable, the compute step is not.

## Already covered in this repo (verified by grep on `51043f37`)

| Peach concept | Repo equivalent |
|---|---|
| Relations (size/count/offset) | `core/mutations/field_relation.py` (relation polarity, ZIP/WebP/ftyp only), `core/structural_constraints.py`, `core/field_spec.py` (fixed-offset fields), `tlv_mutate` |
| Fixups: CRC / LRC / Adler / sums | `core/checksum_sites.py` (`Algo`: CRC32, ADLER32, CRC16, CRC16_XMODEM, FLETCHER16, SUM16, XOR8 (LRC), SUM8, NEG8, CRC32C; `locate()` on a valid seed, `repair()` after mutation), `core/crc_engine.py`, `core/int_checksum_solver.py`, `--merkle-sites` |
| Generation + mutation | `--formatfuzzer` wrapper (`core/mutations/formatfuzzer.py`), ~155 hand-written format mutators, `core/grammar.py` |
| Validity-weighted scheduling (AFLSmart used Peach models) | `--validity-energy`, `--reject-code` |
| Minset | `minimize` subcommand (many modes) |
| TCP publisher | `adapters/tcp_seed.py` (inbound only, `--tcp-listen-port`) |
| StateModel-ish | `core/format_fsm.py` (StateLifter FSM), `core/mutations/seq_mutate.py`, `core/state_vars.py` |
| Memory monitor | `--rss-limit-mb` (allocation cap, not a growth-fault) |

## Gaps, ranked

### 1. Generic transformer chain operator (effort S-M, **recommended first**)

Peach decodes a field, fuzzes the clear value, re-encodes. The repo has per-format
versions (`mutations/base64.py`, `gzip.py`, `deflate_struct.py`) but no composable wrapper.

Design sketch:

* New `core/transformers.py`: encode/decode pairs, stdlib only: base64 (std/urlsafe), hex,
  gzip, zlib, bz2, URL-encode, UTF-16LE/BE. Each `decode(b) -> bytes | None` (None = not
  this encoding), `encode(b) -> bytes`.
* **Auto-detect on a valid seed** by trial decode (same locate-then-repair pattern as
  `checksum_sites.locate`): whole-input first, then delimited spans (base64 runs inside
  JSON/text, hex runs). Cache sites per seed.
* New operator `transform_wrap`: decode a detected span, run one operator drawn from the
  existing pool on the decoded bytes, re-encode, splice back. Re-encode failure = no-op
  (return the input), never raise.
* Compose with item 2: decode, mutate, repair relations, re-encode.

Why it matters: byte-level havoc cannot get past a base64/gzip layer; only the outer
framing gets mutated.

Flag: `--transform-wrap` (off by default), A/B via `tools/lib/bench_paired.py`.
Tests: round-trip per codec, detection on mixed seeds, no-op on non-decodable input,
bounded output size, deterministic under `rand_pool` RNG (no `random` module).

### 2. `--relation-sites`: length / count / offset relation repair (effort M)

Peach's central idea: mutate data while keeping size/count/offset fields consistent, or
deliberately break one relation at a time. Today that exists only for 3 formats
(`field_relation.py`) and as blind boundary writes (`tlv_mutate`).

Design sketch, extending `checksum_sites`:

* `locate()` on a valid seed: a field `F` (1/2/4 bytes, BE/LE) is a **size site** when
  `F == len(region)` for a candidate region (to EOF, to a following delimiter, fixed
  window), a **count site** when `F` equals the item count of a detected repeated
  structure, an **offset site** when `F` points at a byte offset inside the input.
* `repair()` after mutation: recompute the sites so length-checking code is passed
  (the "keep consistent" half).
* Inverse operator `relation_lie`: leave the data, set one site to a boundary value or a
  wrapping sum (reuse the `offset + size` wrap goals in `structural_constraints.py`)
  (the "break one at a time" half; same polarity idea as `field_relation`).
* Offset sites + array detection cover Peach's **Placement** use case.

Risk: false positives (a field that equals a length by coincidence). Mitigate with the
same minimum-region and entropy gates `checksum_sites` uses, and cap sites per seed
(`MAX_SITES` analogue).

Flag: `--relation-sites`. Tests: locate/repair round trip on PNG-chunk-like, TLV and
length-prefixed seeds; repair is idempotent; repair never grows or shrinks the buffer.

### 3. Pit-lite declarative data model (effort L, later)

A text DSL (no JSON per `AGENTS.md`) like `field_spec`'s `NAME@OFFSET:SIZE[be|le]` but a
tree: Block / Choice / Number / Blob / String / Flags / Padding + Relation / Fixup /
Transformer attributes, with generate, crack and mutate-with-repair. Overlaps backlog A4
(010 templates) and `formatfuzzer`. Items 1 and 2 are its building blocks; do them first
and see whether a full model is still needed.

### 4. Small items

* **Fixups not in `checksum_sites.Algo`:** RFC 1071 internet checksum (Peach
  Ethernet/ICMP fixups), MD5/SHA-1/SHA-256 (Peach hash fixups, e.g. a digest field over a
  region), sequence-increment and sequence-random (counters). LRC (`XOR8`/`NEG8`) and
  Adler are already there. Adding an `Algo` member needs a width entry and a `_scan*` path;
  hashes are 16-32 bytes, so they need a new wide-field scan.
* **Minset split:** write per-sample traces to portable files, compute the cover in a
  second step, so trace collection can be distributed (Peach's two-step model).
* **Cleanup-folder monitor:** only matters for targets that litter temp files.

### Skip

* StateModel/Publisher machinery: targets here are file-reading binaries and in-process
  harnesses; `format_fsm`, `seq_mutate` and `tcp_seed` cover the useful part. Revisit only
  for a real network target.
* Windows/macOS/VM/hardware-relay monitors, Ping, Pcap, SSH: out of scope.

## Suggested order and rules to follow

1. `transform_wrap` (item 1), then `--relation-sites` (item 2). Each behind its own A/B
   flag, none with a measured delta yet.
2. Per `AGENTS.md`: TDD, run only the affected tests, ruff/lizard only on own files, no
   `json`, RNG from `rand_pool`, add new parameters at the **end** of signatures, update
   `docs/DEEP_DIVE.md` + `CHANGELOG.md` + `docs/TODO.md`, Hard Rule 57 (push ifs up, fors
   down) in hot loops, optimization bar (never slower) if anything touches a hot path.
3. Measure with `tools/lib/bench_paired.py` on a target that has a real encoding layer
   (base64/gzip inside a container) and a length-prefixed format; synthetic-only numbers
   must be labeled as such.
4. Add both items to `docs/port-backlog.md` under group A (structure-aware generation)
   when starting; the backlog says to grep first.

## Open questions

* Are there existing targets with a nested encoding layer to benchmark `transform_wrap`
  on? (ffmpeg containers carry few; JSON/base64 and gzip-in-container targets are better.)
* Should relation sites share `Site`/`locate`/`repair` with checksum sites (one scan, one
  repair pass) or live in their own module? Sharing saves a scan per seed but couples the
  two flags.

## Verification done in this session

* Cloned `master` fresh, read the repo layout, `checksum_sites` API (`Algo`, `Site`,
  `locate`, `repair`), `field_relation`, `field_spec`, `format_fsm`, `tcp_seed`, and
  `docs/port-backlog.md`.
* Fetched the six Peach pages listed above. No code was written, no tests were run.
