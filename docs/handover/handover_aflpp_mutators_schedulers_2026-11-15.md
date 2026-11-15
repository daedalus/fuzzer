# AFL++ mutators and schedulers: what is left to port — 2026-11-15

Source: AFL++ `d798677` (5.04a dev, `~/fuzzing/AFLplusplus`): `include/afl-mutations.h`,
`src/afl-fuzz-{one,queue,redqueen,mopt-adaptive,bitmap}.c`, `src/afl-main.c`, `custom_mutators/`.
Ours: `core/operator_registry.py` (305 ops, 10 bands), `core/schedulers/` (60 op, 27 seed,
24 position, 3 target arms), `core/schedules.py`, `core/rq_encodings.py`.
Method: list every AFL++ op/schedule/transform, grep ours with synonyms, open the code on
ambiguous hits. Evidence is grep + read, not measurement. Follows
`handover_aflpp_port_candidates_2026-11-15.md` (engine-level audit).

## Parity: already covered

| AFL++ | ours |
|---|---|
| havoc `MUT_FLIPBIT`, `INTERESTING8/16/32(BE)`, `ARITH8..32(BE)`, `RAND8`, `BYTEADD/SUB`, `FLIP8` | `bit_flip`, `interesting_*`, `arithmetic`, `random_bytes`, `byte_flip`, `endianness_swap` |
| `CLONE_COPY/FIXED`, `OVERWRITE_COPY/FIXED` | `block_duplicate`, `clone_fixed`, `overwrite_copy`, `overwrite_fixed` |
| `SWITCH`, `DEL`, `SHUFFLE`, `DELONE`, `INSERTONE` | `swap_regions`, `block_delete`, `byte_shuffle`, `byte_delete`, `byte_insert` |
| `ASCIINUM`, `INSERTASCIINUM` | `ascii_num_*`, `insert_ascii_num` |
| `(AUTO_)EXTRA_OVERWRITE/INSERT` | `dict_*`, `auto_extras` |
| `SPLICE_OVERWRITE/INSERT` | `splice`, `insert_range_from_other` |
| schedules fast, coe, lin, quad, rare; mmopt | `--schedule`; `mopt` = `_mopt_factor(max_depth, depth)` (MMOPT) |
| `calculate_score` speed × bitmap × handicap × depth | `schedules.py` base energy |
| MOpt PSO | `op_mopt.py` |
| redqueen encodings: plain, sext/zext, ASCII, base64, hex, case, XOR key, routines | `rq_encodings.py` (more: CRC, FNV, byte-hash preimages, mixers, LEB128, UTF-16, line codes) |
| ±1 for SUB/LEA compares | `generate_mutations(hammer=True)` |
| skip-det / effector | `skipdet_probe`, `--pos-effector` |
| autotokens, grammar tree (Superion/Nautilus-style), protobuf, libFuzzer `CrossOver`/`InsertRepeatedBytes`/`ChangeASCIIInteger`, honggfuzz `GradientCmp` | `--autotokens`, `core/grammar.py` + `tree_mutator.py`, `mutations/protobuf.py`, `crossover`, `insert_repeated_bytes`, `ascii_num_arithmetic`, `gradient_cmp` |
| libFuzzer `PersAutoDict` (reuse tokens that paid) | Dirichlet dict picker reward (`fuzz_round._feed_models`) |
| alias-table seed draw | CDF bisect over cached weights (`_cdf_pick`): O(log n) vs O(1), not worth it |

## Status (2026-11-15, same day)

Shipped: 1 input-mode context, 2 `FloatAsciiEncoder`, 3 `--rq-add-key`, 4 `--stall-mode exploit`,
6/7 `--schedule seek|exploit`, 8 Gramatron ops, 10 `c_source_mutate`. Covered already: 5 (`op_mopt`
`min_prob_frac` floor = AFL++ revival). Not ported: 9 ATNwalk (drives a Java ANTLR process over a
UNIX socket; we have no ANTLR grammars). All unmeasured; A/Bs in `docs/AB_PENDING.md` / TODO.

## Gaps, ranked (value / effort)

| # | Item | AFL++ loc | Ours | Value | Effort |
|---|---|---|---|---|---|
| 1 | Text/binary context for op bandits | `afl-fuzz-one.c:2288` `input_mode`, `mopt_ctx_index` | absent: `_context_vector` has size, entropy, edges, lineage, cmplog, format one-hot | med: AFL++ splits every op weight table on it | S |
| 2 | Float-aware I2S | `afl-fuzz-redqueen.c` `IS_FP` path in `cmp_extend_encoding` | absent: `cmpf/cmpd` logged as bit patterns; no float encoder | med on number parsers (JSON, CSV) | S |
| 3 | Additive byte key (`operand = input + k`) | `rtn_extend_encoding` `arith` counter | absent: `XorKeyEncoder` only | low–med (Caesar/ROT, `c - '0'` loops) | S |
| 4 | Explore → exploit mode switch (`-P`) | `afl-main.c:442`, `afl-fuzz-bitmap.c:1244` | absent; `--stall` goes the other way (random mode) | med: needs A/B vs stall | S |
| 5 | Adaptive MOpt (context priors + starved-op revival) | `afl-fuzz-mopt-adaptive.c` | partial: PSO MOpt + 60 bandits; no revival by slot-stealing | low–med | M |
| 6 | `seek` schedule (explore without exec time) | `afl-fuzz-queue.c` `SEEK` | absent | low: timing noise in containers | S |
| 7 | `exploit` schedule (vanilla AFL constant) | `EXPLOIT` | absent | low | S |
| 8 | Gramatron (grammar automaton walks) | `custom_mutators/gramatron` | absent (backlog A2) | med on text grammars | M |
| 9 | ATNwalk (ANTLR grammar via bit mutations) | `custom_mutators/atnwalk` | absent | low: needs ANTLR server | L |
| 10 | Compiler fuzzer (comby source rewrites) | `custom_mutators/compiler_fuzzer` | absent | low unless a compiler/interpreter target | M |

Not worth porting: `custom_send_tcp`, `guifuzz` (X11), `symqemu` (binary-only; `--enable-smt-z3` covers
the concolic need), `libafl_*` ops (mapped above), radamsa (removed upstream; ours exists).

## Sketches

**1. Text/binary context.** AFL++ classifies each input once (`input_mode` 0 text / 1 binary / 2
mixed). Add the same three-way one-hot to the shared seed features; LinUCB and C2UCB pick it up.

```python
# services/operators.py, in the shared seed-feature builder
mode = generic.input_mode(data)            # new: ascii ratio > 0.9 text, < 0.4 binary, else mixed
mode_onehot = [1.0 if mode is m else 0.0 for m in InputMode]
return [log_size, entropy_norm, ..., *fmt_onehot, *mode_onehot]
```

Wiring: `core/mutations/generic.py` (`InputMode` enum + `input_mode()`, reuse `_is_texty`),
`operators.py` feature builder and the dimension comment at `:350`, `ContextualLinUCBScheduler` /
C2UCB `d` (derive from the vector length, not a constant). Tests: text seed vs binary seed give
different one-hots (falsification: same class for both fails); empty input -> binary (adversarial).
A/B: `--contextual` before/after, fuzzgoat (text) and png (binary).

**2. Float I2S.** New `FloatEncoder(width)`: applicable for 4/8-byte CMP pairs where both operands
decode to finite floats with exponent in range. Variants: the operand, next/previous ULP, and its
ASCII renderings (`repr`, `%g`, `%.17g`) for text inputs.

```python
class FloatEncoder(Encoder):
    def encode(self, val):          # val: raw IEEE bytes of operand_b
        x = struct.unpack(self._fmt, val)[0]
        ulps = (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf))
        return [struct.pack(self._fmt, u) for u in ulps] + [repr(x).encode()]
```

Wiring: register in `_applicable_encoders` (`rq_encodings.py`); the shim already logs
`trace_cmpf/cmpd` as 4/8-byte CMP records (`afl_shim.c` `__sanitizer_cov_trace_cmpf`). Tests: a
`strtod` target compared against `3.25` is solved; NaN/inf operands are skipped (adversarial).

**3. Additive key.** `AddKeyEncoder(period)`, the XOR encoder with `(b - a) mod 256` as the key.
Wiring: next to `XorKeyEncoder`, same `_XOR_MIN_INVARIANT` gate. Test: ROT13-compared token solved.

**4. Exploit mode.** `--stall-mode {random,exploit}` (enum). `exploit`: on stall, restrict ops to
the `adaptive` and `dict` bands (redqueen, I2S, dict, colorization) instead of random mode; new
coverage switches back, as AFL++ does.

Wiring: `Fuzzer._stall_recovery_enter` / `_stall_recovery_exit`, the stall branches in
`operators.py:1459` and `:6295`; CLI in `commands.py`. A/B: random vs exploit, fuzzgoat
20 seeds × 50k (stall must trigger), A/A first.

**6/7. seek, exploit schedules.** Two branches in `PowerSchedule.compute` (`schedules.py:296`):
`seek` = base with speed factor 1; `exploit` = AFL's constant. Add to `--schedule` choices.

**8. Gramatron.** Unchanged from `docs/port-backlog.md` A2: `core/grammar.py` beside `TreeMutator`.

## Next

1, 2 and 3 are small, independent and testable without an A/B; 4 needs one. Ship 1-3 first.
