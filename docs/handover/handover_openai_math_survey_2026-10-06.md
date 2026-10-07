# Handover: survey of `openai/math` for fuzzer-relevant results (2026-10-06)

> **Status (2026-10-06): analysis only, nothing implemented.** No production code changed. The
> survey read catalogue titles and abstracts (`CONTENTS.md`), plus the Lean scope page for family
> 192. It did **not** read any manuscript PDF or check any Lean proof. Every fuzzer connection
> below is a heuristic motivation, not a derived guarantee. Next step: pick a proposal (§4),
> read the cited paper first, then build the falsifier.

## 1. Source

- Repository: <https://github.com/openai/math>, cloned at `adc7f1241` ("Initial commit",
  Tue 2026-10-06 14:58:50 -0700). Revisions will arrive as new versions, so re-check the
  commit before relying on any claim here.
- Content: 722 manuscripts in 372 result families, produced by an unreleased internal OpenAI
  model. About 4,000 problems were posed; on average roughly three hours of ChatGPT Pro thinking
  compute went into each result.
- Layout: `CONTENTS.md` (manuscript map), `overview.pdf`, `preprints/` (722 directories with PDF,
  TeX, BibTeX), `lean/` (about 122k `.lean` files; `lean/formalization.yaml`, per-family scope
  pages `lean/docs/NNN.md`, Comparator challenges), `reasoning_traces/` (10 abridged summaries).
- **Provenance caveat (from the repo's own README):** results sit at different verification
  stages, not all have Lean formalizations, and some unformalized results may contain errors.
  Treat every result below as a lead to verify, not a citation-grade fact.
- Lean status of the families discussed here, from the *existence of a scope page* only
  (scope not compared against the paper claim, proofs not rebuilt): 127, 128, 131, 132, 192,
  235, 238 have `lean/docs/NNN.md`; **284 and 103 do not**. Only 192's page was opened; it
  states the formalization covers the disproof of the Gopalan-Servedio square-root bound.

## 2. Method and its limits

1. Parsed `CONTENTS.md` into 372 family headlines plus first-paragraph descriptions.
2. Keyword filter over title + first 200 characters of the description (bandit, regret,
   good-turing, coverage, hash, boolean, finite field, sampling, mixing, sorting, sat, entropy,
   quantum, query complexity, catalytic, string, automata, ...). 121 families matched; most
   are false positives (geometry, operator algebras, PDE).
3. Hand-triaged the matches against fuzzer subsystems, then read the full catalogue entry for
   the shortlist (127, 128, 131, 132, 192, 238, 284, plus 103, 110, 113, 122, 139, 235).

Limits: absence of a family is **not** proven (title/abstract keyword scan only). In particular
no family matched bandits/regret, Good-Turing/species estimation, or coverage estimation, so
the scheduler stack (`kl_ducb`, FEWA, Good-Turing/Toulmin arms) gets nothing direct. A manual
read of the 372 headlines might still find something the keywords missed.

## 3. Shortlist: what each result says

Paraphrased from the catalogue abstracts; theorem statements must be checked in the papers.

| Family | Result (paraphrase) | Lean page |
|---|---|---|
| 192 | Disproves the Gopalan-Servedio conjecture: the sum of a Boolean function's degree-1 Fourier coefficients can exceed any constant times sqrt(degree). | yes |
| 132 | Total Boolean functions with block sensitivity at least `s(f)^alpha` for some alpha > 2, disproving the quadratic strengthening of the Sensitivity Conjecture (`s` = single-bit flips that matter, `bs` = disjoint blocks of bits that matter together). | yes |
| 127 | Degree-d polynomial threshold functions on the cube have average sensitivity at most `8 d sqrt(n)` (asymptotic Gotsman-Linial). | yes |
| 238 | Thorp shuffle on `N = 2^d` cards mixes in Theta(d) = Theta(log N) rounds in total variation, worst-case start, whole permutation. Proof constants are large (1600d, 512d and 32800d appear across the three papers); the lower bound is `2d - O(1)`. | yes |
| 131 | Lazy edge-switch chain on simple graphs mixes in `O(n^8)` for every graphical degree sequence; also an exact uniform sampler with expected polynomial bit cost. | yes |
| 128 | Deterministic polynomial-time algorithm for a common superstring at most 2x optimal. The guarantee is for the paper's new algorithm, **not** classical Greedy. | yes |
| 284 | `R(f) = O(Q(f)^4)` for total Boolean functions is tight in the exponent (disproves the conjectured cubic bound). Worst-case bit queries, error 1/3. | **no** |
| 235 | Random k-SAT: finite positive limiting thresholds for every fixed k >= 3, hitting-time variance Theta(n), computable 3-SAT threshold. | yes |
| 103 | L = RL = BPL (log-space derandomization). | no |

Skimmed and judged only loosely related: 110 (randomized k-server), 113 and 115 (counting and
sampling matchings / contingency tables), 122 (trace reconstruction), 139 (log-concave sampling).

## 4. Proposals, ranked

### P1. Block-sensitivity probing for region liveness (families 132, 192, 127) — best fit

Touchpoints: `core/live_bit_mask.py` (`LiveBitMaskEstimator`: OR of `baseline ^ mutant`
coverage XORs plus a no-growth convergence detector; no consumers of the DEAD verdict beyond a
soft down-weight, `_LIVENESS_DEAD_WEIGHT`), the per-region estimator cache in
`services/operators.py` (around the "one LiveBitMaskEstimator per region index" block, line
~1339), and the observation site in `services/fuzz_round.py` (line ~889).

Idea: a region judged coverage-dead under *single-region* mutations could be live only when
changed *jointly* with another region (a field plus the checksum/length/offset that guards it).
Family 132 is the formal reason single-flip sensitivity can badly understate joint influence:
block sensitivity can grow polynomially faster than sensitivity. Proposal: before treating a
region as dead for a long time, sample pairs (or small sets) of disjoint dead regions and mutate
them together, feeding the results to the same estimator.

Caveats: these are asymptotic existence results about total Boolean functions; they give no
rate for real parsers. Pair probing is quadratic in region count, so it must be sampled and
budgeted. The existing evidence is that four real campaigns (zlib, png x2, jpeg) produced no
dead region at all, and CRC-covered formats rule out dead bytes by construction
(`tools/gen_synthetic_target.py`, `docs/sweeps/synthetic_liveness_calibration_2026-08-29.md`).

Falsifier / test plan:
1. Extend `tools/gen_synthetic_target.py` with a region pair that moves coverage only when both
   change (paired-field target). Confirm single-region probing marks both DEAD (expected) and
   paired probing flips them LIVE. If the single-region estimator already handles it, stop.
2. Real-target check (png, gzip, sqlite): count DEAD regions that flip LIVE under paired
   probing. If none across the corpus, drop the feature and record the negative result.
3. Cost: paired probes per DEAD region per round; compare edges per exec against baseline with
   `tools/lib/bench_paired.py`.

### P2. Thorp-style layered permutation operator (family 238) — low priority

Touchpoints: `core/mutations/generic.py::_swap_pair`, users in `core/mutations/riff.py` and
`core/mutations/arm.py`, `services/operators.py` swap ops (`_op_swap_regions`,
`_op_swap_bytes`, `_op_weizz_chunk_swap`), `core/token_shuffle.py`.

Only the *scaling law* (Theta(log N) layered rounds for `N = 2^d` items) is usable; the proven
constants are orders of magnitude above anything a mutator would run, and the lower bound is
`2d - O(1)`. It also requires power-of-two sizes. More fundamentally, a mutator usually wants a
*local* move from a good seed, not uniform mixing, so a full-shuffle operator may simply be a
bad havoc step. Worth building only as an opt-in "deep reorder" op for chunk lists, gated by a
paired A/B on a chunk-permutation target (the earlier `_swap_pair` validity-cliff study is the
prior art). Expect a possible null result.

### P3. Dictionary packing via shortest common superstring (family 128) — speculative

Touchpoints: `dictionaries/`, `core/auto_dict.py` (tokens of 3 to 32 bytes).

First, a cheap diagnostic before reading the paper: for each dictionary and for auto-dict
output, compare total token bytes against a plain greedy-merge superstring length. If overlap
savings are small (likely for atomic tokens like `IHDR`), the proposal is dead. Even when
savings exist, packing helps only if an operator splices substrings of the packed string.
Note the paper's algorithm is new and unread: implementability and practical constants are
unknown. The related KMP/covers gap noted from the myoeis sweep is the closer, cheaper lead.

### P4. Graph-switch mutation (family 131) — speculative, no consumer found

A degree-preserving edge-switch mutator would suit inputs that encode graphs. No such target
or input format in this repo was identified (a narrow look, not an exhaustive search). Park
unless a graph-format target appears.

### Informational, no action

- **284 (R vs Q, quartic tight, no Lean page).** Relevant only as context for the earlier
  Grover/QEA analysis. The adaptive-angle prototype was dropped because the OR-mask estimator
  saturates toward N, not because of a query-complexity bound, and this result does not change
  that. It says nothing about structured search over program inputs.
- **235 (random SAT thresholds), 103 (L = BPL), 113/115/139/122.** No concrete hook found.
  z3 path-constraint solving is not random k-SAT; Cook-Mertz (`cook_mertz.py`) stays unwired for
  its documented performance reasons.

## 5. Conventions to follow when implementing

`AGENTS.md` hard rules that apply: surgical changes (0); follow the closest existing example
(1); clang only (4); register operators only in `core/operator_registry.py::REGISTRY` (12);
`RandPool` for all randomness (16); no artifacts or corpora inside the source tree (17, 18);
wire new functionality where needed (19); vectorize after verifying and keep the fastest (14);
update `docs/DEEP_DIVE.md` for any shipped feature (11); only run the full pytest suite if code
changed (13); commit and push when done (10).

## 6. Reproduction

```bash
git clone https://github.com/openai/math && cd math
git checkout adc7f1241
python3 - <<'EOF'
import re
txt = open('CONTENTS.md', encoding='utf-8').read()
fams = re.findall(r'\*\*(\d{3})\.\s*(.*?)\*\*\s*(.*)', txt)
print(len(fams))  # 372
kw = re.compile(r'bandit|good.?turing|coverage|boolean|finite field|hash|sampling|mixing|'
                r'sensitivity|entropy|quantum|query complexity|string|automata|regret', re.I)
for n, t, d in fams:
    if kw.search(t + ' ' + d[:200]):
        print(n, t[:120])
EOF
```

Print through an ASCII-safe encoder or a UTF-8 terminal: the catalogue uses non-ASCII names
(Kahler, Erdos, ...), and an earlier unfiltered run failed on terminal encoding.

## 7. Open questions

1. Do the P1 premises hold on a real parser, i.e. does any region's DEAD verdict flip under
   joint mutation? This is the whole value of P1.
2. Are there families outside the keyword net that bear on bit-level influence, linear
   recurrences over GF(2) (see `core/gf2_common.py`), or hashing? A full read of the 372
   headlines is the only way to say.
3. Is family 128's algorithm simple enough to implement and does it beat greedy on real
   dictionaries? Unknown until the paper is read.
