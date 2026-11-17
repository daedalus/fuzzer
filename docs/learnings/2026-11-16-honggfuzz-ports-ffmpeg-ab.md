# honggfuzz ports on FFmpeg: `--stack-depth-max` and `cmp_solve` (2026-11-16)

Ledger #148, #149. Both NULL.

## Design

- Target: `ffmpeg_read_9.0.2_noasan.so`, clang 18, trace-pc-guard + trace-cmp, direct_lite, rebuilt after master's `afl_shim.c` changes.
- `--set ffmpeg`, 20 seeds × 10k execs, rep 1. Code `d03b068`.
- Threshold frozen beforehand in `docs/AB_PENDING.md` (`93f87d9`): real_edges, Wilcoxon two-sided α 0.05.
- Three invocations on 4 cores:
  - `baseline`, then `stack-depth-max-ctl`;
  - `stack-depth-max`;
  - `cmp-solve-off`, then `cmp-solve-off-ctl`, from a worktree with `cmp_solve` dropped from the registry.
- `baseline` is the `cmp_solve` on arm.
- Deviation from the frozen design: new-coverage trim became opt-in (`--trim-new-coverage`, off) between freezing and running. It applies to every arm. The trim walk had taken ~98% of FFmpeg campaign time (1,000 execs in up to 750 s).

## Results (real_edges, baseline median 20,823)

| Comparison | W/L | Wilcoxon p | med Δ | cpu_user |
|---|---|---|---|---|
| `stack-depth-max-ctl` vs `baseline` (A/A) | 11/9 | 0.87 | +246.5 | 13/7, p=0.18, +1.9 s |
| `stack-depth-max` vs `baseline` | 11/9 | 0.70 | +360.5 | 14/6, p=0.048, +4.4 s of 73 |
| `cmp-solve-off-ctl` vs `cmp-solve-off` (A/A) | 9/11 | 0.52 | −95.0 | 8/12, p=0.52, −2.0 s |
| `baseline` (cmp_solve on) vs `cmp-solve-off` | 10/10 | 0.67 | −58.0 | 7/13, p=0.19, −4.7 s |

`stack-depth-max` corpus: 14/6, p=0.11, med +46 seeds. McNemar agrees: 0.82 for both stack rows; 1.0 and 0.82 for the two cmp rows.

CPU is confounded by scheduling. The arms in the two-arm invocations ran sequentially, so they overlapped different neighbours. The A/A arm already moves +1.9 s, so the +4.4 s is not a measured cost.

## Mechanism live (seed 0 rerun, 3k execs)

- `--stack-depth-max`: 7 admissions, max depth 44,544 B. FFmpeg's demuxers rarely recurse deeper, so the signal is sparse.
- `cmp_solve`: picked 70 times out of 16,525 mutations, credited with a success 23 times. `gradient_cmp` is comparable at 84 picked, 21 credited. The op works but is a small share of the mix, too small to move campaign edges at 10k.

## Reading

Both NULL inside clean A/A spreads. Neither earns a default. Better targets for stack depth would be recursive parsers: expression grammars, nested containers. `cmp_solve` would need a magic-heavy target where the cmplog pool stays small.
