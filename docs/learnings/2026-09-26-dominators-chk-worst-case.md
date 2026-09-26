# Dominators: CHK worst case vs Semi-NCA (Georgiadis 2005)

Source: L. Georgiadis, *Linear-Time Algorithms for Dominators and Related
Problems*, PhD thesis, Princeton 2005. Ch. 2 (practical algorithms) applies
here; Ch. 3 (linear-time pointer machine), Ch. 4 (verification), Ch. 5
(independent spanning trees) do not change anything in this repo.

## Thesis claims relevant to `core/dominators.py`

- CHK iterative (IDFS, what we ship) is simplest but not robust: Θ(k) passes
  on a back-arc chain, worst case Θ(k⁴) (`itworst`), Θ(k²) on sparse
  irreducible `idfsquad` (out-degree ≤ 2, i.e. CFG-shaped).
- Semi-NCA (SNCA, Fig. 2.8) and simple Lengauer-Tarjan (SLT) were the most
  consistently fast; SLT preferred where guarantees matter.
- §2.4 notes worst-case inputs as a DoS vector against compilers. Here the
  CFG comes from the target binary.

## Measured (Python 3.11, this repo's `compute_idom` vs a 60-line SNCA prototype)

Both agree with a brute-force oracle (CHK 0/400, SNCA 0/1000 random graphs)
and with each other on every family below. Best of 3.

| family     | n    | m     | CHK        | SNCA    |
|------------|------|-------|------------|---------|
| idfsquad   | 1201 | 2000  | 1.25 s     | 2.5 ms  |
| idfsquad   | 4096 | 6825  | **40.5 s** | 19 ms   |
| itworst    | 401  | 10500 | 6.5 s      | 2.7 ms  |
| itworst    | 801  | 41000 | 115 s      | —       |
| sncaworst  | 4095 | 6141  | 342 ms     | 72 ms   |
| CFG-like   | 4096 | 6157  | 16 ms      | 4.8 ms  |

`_MAX_CFG_BLOCKS = 4096` (`analyzer_distance.py`) does not bound this: a
4096-block irreducible function stalls `gate_blocks` ~40 s per target
function. Only reached with `--gate-bonus > 0`.

## Other findings

- `dominates()` is O(depth) with a per-call `set`; pre/post numbering of the
  dominator tree gives O(1). No hot caller today.
- `core/dominators.py` and `core/mincut.py` docstrings justify CHK with
  "revisit only if profiling shows this hot" — the risk is adversarial
  input, not average profile.
- Unused idea from Ch. 1 (circuit testing / coverage): a covered block
  implies its dominators are covered, so `ptrace_coverage.py` could place
  int3 only on dominator-tree leaves. Not measured.

## Recommendation

Replace CHK in `compute_idom` with SNCA (same signature, same unreachable
semantics). Faster on every family measured, linear-ish on the ones that
stall CHK.
