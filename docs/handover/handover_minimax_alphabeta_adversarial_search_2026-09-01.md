# Handover — Minimax, Alpha-Beta Pruning, and Adversarial Search Applied to Fuzzer Scheduling

**Date:** 2026-09-01
**Status:** RESEARCH SPIRE — implementation-ready investigations, not yet wired in.

---

## 0 Rule 1.

The file where all the percolation primitives live is `core/minimax.py`.

## 1. What the three algorithms are (and why they matter here)

### 1.1 Minimax estimator (statistical decision theory)

An estimator $\delta^M$ is **minimax** when it minimizes the maximum risk across all possible parameter values:

$$\sup_{\theta \in \Theta} R(\theta, \delta^M) = \inf_\delta \sup_{\theta \in \Theta} R(\theta, \delta)$$

The key insight: **optimize for the worst case, not the average case.** A minimax estimator is the Bayes estimator against the *least favorable prior* — the prior that maximizes the Bayes risk. This is robustness-by-construction.

**Relevance to fuzzing:** The fuzzer already has 14+ schedulers (Thompson sampling, CMA-ES, MCTS, GP-UCB, etc.) arbitrated by Elo. The current Elo system rewards *average* performance. A minimax lens asks: which scheduler performs best in the *worst-case* target/phase combination? This is the difference between "best on average" and "never catastrophically bad."

### 1.2 Minimax algorithm with alternate moves (combinatorial game theory)

The classic two-player game tree search:

```
function minimax(node, depth, maximizingPlayer):
    if depth == 0 or node is terminal:
        return heuristic(node)
    if maximizingPlayer:
        value = -∞
        for each child:
            value = max(value, minimax(child, depth-1, FALSE))
        return value
    else:
        value = +∞
        for each child:
            value = min(value, minimax(child, depth-1, TRUE))
        return value
```

Players alternate. The maximizer picks the move that maximizes the minimum value the opponent can force. This is the foundation of adversarial search.

**Relevance to fuzzing:** The fuzzer already models the world as a game — the mutation lineage tree in `core/schedulers/mcts.py` is a game tree where the "opponent" is the target program's coverage response. But MCTS uses UCT (a bandit-based rollout), not true minimax. The comparison-wall problem (memcmp/strcmp gates) is literally an adversarial game: the fuzzer tries to satisfy comparisons, the target "chose" hard-to-satisfy values.

### 1.3 Alpha-beta pruning

An optimization of minimax that provably returns the same result while exploring fewer nodes:

- **Alpha (α):** minimum score the maximizing player is assured of
- **Beta (β):** maximum score the minimizing player is assured of
- **Prune when β < α:** the current branch cannot influence the final decision

With optimal move ordering: $O(b^{d/2})$ instead of $O(b^d)$ — twice as deep in the same time.

**Relevance to fuzzing:** The mutation lineage tree in MCTS already has a branching factor of ~147 (operators) and depth limited to 64. Alpha-beta pruning could double effective search depth. More importantly, the *operator selection* problem has a natural move-ordering heuristic (Thompson sample = prior on best move), which is exactly what alpha-beta needs for optimal pruning.

---

## 2. Where these algorithms fit in the fuzzer architecture

### 2.1 Current state: the scheduling stack

| Layer | Current implementation | File |
|-------|----------------------|------|
| Operator selection | 14 bandit/optimizer algorithms under Elo arbitration | `core/schedulers/*.py`, `core/elo.py` |
| Seed selection | Boltzmann, Pareto, GA, QEA, Bayesian, Markov-gen, MCTS/UCT | `services/seed_picker.py`, `core/schedulers/mcts.py` |
| Meta-scheduling | Elo rating across all strategies | `core/elo.py` |
| Mutation lineage | Parent/child forest with LCA-based diversity | `core/lineage.py` |

The fuzzer already treats scheduling as an optimization problem. What it does **not** yet do is treat it as an *adversarial* optimization problem.

### 2.2 The adversarial gap

The fuzzer's current schedulers are all **single-agent** optimization:

- Bandits maximize expected reward (coverage, new edges)
- MCTS/UCT maximizes expected reward over the lineage tree
- Elo picks the best-performing strategy on average

But fuzzing is a two-player game:

- **Player 1 (fuzzer):** picks seed + operator + mutation parameters
- **Player 2 (target):** responds with coverage, crashes, comparison walls, path explosion

The target is not passive. A target with deep comparison chains is *actively defending* against the fuzzer. A target with path explosion is *diluting* the fuzzer's budget. The current schedulers assume a stationary reward distribution — but the target's "strategy" shifts as the campaign progresses (early: easy paths; late: comparison walls and rare edges).

---

## 3. Five concrete integration points

### 3.1 Minimax-robust scheduler selection (minimax estimator)

**Problem:** Elo arbitration picks the scheduler with the highest *average* reward. A scheduler that does well on most targets but catastrophically fails on one (e.g., CMA-ES on a target with rare edges) can dominate Elo while being dangerous.

**Proposal:** Maintain a **risk matrix** $R_{i,j}$ = worst-case regret of scheduler $i$ on target class $j$. The minimax scheduler choice is:

$$\arg\min_i \max_j R_{i,j}$$

This is the direct analog of the minimax estimator: minimize the maximum risk.

**Where:** `core/elo.py` — extend the Elo tracker with a worst-case regret channel alongside the mean rating. The risk matrix is populated from benchmark sweeps (`tools/bench_paired.py`).

**Precedent in codebase:** The `boltzmann-count` vs `boltzmann-cost` A/B in `docs/handover/handover_boltzmann_ab_2026-08-30.md` is exactly this kind of analysis — but it's done offline. The proposal is to make it online and structural.

**Falsifiable prediction:** On a heterogeneous target set (png, jpeg, ffmpeg, sqlite), the minimax-robust scheduler mix will have lower *variance* in edge-discovery rate than the Elo-optimal mix, at the cost of 5-15% lower *mean* edge-discovery rate. The tradeoff is worth it for campaigns that run days/weeks (variance dominates mean over long horizons).

### 3.2 Alpha-beta pruning for MCTS seed selection (alpha-beta pruning)

**Problem:** `MCTSSeedScheduler` in `core/schedulers/mcts.py` uses UCT for descent. UCT is a bandit algorithm — it explores by random rollout. But the mutation lineage tree is not a bandit problem; it's a *game tree* where the "opponent" is the target's coverage response. UCT wastes budget on random rollouts that a minimax evaluation would prune.

**Proposal:** Replace UCT descent with **alpha-beta minimax** over the lineage tree, using the existing `_squash(new_edges)` function as the leaf evaluation. The "minimizing player" is modeled by a pessimistic coverage estimate (e.g., the minimum new-edges-found among sibling nodes at each depth).

**Key insight:** The lineage tree already has the right structure. Each node is a seed; children are mutations of that seed. The maximizer (fuzzer) picks which seed to mutate next. The minimizer (target model) picks which mutation "responds" with the least new coverage. Alpha-beta prunes branches where even the best possible mutation cannot beat an already-found alternative.

**Where:** `core/schedulers/mcts.py` — add an `AlphaBetaMCTSSeedScheduler` class parallel to the existing `MCTSSeedScheduler`. Same interface, different descent algorithm.

**Expected speedup:** With Thompson-ordered move evaluation (best UCT child first), alpha-beta should achieve near-optimal $O(b^{d/2})$ pruning. On the current `max_depth=64` and effective branching factor ~50 (after eligibility filtering), this is the difference between depth-64 and effective depth-32 search — a qualitative improvement in lineage exploitation.

**Risk:** The pessimistic "target model" is a heuristic. If it's too pessimistic, the search becomes overly conservative and misses high-risk/high-reward mutations. Calibration against the existing UCT scheduler on `targets/png_read` is the first validation step.

### 3.3 Adversarial operator sequencing (minimax with alternate moves)

**Problem:** The `MonteCarloScheduler` in `core/schedulers/monte_carlo.py` already tracks pairwise transition counts (`transition_counts[prev][next]`). This is a first-order Markov model of operator sequences. But it's used only for blending with Thompson sampling — not for *planning* sequences.

**Proposal:** Model operator selection as a two-player game where:
- **Maximizer:** picks operator $o_t$ to maximize expected new edges
- **Minimizer (target model):** picks the "response" — which comparison wall or path explosion the target throws back

The minimax value of an operator sequence is:

$$V(o_1, o_2, ..., o_k) = \max_{o_1} \min_{r_1} \max_{o_2} \min_{r_2} ... Q(\text{state after } k \text{ plies})$$

where $r_i$ is the target's "response" (modeled pessimistically from historical coverage data).

**Where:** `core/schedulers/monte_carlo.py` — extend the transition matrix into a full game tree search. The existing `stationary_distribution()` and `spectral_gap()` methods already provide the Markov chain analysis; the next step is *control* rather than just *analysis*.

**Connection to existing code:** The `correlated_select()` method already uses Cholesky-decomposed covariance to select operators that "move together." This is a single-ply lookahead. Minimax extends it to multi-ply lookahead with a model of the target's adversarial response.

### 3.4 Comparison-wall solving as a minimax game (minimax algorithm)

**Problem:** Comparison walls (memcmp, strcmp, checksum checks) are the single hardest problem in coverage-guided fuzzing. The current approach uses:
- Redqueen (input-to-comparison operand matching)
- SMT solving (Z3)
- Colorization (byte-level taint approximation)

All of these are *reactive*: they observe comparisons and try to solve them. None models the problem as a *game*.

**Proposal:** Model each comparison wall as a minimax game:

- **Maximizer (fuzzer):** chooses which byte to mutate and what value to try
- **Minimizer (target):** the comparison function itself — it "chooses" to fail until the exact right byte sequence is found

The minimax value of a comparison state is the minimum number of mutations needed to satisfy the comparison. Alpha-beta pruning applies naturally: if a partial mutation already fails an early byte in a memcmp, no need to explore later bytes.

**Where:** `core/cond_stmt.py` (conditional statement solving) and `core/smt_solver.py` (Z3 integration). The existing `path_negate` operator already solves for inputs that take the opposite branch — this extends it to a full game-tree search over comparison sequences.

**Why this matters:** Deep comparison chains (e.g., PNG chunk headers → CRC checks → filter reconstruction) are a *nested* minimax problem. Each comparison is a ply in the game. Alpha-beta pruning can cut the search space exponentially.

### 3.5 Minimax robust corpus admission (minimax estimator)

**Problem:** Corpus admission is currently based on a single signal: does this input find new edges? The `rate_distortion.py` module does corpus minimization preserving diversity. But neither considers *robustness*: an input that covers a rare edge only under a narrow set of conditions is fragile.

**Proposal:** Apply the minimax estimator framework to corpus admission. Define the "risk" of a corpus as the maximum edge-coverage loss if any single input is removed. The minimax-robust corpus is the one that minimizes this maximum loss:

$$\min_{\text{corpus } C} \max_{c \in C} \text{coverage}(C \setminus \{c\})$$

This is the direct analog of the minimax estimator's "least favorable prior" — the corpus that is robust to the worst-case loss of any single seed.

**Where:** `core/rate_distortion.py` and `services/corpus_manager.py`. The existing greedy set-cover minimization (`minimize` subcommand) is the *average-case* version of this problem. The minimax version is the *worst-case* analog.

---

## 4. Implementation priority and sequencing

| Priority | Integration | Effort | Risk | Expected impact |
|----------|-------------|--------|------|-----------------|
| **P0** | 3.2 Alpha-beta MCTS | Medium — new class, same interface | Low — parallel to existing UCT | High — deeper lineage search |
| **P1** | 3.1 Minimax-robust scheduler | Medium — extend Elo tracker | Low — additive to existing arbitration | Medium — lower variance on heterogeneous targets |
| **P1** | 3.4 Comparison-wall minimax | High — new solver architecture | Medium — game model may be wrong | Very High — if it works |
| **P2** | 3.3 Adversarial operator sequencing | Medium — extend MonteCarloScheduler | Medium — multi-ply lookahead is expensive | Medium — better operator chains |
| **P2** | 3.5 Minimax robust corpus | Low — extend rate_distortion | Low — additive | Low — marginal improvement |

**Recommended first step:** Implement 3.2 (Alpha-beta MCTS) as a drop-in replacement for `MCTSSeedScheduler` and A/B it against the UCT baseline on `targets/png_read` and `targets/ffmpeg_read`. The existing `tools/bench_paired.py` framework supports exactly this comparison.

---

## 5. Key references from the literature

1. **Minimax estimator theory:** Lehmann & Casella, *Theory of Point Estimation* (1998) — the foundational text. Theorem 1 (Bayes estimator with constant risk is minimax) is the theoretical justification for 3.1.

2. **Alpha-beta pruning:** Knuth & Moore, "An analysis of alpha-beta pruning" (1975) — proves the $O(b^{d/2})$ bound. Pearl (1980, 1982) proves optimality for random trees. The move-ordering requirement (best moves first) maps directly to Thompson-sorted children in the lineage tree.

3. **Minimax in games:** Russell & Norvig, *AI: A Modern Approach* (4th ed., 2021), §5.2 — the canonical reference for minimax with alternate moves. The negamax simplification ($\max(a,b) = -\min(-a,-b)$) is already used implicitly in the fuzzer's MCTS code.

4. **Application to fuzzing:** Alphuzz (ACSAC 2021) — applies MCTS to seed scheduling, which is exactly what `core/schedulers/mcts.py` does. The gap is that Alphuzz uses UCT; this proposal is to upgrade to alpha-beta minimax.

5. **Adversarial bandits:** EXP3 (Auer et al., 1995) — already implemented in `core/schedulers/exp3.py`. The connection: EXP3 is the bandit version of minimax. The full-information version (minimax over the game tree) is what alpha-beta computes.

---

## 6. What would falsify this research direction

1. **Alpha-beta MCTS performs worse than UCT** on the lineage tree. This would happen if the pessimistic target model is too pessimistic, causing over-pruning. Fix: calibrate the pessimism factor against UCT on a held-out target.

2. **Minimax-robust scheduler selection is indistinguishable from Elo** on the current benchmark set. This would happen if all schedulers have similar worst-case regret (no single scheduler is catastrophically bad on any target). Fix: expand the benchmark set to include pathological targets (heavy comparison walls, deep paths).

3. **Comparison-wall minimax is slower than Z3** for single comparisons. This is expected — minimax shines on *nested* comparisons, not single ones. The test is whether it outperforms Z3 on targets with 3+ sequential comparison walls (e.g., PNG: signature → IHDR → IDAT CRC → filter reconstruction).

4. **The adversarial model is too abstract** to be useful. If the "target model" in the minimax search is just a heuristic that doesn't correlate with actual target behavior, the search will be no better than random. Fix: validate the target model by comparing predicted vs. actual coverage response on historical campaign data.

---

## 7. Relationship to existing handover docs

| Handover | Connection |
|----------|-----------|
| `handover_boltzmann_ab_2026-08-30.md` | The Boltzmann A/B is a single-instance of the minimax-robust scheduler analysis (3.1). Generalize it. |
| `handover_percolation_theory_2026-08-31.md` | Percolation theory models coverage as a phase transition. The minimax framework adds an *adversarial* phase transition: the target "defends" edges. |
| `handover_qea_hilbert_space_analysis_2026-08-31.md` | QEA uses quantum amplitude encoding. The minimax "least favorable prior" is the classical analog of QEA's worst-case superposition collapse. |
| `handover_persistence_mechanics_2026-08-29.md` | Persistent mode reduces per-execution cost. Alpha-beta pruning reduces per-decision search cost. Both are efficiency multipliers. |

---

## 8. Open questions for the next researcher

1. **Can the target model in alpha-beta be learned online?** The current proposal uses a fixed pessimistic heuristic. A learned model (e.g., a small neural net predicting coverage response from seed features) would be more accurate but adds training overhead.

2. **Is there a transposition table for the operator game tree?** The same operator sequence can be reached via different paths. Caching minimax values (like a chess engine's transposition table) could dramatically speed up 3.3.

3. **Does the minimax estimator framework extend to the *corpus* level?** Instead of admitting inputs that find new edges, admit inputs that are *robust* to the worst-case removal of other inputs. This is 3.5 but at a deeper level.

4. **Can alpha-beta pruning be applied to the *mutation* level?** Instead of searching over seeds, search over individual mutations within a single seed. The "minimizing player" is the target's response to each mutation. This would be a very fine-grained search but could solve comparison walls (3.4) more directly.

5. **What is the relationship between the spectral gap of the operator Markov chain (already computed in `MonteCarloScheduler.spectral_gap()`) and the minimax value?** A small spectral gap means the fuzzer is "stuck" in a narrow operator cycle — this is exactly the condition where minimax lookahead should help most.
