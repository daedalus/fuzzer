"""HierarchicalBanditScheduler: two-level (category → operator) bandit."""

import random

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, category_of


class HierarchicalBanditScheduler:
    """Two-level hierarchical bandit for operator selection.

    A top-level bandit selects an *operator category*, then a bottom-level
    bandit selects a specific *operator within that category*. Both levels
    use Thompson sampling with Beta-Bernoulli posteriors.

    Categories group structurally similar operators:
        - bit:       bit-level flips and transpositions
        - byte:      single-byte mutations (interesting values, arithmetic, etc.)
        - block:     block-level insert/delete/duplicate/transpose
        - dict:      dictionary-based token operations
        - structural: splice, crossover, type-aware replacements
        - radamsa:   Radamsa-style mutations (fuse, tree, line, UTF-8)
        - format:    format-aware (PNG, JPEG, BMP, GZIP, ZLIB)
        - adaptive:  learned/meta operators (markov, CEM, cmplog, havoc, etc.)

    The top-level gets credit (alpha/beta update) whenever *any* operator
    in the selected category produces a discovery, naturally boosting
    categories with collectively high yield.
    """

    supports_priors = True  # top-level accepts format-specific priors

    # Operator categories: every operator known to exist
    # (shared taxonomy from core.operator_categories, kept as a class
    # attribute for backward compatibility with external read sites)
    CATEGORIES: dict[str, set[str]] = OPERATOR_CATEGORIES

    def __init__(
        self,
        arm_decay: float = 0.999,
        decay_interval: int = 100,
        max_pseudocount: float = 200.0,
    ):
        self.arm_decay = arm_decay
        self.decay_interval = decay_interval
        # Ceiling on alpha + beta for any posterior. Beta(a, b) has variance
        # ~1/(a+b), so an uncapped posterior becomes arbitrarily confident and
        # its Thompson samples collapse onto the mean. That is fatal at the
        # top level here: a category whose mean drifts to ~0.02 before the
        # good operator inside it is discovered stops being sampled at all,
        # and the bottom level never gets the chance to find it. Measured
        # starvation of the best arm's whole category on 3 of 200 seeds --
        # and, separately, near-total inability to leave a decayed arm.
        #
        # Capping the total rescales alpha and beta together, so the posterior
        # mean is preserved while its variance stops shrinking: sliding-window
        # Thompson sampling. This is what arm_decay was reaching for, but at
        # 0.999 per 100 pulls it only removes ~6% of the mass over a 6000-pull
        # campaign, which is far too slow to matter.
        self.max_pseudocount = max_pseudocount

        # Top-level: Beta posteriors per category
        self.cat_alpha: dict[str, float] = {}
        self.cat_beta: dict[str, float] = {}

        # Bottom-level: Beta posteriors per operator
        self.op_alpha: dict[str, float] = {}
        self.op_beta: dict[str, float] = {}

        # Reverse lookup: operator name → category name.
        #
        # Sorted, because CATEGORIES maps to *sets*: iterating them directly
        # makes both this mapping and the downstream order of betavariate()
        # calls depend on PYTHONHASHSEED. Measured with the RNG seed pinned at
        # 92, tail share on the best arm ranged from 0.001 to 0.998 across
        # hash seeds, with 4 of 26 producing total starvation. That means
        # --seed did not determine this scheduler's behaviour and a crash
        # found under hierarchical scheduling could not be replayed.
        self._op_to_cat: dict[str, str] = {}
        for cat in sorted(self.CATEGORIES):
            for op in sorted(self.CATEGORIES[cat]):
                self._op_to_cat[op] = cat

        self._total_pulls: int = 0

    def _cat_for(self, name: str) -> str:
        """Category for *name*, memoized, resolved against the live registry.

        ``_op_to_cat`` is seeded from ``CATEGORIES``, a snapshot taken when
        ``core.operator_categories`` was imported. Anything registered after
        that — every operator added through ``REGISTRY.register_mutator()``,
        the documented extension path — is absent from it.

        This used to be a bare ``self._op_to_cat.get(name)``, and both
        ``init_arm`` and ``select_op`` treated a miss as "skip this
        operator". The arm was then never initialized *and* never a
        candidate: measured at 0 pulls out of 60,000 selections for a
        mutator the registry knew about and every other scheduler could
        reach. A scheduler that cannot select part of the operator table is
        not exploring a smaller space, it is exploring the wrong one.

        Falling back to ``UNCATEGORIZED`` rather than dropping the arm keeps
        the unknown-name case (test harnesses, an op list from a state file
        written by a different build) selectable too — it lands in its own
        top-level bucket, which is the honest answer for an operator whose
        structural kin are unknown.
        """
        cat = self._op_to_cat.get(name)
        if cat is None:
            cat = category_of(name)
            self._op_to_cat[name] = cat
        return cat

    def init_arm(self, name: str) -> None:
        """Register an operator. Initializes both category and operator posteriors."""
        cat = self._cat_for(name)
        self.cat_alpha.setdefault(cat, 1.0)
        self.cat_beta.setdefault(cat, 1.0)
        self.op_alpha.setdefault(name, 1.0)
        self.op_beta.setdefault(name, 1.0)

    def select_op(self, ops: list[str]) -> str:
        """Select operator via hierarchical Thompson sampling.

        1. Map available operators to their categories.
        2. Thompson-sample from category posteriors to pick a category.
        3. Thompson-sample from operator posteriors within that category.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        # Map available ops to categories.
        #
        # cat_ops preserves the caller's operator order, and its keys are
        # iterated below instead of the set: iterating avail_cats made the
        # sequence of betavariate() draws depend on PYTHONHASHSEED, so a fixed
        # --seed did not fix this scheduler's behaviour at all.
        # _cat_for(), not _op_to_cat.get(): a candidate missing from the
        # snapshot taxonomy used to be dropped from cat_ops here and was
        # therefore unselectable no matter how often it was offered.
        avail_cats: set[str] = set()
        cat_ops: dict[str, list[str]] = {}
        for op in ops:
            cat = self._cat_for(op)
            avail_cats.add(cat)
            cat_ops.setdefault(cat, []).append(op)

        # If no categorical mapping found, fall back to uniform random
        if not avail_cats:
            return random.choice(ops)

        # Apply periodic decay to both levels
        if (
            self.arm_decay < 1.0
            and self.decay_interval > 0
            and self._total_pulls > 0
            and self._total_pulls % self.decay_interval == 0
        ):
            for k in self.cat_alpha:
                self.cat_alpha[k] *= self.arm_decay
                self.cat_beta[k] *= self.arm_decay
            for k in self.op_alpha:
                self.op_alpha[k] *= self.arm_decay
                self.op_beta[k] *= self.arm_decay

        # Top-level: Thompson sample categories
        cat_scores = {}
        for cat in cat_ops:
            a = self.cat_alpha.get(cat, 1.0)
            b = self.cat_beta.get(cat, 1.0)
            cat_scores[cat] = random.betavariate(a, b)
        chosen_cat = max(cat_scores, key=cat_scores.get)

        # Bottom-level: Thompson sample operators within the chosen category
        op_candidates = cat_ops.get(chosen_cat, ops)
        op_scores = {}
        for op in op_candidates:
            a = self.op_alpha.get(op, 1.0)
            b = self.op_beta.get(op, 1.0)
            op_scores[op] = random.betavariate(a, b)
        return max(op_scores, key=op_scores.get)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update both category and operator posteriors.

        The category posterior is updated based on whether *any* operator
        in that category succeeded. This means productive categories rise
        even when individual operators within them have mixed results.
        """
        self._total_pulls += 1
        cat = self._cat_for(name)

        # A Beta posterior is a pair, but the branches below each move one
        # side only, so an arm that has seen a single kind of outcome exists
        # in one dict and not the other. The decay block in select_op() walks
        # op_alpha and subscripts op_beta bare, which is a KeyError at the
        # next decay tick. init_arm() pairs both dicts for operators armed at
        # startup; ones that self-register at runtime through
        # REGISTRY.register_mutator() reach record() without it.
        #
        # The beta side is the sentinel: it is missing on a first success and
        # created here on a first failure, so testing it alone covers both
        # entry orders and costs one failed dict lookup per call afterwards.
        if name not in self.op_beta:
            self.op_alpha.setdefault(name, 1.0)
            self.op_beta[name] = 1.0
        if cat not in self.cat_beta:
            self.cat_alpha.setdefault(cat, 1.0)
            self.cat_beta[cat] = 1.0

        # Update bottom-level (per-operator)
        if success:
            self.op_alpha[name] += weight
        else:
            self.op_beta[name] += 1

        # Update top-level (per-category) — same success signal
        if success:
            self.cat_alpha[cat] += weight
        else:
            self.cat_beta[cat] += 1

        self._cap(self.op_alpha, self.op_beta, name)
        self._cap(self.cat_alpha, self.cat_beta, cat)

    def _cap(self, alpha: dict[str, float], beta: dict[str, float], key: str) -> None:
        """Rescale a Beta posterior so alpha + beta stays within the ceiling.

        Preserves the posterior mean and floors both parameters at 1.0 so the
        distribution stays proper.
        """
        if self.max_pseudocount <= 0.0:
            return
        a = alpha.get(key, 1.0)
        b = beta.get(key, 1.0)
        total = a + b
        if total <= self.max_pseudocount:
            return
        scale = self.max_pseudocount / total
        alpha[key] = max(a * scale, 1.0)
        beta[key] = max(b * scale, 1.0)

    def bandit_stats(self) -> dict:
        """Return hierarchical bandit diagnostics."""
        top_cat = (
            max(self.cat_alpha, key=lambda c: self.cat_alpha[c] / self.cat_beta.get(c, 1))
            if self.cat_alpha
            else None
        )
        return {
            "hierarchical_pulls": self._total_pulls,
            "categories": len(self.cat_alpha),
            "top_category": top_cat,
        }
