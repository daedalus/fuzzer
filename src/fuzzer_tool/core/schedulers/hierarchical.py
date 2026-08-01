"""HierarchicalBanditScheduler: two-level (category → operator) bandit."""

import random


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
    CATEGORIES: dict[str, set[str]] = {
        "bit": {
            "bit_flip",
            "bit_offset_flip",
            "bit_offset_span",
            "bit_transpose_8",
            "bit_transpose_16",
            "bit_transpose_32",
            "bit_transpose_64",
        },
        "byte": {
            "byte_flip",
            "interesting_8",
            "interesting_16",
            "interesting_32",
            "arithmetic",
            "random_bytes",
            "radamsa_num",
            "byte_shuffle",
            "byte_delete",
            "byte_insert",
            "swap_bytes",
            "endianness_swap",
        },
        "block": {
            "block_insert",
            "block_delete",
            "block_duplicate",
            "swap_regions",
            "repeat_clone",
            "truncate",
            "length_grow",
            "length_shrink",
            "length_boundary",
            "transpose_16",
            "transpose_32",
            "transpose_64",
            "simd_boundary",
        },
        "dict": {
            "dict_insert",
            "dict_replace",
            "dict_overwrite",
            "dict_prepend",
            "dict_append",
            "checksum_repair",
            "token_dup",
            "dict_compound",
        },
        "structural": {
            "splice",
            "splice_diff_located",
            "crossover",
            "type_replace",
            "ascii_num",
            "ascii_num_arithmetic",
            "insert_ascii_num",
            "tlv_mutate",
            "token_shuffle",
            "chunk_shuffle",
            "punctuation_insert",
            "special_strings",
            "magic_values",
        },
        "radamsa": {
            "fuse_this",
            "fuse_next",
            "fuse_old",
            "tree_mutate",
            "line_mutate",
            "utf8_widen",
            "utf8_insert",
        },
        "format": {
            "png_chunk_mutate",
            "png_crc_fix",
            "jpeg_chunk_mutate",
            "jpeg_crc_fix",
            "bmp_chunk_mutate",
            "gzip_chunk_mutate",
            "zlib_chunk_mutate",
            "format_lock",
            "pgs_chunk_mutate",
            "isobmff_chunk_mutate",
            "nal_chunk_mutate",
            "protobuf_chunk_mutate",
            "gif_chunk_mutate",
            "webp_chunk_mutate",
            "webm_chunk_mutate",
            "zip_chunk_mutate",
            "x86_chunk_mutate",
            "arm_chunk_mutate",
        },
        "adaptive": {
            "markov_bytes",
            "cem_bytes",
            "colorization",
            "skipdet_probe",
            "auto_extras",
            "redqueen_xform",
            "gradient_cmp",
            "redqueen",
            "havoc",
            "overwrite_copy",
            "overwrite_fixed",
            "clone_fixed",
            "regex_bomb",
        },
    }

    def __init__(
        self,
        arm_decay: float = 0.999,
        decay_interval: int = 100,
    ):
        self.arm_decay = arm_decay
        self.decay_interval = decay_interval

        # Top-level: Beta posteriors per category
        self.cat_alpha: dict[str, float] = {}
        self.cat_beta: dict[str, float] = {}

        # Bottom-level: Beta posteriors per operator
        self.op_alpha: dict[str, float] = {}
        self.op_beta: dict[str, float] = {}

        # Reverse lookup: operator name → category name
        self._op_to_cat: dict[str, str] = {}
        for cat, ops in self.CATEGORIES.items():
            for op in ops:
                self._op_to_cat[op] = cat

        self._total_pulls: int = 0

    def init_arm(self, name: str) -> None:
        """Register an operator. Initializes both category and operator posteriors."""
        cat = self._op_to_cat.get(name)
        if cat is None:
            return  # unknown operator, skip
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

        # Map available ops to categories
        avail_cats: set[str] = set()
        cat_ops: dict[str, list[str]] = {}
        for op in ops:
            cat = self._op_to_cat.get(op)
            if cat:
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
        for cat in avail_cats:
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
        cat = self._op_to_cat.get(name)
        if cat is None:
            return

        # Update bottom-level (per-operator)
        if success:
            self.op_alpha[name] = self.op_alpha.get(name, 1.0) + weight
        else:
            self.op_beta[name] = self.op_beta.get(name, 1.0) + 1

        # Update top-level (per-category) — same success signal
        if success:
            self.cat_alpha[cat] = self.cat_alpha.get(cat, 1.0) + weight
        else:
            self.cat_beta[cat] = self.cat_beta.get(cat, 1.0) + 1

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
