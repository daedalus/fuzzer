"""Mutation operator dispatch and execution.

Extracted from Fuzzer class (~lines 1116-1845). Contains:
- All _op_* handler methods
- build_dispatch() — maps operator names to handlers (via REGISTRY)
- _havoc_mutate() / _apply_single_mutation() — random compound mutations
- build_ops() — builds list of available operators (via REGISTRY)
- _select_op() — selects operator via scheduling strategy
- _select_position() — selects byte position for mutation
- mutate() — main mutation orchestrator

Operator names, categories, and availability conditions are the single
source of truth in ``fuzzer_tool.core.operator_registry.REGISTRY``; this
module only supplies the ``_op_*`` handlers that the registry dispatches to.
"""

import bisect
import logging
import struct
from array import array

from fuzzer_tool.core.crc32 import crc32
from fuzzer_tool.core.mutations import (
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    INTERESTING_UNSIGNED_8,
    INTERESTING_UNSIGNED_16,
    INTERESTING_UNSIGNED_32,
    MAGIC_TABLE,
    SPECIAL_STRINGS,
    could_be_arith,
    could_be_bitflip,
    could_be_interest,
    radamsa_mutate_num,
    splice,
    splice_diff_located,
)
from fuzzer_tool.core.operator_registry import REGISTRY

log = logging.getLogger(__name__)

# Precomputed colorization lookup table: byte -> different value from same class.
# Avoids per-call table construction in _op_colorization().
_COLORIZE_TBL = bytearray(256)
for _b in range(256):
    if 0x30 <= _b <= 0x39:
        _COLORIZE_TBL[_b] = 0x30 + (_b * 7 + 3) % 10
    elif 0x41 <= _b <= 0x5A:
        _COLORIZE_TBL[_b] = 0x41 + (_b * 13 + 5) % 26
    elif 0x61 <= _b <= 0x7A:
        _COLORIZE_TBL[_b] = 0x61 + (_b * 17 + 7) % 26
    elif _b in (0x20, 0x09, 0x0A, 0x0D):
        _COLORIZE_TBL[_b] = (0x20, 0x09, 0x0A, 0x0D)[_b % 4]
    else:
        _COLORIZE_TBL[_b] = 0x21 + (_b * 31 + 11) % 94
_COLORIZE_TBL = bytes(_COLORIZE_TBL)


class OperatorEngine:
    """Manages mutation operator selection and execution.

    Holds a reference to the Fuzzer instance for accessing shared state
    (dictionary, markov, mc, grammar, corpus, seed_meta, etc.).
    """

    def __init__(self, fuzzer):
        self.f = fuzzer
        # Cache for _op_redqueen_xform: sorted cmplog pairs + version counter.
        # Rebuilt only when the pair list grows or is recreated (collect_tokens),
        # avoiding O(N log N) sort on every invocation (~2,500 sorts saved per run).
        self._redqueen_sorted_pairs: list | None = None
        self._redqueen_pair_lengths: array | None = None
        self._redqueen_sorted_version: int = 0

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            buf[byte_idx] ^= 1 << rng.randint(0, 7)

    def _op_bit_offset_flip(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if not buf:
            return
        total_bits = len(buf) * 8
        bit_offset = rng.randint(0, total_bits - 1)
        byte_idx = bit_offset >> 3
        bit_idx = bit_offset & 7
        buf[byte_idx] ^= 1 << bit_idx

    def _op_bit_offset_span(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if not buf:
            return
        total_bits = len(buf) * 8
        span_width = rng.weighted_choice(
            [1, 2, 3, 4, 5, 6, 7, 8], weights=[10, 15, 20, 20, 15, 10, 5, 5]
        )
        start_offset = rng.randint(0, max(0, total_bits - span_width))
        for i in range(span_width):
            bit_offset = start_offset + i
            if bit_offset >= total_bits:
                break
            byte_idx = bit_offset >> 3
            bit_idx = bit_offset & 7
            buf[byte_idx] ^= 1 << bit_idx

    def _op_simd_boundary(self, buf, _byte_idx, _data):
        """Resize buffer to SIMD boundary lengths (AVX2: 32, SSE2: 16)."""
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import SIMD_BOUNDARIES

        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.choice(SIMD_BOUNDARIES)))
            return
        target_len = rng.choice(SIMD_BOUNDARIES)
        current_len = len(buf)
        if target_len == current_len:
            return
        elif target_len < current_len:
            del buf[target_len:]
        else:
            grow = min(target_len - current_len, self.f.max_len - current_len)
            if grow > 0:
                buf.extend(bytearray(grow))  # zero-filled, fast

    def _op_regex_bomb(self, buf, _byte_idx, _data):
        """Replace input with a known regex backtracking bomb pattern."""
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import REGEX_BOMBS

        pattern = rng.choice(REGEX_BOMBS).encode()
        if len(buf) < len(pattern):
            buf.extend(b"\x00" * (len(pattern) - len(buf)))
        # Insert bomb at random position
        pos = rng.randint(0, max(0, len(buf) - len(pattern)))
        buf[pos : pos + len(pattern)] = pattern

    def _op_clone_fixed(self, buf, _byte_idx, _data):
        """Insert a block of repeated constant bytes (AFL++ clone_fixed)."""
        rng = self.f._rand_pool
        if not buf or len(buf) >= self.f.max_len:
            return
        fill_byte = rng.choice([buf[rng.randint(0, len(buf) - 1)], 0, 0xFF])
        block_size = rng.randint(1, min(32, self.f.max_len - len(buf)))
        ins_pos = rng.randint(0, len(buf))
        buf[ins_pos:ins_pos] = bytes([fill_byte] * block_size)

    def _op_overwrite_copy(self, buf, _byte_idx, _data):
        """Overwrite a region with bytes from another position (AFL++ overwrite_copy)."""
        rng = self.f._rand_pool
        if len(buf) < 2:
            return
        src_len = rng.randint(1, min(16, len(buf)))
        src_pos = rng.randint(0, len(buf) - src_len)
        dst_pos = rng.randint(0, max(0, len(buf) - src_len))
        if src_pos != dst_pos:
            buf[dst_pos : dst_pos + src_len] = buf[src_pos : src_pos + src_len]

    def _op_overwrite_fixed(self, buf, _byte_idx, _data):
        """Overwrite a region with repeated constant bytes (AFL++ overwrite_fixed)."""
        rng = self.f._rand_pool
        if len(buf) < 2:
            return
        fill_byte = rng.choice([buf[rng.randint(0, len(buf) - 1)], 0, 0xFF])
        block_len = rng.randint(1, min(16, len(buf)))
        dst_pos = rng.randint(0, len(buf) - block_len)
        buf[dst_pos : dst_pos + block_len] = bytes([fill_byte] * block_len)

    def _op_redqueen_xform(self, buf, byte_idx, _data):
        """RedQueen transforms: solve comparisons via encoding transforms.

        Uses the encoding engine from ``rq_encodings.py`` (ported from Redqueen
        NDSS 2019) to find and replace comparison operands using a variety of
        encoding strategies: plain substitution, zero/sign extension, ASCII
        number representation, C-string termination, split 64-bit words, etc.

        Falls back to single-byte transforms (XOR, arithmetic, boundary) when
        the encoding engine yields no applicable mutations for the current pair.
        """
        f = self.f
        rng = f._rand_pool
        if not hasattr(f, "_cmplog") or not f._cmplog or not f._cmplog.pairs:
            return
        if not buf or len(buf) < 2:
            return
        from fuzzer_tool.core.rq_encodings import generate_mutations  # noqa: PLC0415

        # Sample up to 3 pairs from the cmplog pool, prefering shorter pairs
        # (they're more likely to be found in the buffer).
        # Use cached sorted pair list, resorting only when pairs change,
        # to avoid O(N log N) sort on every invocation.
        cmplog_pairs = f._cmplog.pairs
        _version = id(cmplog_pairs) + len(cmplog_pairs)
        if not self._redqueen_sorted_pairs or _version != self._redqueen_sorted_version:
            _temp = [(len(p[0]), p) for p in cmplog_pairs]
            _temp.sort(key=lambda x: x[0])
            self._redqueen_sorted_pairs = [p for n, p in _temp if n >= 2]
            self._redqueen_pair_lengths = array("I", (n for n, _p in _temp if n >= 2))
            self._redqueen_sorted_version = _version
        # Pairs with 2 <= len(op_a) <= len(buf) form a prefix of the
        # length-sorted list — find the cutoff with a bisect instead of
        # rescanning every pair on every invocation.
        cutoff = bisect.bisect_right(self._redqueen_pair_lengths, len(buf))
        pairs = self._redqueen_sorted_pairs[:cutoff]
        if not pairs:
            return
        _sample_idx = rng.sample(len(pairs), min(3, len(pairs)))
        sample = [pairs[i] for i in _sample_idx]
        input_bytes = bytes(buf)

        for op_a, op_b in sample:
            cmp_size = 512 if len(op_a) > 8 or len(op_b) > 8 else max(len(op_a), len(op_b)) * 8
            cmp_type = "STR" if len(op_a) > 8 else "CMP"

            mutations = generate_mutations(
                op_a,
                op_b,
                cmp_size,
                cmp_type,
                input_bytes,
                hammer=True,
                is_hash=f._cmplog.is_hash_candidate
                if hasattr(f._cmplog, "is_hash_candidate")
                else None,
            )
            if mutations:
                offsets, replacements, enc = rng.choice(mutations)
                for i, off in enumerate(offsets):
                    if i < len(replacements):
                        chunk = replacements[i]
                        end = off + len(chunk)
                        if end <= len(buf):
                            buf[off:end] = chunk
                return

        # Fallback: single-byte transforms on a random pair
        if not pairs:
            return
        op_a, _ = rng.choice(pairs)
        if len(op_a) <= len(buf):
            pos = 0
            candidates = []
            buf_bytes = bytes(buf)
            while pos <= len(buf_bytes) - len(op_a):
                idx = buf_bytes.find(op_a, pos)
                if idx == -1:
                    break
                candidates.append(idx)
                pos = idx + 1
                if len(candidates) >= 5:
                    break
            if candidates:
                offset = rng.choice(candidates)
                vb = int.from_bytes(op_a, "little") & 0xFF
                transform = rng.choice(
                    ["xor", "arithmetic", "boundary", "hex", "toupper", "tolower"]
                )
                if transform == "xor":
                    const = rng.randint(1, 255)
                    buf[offset] = (vb ^ const) & 0xFF
                elif transform == "arithmetic":
                    delta = rng.randint(-128, 127)
                    buf[offset] = (vb - delta) & 0xFF
                elif transform == "boundary":
                    buf[offset] = (vb - 1) & 0xFF
                elif transform == "hex":
                    hex_chars = b"0123456789abcdef"
                    buf[offset] = hex_chars[vb % 16]
                elif transform == "toupper":
                    if ord("a") <= buf[offset] <= ord("z"):
                        buf[offset] = buf[offset] - 0x20
                    else:
                        buf[offset] = ord("A") + (vb % 26)
                elif transform == "tolower":
                    if ord("A") <= buf[offset] <= ord("Z"):
                        buf[offset] = buf[offset] + 0x20
                    else:
                        buf[offset] = ord("a") + (vb % 26)

    # ── Fuse mutations (from Radamsa) ──────────────────────────────

    def _op_fuse_this(self, buf, _byte_idx, _data):
        """Jump between two similar positions within the same input.

        Radamsa sed-fuse-this: picks two positions in the buffer,
        finds a short common prefix at each, and swaps the tails.
        Creates structural hybrids without format awareness.
        """
        rng = self.f._rand_pool
        if len(buf) < 4:
            return
        p1 = rng.randint(0, len(buf) - 2)
        p2 = rng.randint(0, len(buf) - 2)
        if p1 == p2:
            p2 = (p2 + 1) % max(1, len(buf) - 1)
        max_pre = min(16, len(buf) - max(p1, p2))
        pre_len = 0
        for pre_len in range(max_pre):
            if buf[p1 + pre_len] != buf[p2 + pre_len]:
                break
        else:
            pre_len = max_pre
        if pre_len < 1:
            tail1 = bytes(buf[p1:])
            tail2 = bytes(buf[p2:])
            combined = bytearray(buf[:p1]) + tail2 + buf[p1:p2] + tail1
            if len(combined) <= 2 * len(buf) and len(combined) >= len(buf) // 2:
                buf[:] = combined
            return
        a_end = p1 + pre_len
        b_end = p2 + pre_len
        tail_a = bytes(buf[a_end:])
        tail_b = bytes(buf[b_end:])
        result = bytearray(buf[:a_end]) + tail_b + buf[a_end:b_end] + tail_a
        if len(result) <= 2 * len(buf):
            buf[:] = result

    def _op_fuse_next(self, buf, _byte_idx, data):
        """Fuse current input with another corpus entry.

        Radamsa sed-fuse-next: fuses prefix of the current block
        with suffix of a random corpus entry at a shared position.
        """
        rng = self.f._rand_pool
        f = self.f
        corpus = getattr(f, "corpus", [])
        if not corpus or len(buf) < 3:
            return
        other = bytes(rng.choice(corpus))
        if other is data or other == buf:
            others = [c for c in corpus if c is not data]
            if not others:
                return
            other = bytes(rng.choice(others))
        if len(other) < 3:
            return
        split_point = min(len(buf), len(other)) // 2
        split_point = rng.randint(1, max(1, split_point))
        result = bytearray(buf[:split_point]) + other[split_point:]
        if len(result) <= f.max_len:
            buf[:] = result[: f.max_len]

    def _op_fuse_old(self, buf, _byte_idx, _data):
        """Fuse current input with a remembered block from earlier fuzz runs.

        Radamsa sed-fuse-old: maintains a ring buffer of previously
        mutated data and fuses fragments from it into the current input.
        """
        rng = self.f._rand_pool
        if len(buf) < 3:
            return
        if not hasattr(self, "_fuse_memory"):
            self._fuse_memory = []
            self._fuse_mem_max = 32
        mem = self._fuse_memory
        mem.append(bytes(buf))
        if len(mem) > self._fuse_mem_max:
            mem.pop(0)
        if len(mem) < 2:
            return
        old = rng.choice(mem[:-1])
        if len(old) < 3:
            return
        p_cur = rng.randint(1, len(buf) - 1)
        p_old = rng.randint(1, len(old) - 1)
        tail_old = bytes(old[p_old:])
        result = bytearray(buf[:p_cur]) + tail_old
        if len(result) <= getattr(self.f, "max_len", 65536):
            buf[:] = result

    def _op_tree_mutate(self, buf, _byte_idx, _data):
        """Lightweight delimiter-based tree mutation.

        Uses heuristic delimiter parsing (``tree_mutator.py``, ported from
        Radamsa ``sed-tree-*``) to parse the input into a tree using
        ``() {} [] \"\" ''``, then applies one of: delete, duplicate,
        swap, or stutter a subtree node.  Flattens back to bytes.

        No grammar file is needed — structure is detected from delimiters.
        """
        from fuzzer_tool.core.tree_mutator import lightweight_tree_mutate  # noqa: PLC0415

        result = lightweight_tree_mutate(bytes(buf), max_len=self.f.max_len, rng=self.f._rand_pool)
        if result != bytes(buf):
            buf[:] = result[: len(buf)]

    # ── UTF-8 confusion mutations (from Radamsa) ────────────────────

    def _op_utf8_widen(self, buf, _byte_idx, _data):
        """Widen 7-bit ASCII bytes into overlong UTF-8 sequences.

        Radamsa ``sed-utf8-widen``: replaces a random ASCII byte with an
        overlong UTF-8 encoding (2-byte sequence).  Exercises UTF-8
        length-checking bugs in parsers.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        # Find a 7-bit ASCII byte (0x00-0x7F)
        candidates = [i for i, b in enumerate(buf) if b < 0x80]
        if not candidates:
            return
        idx = rng.choice(candidates)
        b = buf[idx]
        # Overlong 2-byte encoding: 110xxxxx 10xxxxxx
        buf[idx : idx + 1] = bytes([0xC0 | (b >> 6), 0x80 | (b & 0x3F)])

    def _op_utf8_insert(self, buf, _byte_idx, _data):
        """Insert problematic Unicode byte sequences.

        Radamsa ``sed-utf8-insert``: inserts curated byte sequences that
        have caused security issues: BOMs, right-to-left override,
        zero-width joiners, illegal surrogates, NFKC expansion bombs, etc.
        """
        rng = self.f._rand_pool
        if not buf or len(buf) >= getattr(self.f, "max_len", 65536):
            return
        from fuzzer_tool.core.mutations import _FUNNY_UNICODE  # noqa: PLC0415

        seq = rng.choice(_FUNNY_UNICODE)
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = seq

    # ── Line-level mutations (from Radamsa) ─────────────────────────

    def _op_line_mutate(self, buf, _byte_idx, _data):
        """Line-level mutation: split on newlines, mutate the line list, rejoin.

        Radamsa ``sed-line-*`` operators: one of:
        - Delete a random line
        - Duplicate a random line
        - Swap two adjacent lines
        - Permute lines
        - Repeat a line
        - Insert a line from elsewhere in the buffer
        """
        rng = self.f._rand_pool
        if not buf or len(buf) < 4:
            return
        # Split on newline
        parts = buf.split(b"\n")
        if len(parts) < 2:
            return
        mutate = rng.choice(["del", "dup", "swap", "perm", "repeat", "clone"])
        if mutate == "del":
            idx = rng.randint(0, len(parts) - 1)
            del parts[idx]
        elif mutate == "dup":
            idx = rng.randint(0, len(parts) - 1)
            parts.insert(idx + 1, parts[idx])
        elif mutate == "swap" and len(parts) >= 2:
            idx = rng.randint(0, len(parts) - 2)
            parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
        elif mutate == "perm" and len(parts) >= 3:
            # Shuffle a subset of lines
            start = rng.randint(0, len(parts) - 3)
            end = min(start + rng.randint(2, 6), len(parts))
            segment = parts[start:end]
            rng.shuffle(segment)
            parts[start:end] = segment
        elif mutate == "repeat" and len(parts) >= 1:
            idx = rng.randint(0, len(parts) - 1)
            n = rng.randint(1, 32)
            for _ in range(n):
                parts.insert(idx, parts[idx])
        elif mutate == "clone" and len(parts) >= 2:
            src = rng.randint(0, len(parts) - 1)
            dst = rng.randint(0, len(parts) - 1)
            parts.insert(dst, parts[src])
        result = b"\n".join(parts)
        if len(result) <= getattr(self.f, "max_len", 65536) and result != bytes(buf):
            buf[:] = result

    def _op_colorization(self, buf, _byte_idx, _data):
        """Taint-aware byte randomization preserving character classes.

        AFL++ colorization: replace each byte with a random value from
        the same character class. Identifies comparison-relevant bytes.

        When cmplog data is available, uses the ``CmplogColorizer`` to
        prefer mutating bytes that appear in comparison operands, which
        increases the chance of cracking comparisons.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        tbl = _COLORIZE_TBL
        # Try to use cmplog-derived colorization mask for targeted selection
        f = self.f
        cmplog_pairs = getattr(f._cmplog, "pairs", None) if hasattr(f, "_cmplog") else None
        if cmplog_pairs:
            from fuzzer_tool.core.colorizer import CmplogColorizer  # noqa: PLC0415

            # Cache color mask per (input hash, cmplog pairs id)
            buf_bytes = bytes(buf)
            buf_hash = hash(buf_bytes)
            pairs_id = id(cmplog_pairs)
            cache = getattr(self, "_colorize_cache", None)
            if cache is None:
                self._colorize_cache = {}
                cache = self._colorize_cache
            cached = cache.get((buf_hash, pairs_id))
            if cached is None:
                colorizer = CmplogColorizer()
                mask = colorizer.colorize_from_cmplog(buf_bytes, cmplog_pairs)
                colorable = [i for i, m in enumerate(mask[: len(buf)]) if m == 0xFF]
                cached = colorable
                # Bounded cache: evict when > 256 entries
                if len(cache) > 256:
                    cache.clear()
                cache[(buf_hash, pairs_id)] = cached
            colorable = cached
            if colorable:
                n_mutate = max(1, min(len(colorable), len(buf) // rng.randint(2, 10)))
                indices = rng.choice_list(colorable, n_mutate)
                for idx in indices:
                    buf[idx] = tbl[buf[idx]]
                return
        # Fallback: random selection from entire buffer
        n_mutate = max(1, len(buf) // rng.randint(2, 10))
        indices = [rng.randint(0, len(buf) - 1) for _ in range(n_mutate)]
        for idx in indices:
            buf[idx] = tbl[buf[idx]]

    def _op_skipdet_probe(self, buf, _byte_idx, _data):
        """Block-flip probe to identify inert byte regions.

        AFL++ SkipDet: flip a block of bytes and check if the path changes.
        If not, those bytes are inert. This operator flips a random block
        to help discover which regions affect execution paths.
        """
        rng = self.f._rand_pool
        if len(buf) < 8:
            return
        # Pick a random block (10-25% of buffer) and flip all bytes
        block_size = rng.randint(len(buf) // 10, len(buf) // 4)
        block_size = max(2, min(block_size, len(buf)))
        start = rng.randint(0, len(buf) - block_size)
        # Bulk XOR via memoryview (avoids per-byte Python loop)
        mv = memoryview(buf)[start : start + block_size]
        for i in range(block_size):
            mv[i] ^= 0xFF

    def _op_auto_extras(self, buf, _byte_idx, _data):
        """Collect and inject sequences of "interesting" bytes.

        AFL++ auto-extras: during deterministic stages, consecutive bytes
        that affect the execution path are collected as magic values.
        This operator injects previously collected extras or generates
        new sequences of consecutive interesting bytes.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        f = self.f
        # Use dictionary tokens as "auto-extras" if available
        if f.dictionary:
            tid = (
                f._dict_scratch[f._dict_scratch_idx]
                if f._dict_scratch_idx < len(f._dict_scratch)
                else 0
            )
            f._dict_scratch_idx += 1
            token = f.dictionary[tid]
            if isinstance(token, str):
                token = token.encode()
            if len(token) <= len(buf):
                pos = rng.randint(0, len(buf) - len(token))
                buf[pos : pos + len(token)] = token
            return
        # Generate a short sequence of consecutive interesting bytes
        seq_len = rng.randint(2, 8)
        seq = bytes(
            rng.choice([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF])
            for _ in range(seq_len)
        )
        if len(buf) >= seq_len:
            pos = rng.randint(0, len(buf) - seq_len)
            buf[pos : pos + seq_len] = seq

    def _op_byte_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 0xFF

    def _op_interesting_8(self, buf, byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(byte_idx, k=5)
                if crash_vals:
                    buf[byte_idx] = rng.choice(crash_vals) & 0xFF
                    return
            vals = INTERESTING_UNSIGNED_8 if rng.random() < 0.5 else INTERESTING_8
            buf[byte_idx] = rng.choice(vals) & 0xFF

    def _op_interesting_16(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 2:
            idx = rng.randint(0, len(buf) - 2)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = rng.choice(crash_vals)
                    fmt = "<H" if v > 32767 or v < -32768 else "<h"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = rng.random() < 0.5
            vals = INTERESTING_UNSIGNED_16 if use_unsigned else INTERESTING_16
            v = rng.choice(vals)
            fmt = "<H" if v > 32767 or v < -32768 else "<h"
            struct.pack_into(fmt, buf, idx, v)

    def _op_interesting_32(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 4:
            idx = rng.randint(0, len(buf) - 4)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = rng.choice(crash_vals)
                    fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = rng.random() < 0.5
            vals = INTERESTING_UNSIGNED_32 if use_unsigned else INTERESTING_32
            v = rng.choice(vals)
            fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
            struct.pack_into(fmt, buf, idx, v)

    def _op_arithmetic(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS

        width = rng.choice([1, 2, 4, 8])
        if len(buf) >= width:
            max_start = len(buf) - width
            idx = (rng.randint(0, max_start) // width) * width
            delta = rng.choice(ARITHMETIC_DELTAS)
            if rng.random() < 0.5:
                delta = -delta
            if width == 1:
                buf[idx] = (buf[idx] + delta) & 0xFF
            elif width == 2:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into("<H", buf, idx, val)
                else:
                    val = (struct.unpack_from(">H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into(">H", buf, idx, val)
            elif width == 4:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into("<I", buf, idx, val)
                else:
                    val = (struct.unpack_from(">I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into(">I", buf, idx, val)
            elif width == 8:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into("<Q", buf, idx, val)
                else:
                    val = (struct.unpack_from(">Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into(">Q", buf, idx, val)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            buf[rng.randint(0, len(buf) - 1)] = rng.randint(0, 255)

    def _op_block_insert(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) < self.f.max_len:
            idx = rng.randint(0, len(buf))
            size = rng.randint(1, min(32, self.f.max_len - len(buf)))
            buf[idx:idx] = bytes(rng.randint(0, 255) for _ in range(size))

    def _op_block_delete(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 1:
            idx = rng.randint(0, len(buf) - 1)
            max_size = min(32, len(buf) - idx, len(buf) - 1)
            if max_size >= 1:
                del buf[idx : idx + rng.randint(1, max_size)]

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) < 2 or len(buf) >= self.f.max_len:
            return
        idx = rng.randint(0, len(buf) - 1)
        size = rng.randint(1, min(16, len(buf) - idx))
        block = buf[idx : idx + size]
        ins = rng.randint(0, len(buf))
        buf[ins:ins] = block

    def _op_dict_insert(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf[rng.randint(0, len(buf)) : 0] = token

    def _op_dict_replace(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and buf and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            idx = rng.randint(0, len(buf) - 1)
            end = min(idx + len(token), len(buf))
            buf[idx:end] = token[: end - idx]

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            return bytearray(token[: f.max_len])

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                return bytearray(token) + buf

    def _op_dict_append(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf.extend(token)

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool

        if buf and len(buf) >= 4:
            pos = rng.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = crc32(bytes(buf[:pos])).to_bytes(4, "big")

    def _op_crc_learn(self, buf, _byte_idx, _data):
        """Patch checksum fields using a polynomial recovered via BM/GCD."""
        if not buf or len(buf) < 4:
            return
        learner = getattr(self.f, "checksum_learner", None)
        if not learner:
            return
        poly = learner.ensure_poly()
        if poly is None:
            return
        rng = self.f._rand_pool

        # Try format-aware patching first
        patched = self._try_format_crc_patch(buf, learner, rng)
        if patched:
            buf[:] = patched
            return

        # Fallback: patch the last 4 bytes (common checksum placement)
        checksum = learner.compute_checksum(bytes(buf[:-4]))
        buf[-4:] = checksum.to_bytes(4, "big")

    # --- helpers for _op_crc_learn ---------------------------------------

    def _try_format_crc_patch(self, buf, learner, rng):
        """Attempt format-aware CRC patching; return patched bytes or None."""
        data = bytes(buf)
        # PNG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return self._patch_png_crc(data, learner)
        # ZIP
        if len(data) >= 30 and data[:4] == b"PK\x03\x04":
            return self._patch_zip_crc(data, learner, rng)
        return None

    def _patch_png_crc(self, data, learner):
        """Recompute all PNG chunk CRCs using the recovered polynomial."""
        if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        out = bytearray(data[:8])
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack_from(">I", data, pos)[0]
            chunk_type = data[pos + 4 : pos + 8]
            chunk_data = data[pos + 8 : pos + 8 + length]
            crc = learner.compute_checksum(chunk_type + chunk_data)
            out += struct.pack(">I", length)
            out += chunk_type
            out += chunk_data
            out += struct.pack(">I", crc)
            pos += 12 + length
            if chunk_type == b"IEND":
                break
        # Append any trailing data
        if pos < len(data):
            out += data[pos:]
        return bytes(out)

    def _patch_zip_crc(self, data, learner, rng):
        """Recompute CRCs in ZIP local file headers using the recovered polynomial."""
        out = bytearray()
        pos = 0
        while pos + 30 <= len(data):
            sig = struct.unpack_from("<I", data, pos)[0]
            if sig != 0x04034B50:
                out += data[pos:]
                break
            fname_len = struct.unpack_from("<H", data, pos + 26)[0]
            extra_len = struct.unpack_from("<H", data, pos + 28)[0]
            comp_size = struct.unpack_from("<I", data, pos + 18)[0]
            # Recompute CRC from filename (heuristic: filename is the data)
            fname = data[pos + 30 : pos + 30 + fname_len]
            crc = learner.compute_checksum(fname)
            header = bytearray(data[pos : pos + 30])
            struct.pack_into("<I", header, 14, crc & 0xFFFFFFFF)
            out += header
            out += data[pos + 30 : pos + 30 + fname_len + extra_len + comp_size]
            pos += 30 + fname_len + extra_len + comp_size
        return bytes(out)

    def _op_token_dup(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and buf and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf[rng.randint(0, len(buf)) : 0] = token

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            idx = rng.randint(0, len(buf) - 1)
            ctx = (
                bytes(buf[max(0, idx - self.f.markov.order) : idx]) if self.f.markov.order else b""
            )
            buf[idx] = self.f.markov.sample_byte(ctx)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if self.f.mc and self.f.mc.cem_fitted:
            if buf:
                buf[rng.randint(0, len(buf) - 1)] = self.f.mc.cem_byte(rng.randint(0, len(buf) - 1))
            else:
                return bytearray(self.f.mc.cem_sample(rng.randint(1, min(32, self.f.max_len))))

    def _op_splice(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        if len(self.f.corpus) >= 2:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(splice(a, b)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(splice(bytes(buf), rng.choice(others))[: self.f.max_len])

    def _op_splice_diff_located(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        if len(self.f.corpus) >= 2:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(splice_diff_located(a, b, rng=rng)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(
                    splice_diff_located(bytes(buf), rng.choice(others), rng=rng)[: self.f.max_len]
                )

    def _op_radamsa_num(self, buf, _byte_idx, data):
        """Radamsa-style number mutation on a random byte."""
        if not buf:
            return None
        rng = self.f._rand_pool
        pos = rng.randint(0, len(buf) - 1)
        val = radamsa_mutate_num(buf[pos], rng=rng)
        buf[pos] = val & 0xFF
        return buf

    def _op_crossover(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import crossover

        if len(self.f.corpus) >= 2 and buf:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(crossover(a, b, rng=rng)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(
                    crossover(bytes(buf), rng.choice(others), rng=rng)[: self.f.max_len]
                )

    def _op_type_replace(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import type_replace

        if buf:
            return bytearray(type_replace(bytes(buf))[: self.f.max_len])

    def _op_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ascii_num_replace

        if buf:
            return bytearray(ascii_num_replace(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_shuffle

        if buf and len(buf) > 1:
            return bytearray(byte_shuffle(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_byte_delete(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_delete

        if buf and len(buf) > 1:
            return bytearray(byte_delete(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_byte_insert(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_insert

        if buf and len(buf) < self.f.max_len:
            return bytearray(
                byte_insert(bytes(buf), self.f.max_len, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import insert_ascii_num

        if buf and len(buf) < self.f.max_len:
            return bytearray(
                insert_ascii_num(bytes(buf), self.f.max_len, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 2:
            return bytearray(
                transpose_bytes(bytes(buf), 2, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 4:
            return bytearray(
                transpose_bytes(bytes(buf), 4, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 8:
            return bytearray(
                transpose_bytes(bytes(buf), 8, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if buf:
            return bytearray(bit_transpose(bytes(buf), 1, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 2:
            return bytearray(bit_transpose(bytes(buf), 2, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 4:
            return bytearray(bit_transpose(bytes(buf), 4, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 8:
            return bytearray(bit_transpose(bytes(buf), 8, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_length_grow(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf and len(buf) < self.f.max_len:
            size = rng.randint(1, min(64, self.f.max_len - len(buf)))
            if size > 0:
                buf.extend(rng.randint(0, 255) for _ in range(size))

    def _op_length_shrink(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 2:
            del buf[rng.randint(1, len(buf) - 1) :]

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf and len(buf) < self.f.max_len:
            idx = rng.randint(0, len(buf) - 1)
            size = rng.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = idx + size
            if ins <= len(buf) and len(buf) + len(block) <= self.f.max_len:
                buf[ins:ins] = block

    def _op_truncate(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 2:
            del buf[rng.randint(2, len(buf)) :]

    def _op_length_boundary(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import LENGTH_BOUNDARIES

        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.randint(1, 32)))
            return
        # 30% chance: bias toward lengths that historically discovered edges
        if hasattr(self.f, "_length_tracker") and self.f._length_tracker and rng.random() < 0.3:
            recs = self.f._length_tracker.recommended_lengths(k=5)
            if recs:
                target_len = rng.choice(recs)
            else:
                target_len = rng.weighted_choice(
                    LENGTH_BOUNDARIES,
                    weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
                )
        else:
            target_len = rng.weighted_choice(
                LENGTH_BOUNDARIES,
                weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
            )
        current_len = len(buf)
        if target_len == current_len:
            return
        elif target_len < current_len:
            del buf[target_len:]
        else:
            grow = min(target_len - current_len, self.f.max_len - current_len)
            if grow > 0:
                buf.extend(bytearray(grow))  # zero-filled, fast

    def _op_swap_regions(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 4:
            i = rng.randint(0, len(buf) - 3)
            j = rng.randint(i + 2, len(buf) - 1)
            size = rng.randint(1, min(j - i, 16))
            a, b = buf[i : i + size], buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 2:
            i, j = rng.sample(len(buf), 2)
            buf[i], buf[j] = buf[j], buf[i]

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            width = rng.choice([2, 4, 8])
            if len(buf) >= width:
                idx = rng.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")

    def _op_tlv_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.tlv_mutate import tlv_mutate

        if buf:
            return bytearray(tlv_mutate(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_token_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        if buf and len(buf) >= 4:
            return bytearray(token_shuffle(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_gradient_cmp(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        if buf and self.f._cmplog and self.f._cmplog.pairs:
            return bytearray(
                gradient_cmp(bytes(buf), self.f._cmplog.pairs, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_special_strings(self, buf, _byte_idx, _data):
        """Insert a security-sensitive string at a random position.

        Ported from honggfuzz mangle_SpecialStrings: 44 curated strings
        covering SQL injection, XSS, path traversal, format strings,
        command injection, JSON edge cases, and control characters.
        """
        rng = self.f._rand_pool
        if not buf or len(buf) >= self.f.max_len:
            return
        s = rng.choice(SPECIAL_STRINGS)
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = s

    def _op_magic_values(self, buf, _byte_idx, _data):
        """Insert a magic/boundary value from the honggfuzz table.

        Ported from honggfuzz mangle_Magic: 229 hardcoded boundary values
        covering 1/2/4/8-byte widths in both LE and BE endianness.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        width, packed = rng.choice(MAGIC_TABLE)
        if len(buf) + width <= self.f.max_len:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = packed
        elif len(buf) >= width:
            pos = rng.randint(0, len(buf) - width)
            buf[pos : pos + width] = packed

    def _op_ascii_num_arithmetic(self, buf, _byte_idx, _data):
        """Mutate an existing ASCII number in-place using arithmetic.

        Ported from honggfuzz mangle_ASCIINumChange: finds a digit sequence
        and applies +1, -1, *2, /2, NOT, random replace, +random, or -random.
        """
        from fuzzer_tool.core.mutations import ascii_num_arithmetic

        if buf and len(buf) >= 1:
            result = ascii_num_arithmetic(bytes(buf), rng=self.f._rand_pool)
            if result is not None:
                return bytearray(result[: self.f.max_len])

    def _op_chunk_shuffle(self, buf, _byte_idx, data):
        """Shuffle fixed-size chunks, preserving chunk boundaries.

        Ported from honggfuzz mangle_ChunkShuffle: divides input into 1-4 byte
        chunks and swaps random pairs. Important for width-sensitive binary formats.
        When the parent seed has an inferred record stride, whole stride-sized
        records are shuffled instead — used with probability 0.5 so the operator
        does not over-latch on a single inferred stride.
        """
        from fuzzer_tool.core.mutations import chunk_shuffle

        if buf and len(buf) >= 8:
            rng = self.f._rand_pool
            parent_meta = self.f.seed_meta.get(data)
            stride = None
            if parent_meta and rng.random() < 0.5:
                stride = parent_meta.get("record_stride")
            return bytearray(chunk_shuffle(bytes(buf), rng=rng, stride=stride)[: self.f.max_len])

    def _op_block_shuffle_variable(self, buf, _byte_idx, _data):
        """Shuffle variable-width blocks using order-statistics spacings trick.

        Divides the input into 2-5 random-width blocks using the normalized-
        Exponential spacing trick (order_statistics.py Part 3). Unlike
        chunk_shuffle (fixed-width chunks), this produces variable-width
        blocks that can rearrange structural elements at any granularity.
        """
        from fuzzer_tool.core.mutations import block_shuffle_variable

        if buf and len(buf) >= 8:
            return bytearray(
                block_shuffle_variable(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_dict_compound(self, buf, _byte_idx, _data):
        """Insert two dictionary tokens concatenated with a random separator.

        Ported from honggfuzz mangle_DictionaryInsert: generates compound
        tokens like ``key=value`` or ``param1&param2`` by joining two
        dictionary entries with a random separator.
        """
        rng = self.f._rand_pool
        f = self.f
        if not f.dictionary or len(f.dictionary) < 2:
            return
        if not buf or len(buf) >= f.max_len:
            return
        from fuzzer_tool.core.mutations import DICT_COMPOUND_SEPARATORS

        t1 = rng.choice(f.dictionary)
        t2 = rng.choice(f.dictionary)
        sep = rng.choice(DICT_COMPOUND_SEPARATORS)
        if isinstance(t1, str):
            t1 = t1.encode()
        if isinstance(t2, str):
            t2 = t2.encode()
        compound = t1 + sep + t2
        if len(buf) + len(compound) <= f.max_len:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = compound

    def _op_punctuation_insert(self, buf, _byte_idx, _data):
        """Insert 1-4 random punctuation characters at a random position.

        Ported from honggfuzz mangle_Punctuation: useful for breaking
        escaping and structure in text-based protocols.
        """
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import PUNCTUATION_CHARS

        if not buf or len(buf) >= self.f.max_len:
            return
        n = rng.randint(1, min(4, self.f.max_len - len(buf)))
        chars = bytes(rng.choice(PUNCTUATION_CHARS) for _ in range(n))
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = chars

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        if self.f.grammar:
            return bytearray(
                self.f.grammar.mutate(bytes(buf), max_len=self.f.max_len, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_grammar_tree_mutate(self, buf, _byte_idx, data):
        if self.f.grammar:
            from fuzzer_tool.core.grammar import TreeMutator

            if not hasattr(self.f, "_tree_mutator"):
                self.f._tree_mutator = TreeMutator(self.f.grammar)
            parent_meta = self.f.seed_meta.get(data)
            stride = parent_meta.get("record_stride") if parent_meta else None
            tree = self.f._tree_mutator.parse(bytes(buf), chunk_size=stride)
            return bytearray(
                self.f._tree_mutator.mutate_tree(
                    tree, max_len=self.f.max_len, rng=self.f._rand_pool
                )[: self.f.max_len]
            )

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.png import PngChunkMutator, parse_png_chunks

        if not hasattr(self.f, "_png_mutator"):
            self.f._png_mutator = PngChunkMutator()
        # Set WFC mode from fuzzer state
        self.f._png_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        rng = self.f._rand_pool
        if parse_png_chunks(bytes(buf)):
            mutated = self.f._png_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._png_mutator._generate_random_png(self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.jpeg import JpegMutator, parse_jpeg_markers

        if not hasattr(self.f, "_jpeg_mutator"):
            self.f._jpeg_mutator = JpegMutator()
        self.f._jpeg_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        rng = self.f._rand_pool
        if parse_jpeg_markers(bytes(buf)):
            mutated = self.f._jpeg_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._jpeg_mutator._generate_random_jpeg(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations.jpeg import (
            STANDALONE_MARKERS,
            parse_jpeg_markers,
            serialize_jpeg_markers,
        )

        if buf:
            markers = parse_jpeg_markers(bytes(buf))
            if markers and len(markers) > 2:
                candidates = [
                    i
                    for i, m in enumerate(markers)
                    if m.marker not in STANDALONE_MARKERS and len(m.data) > 0
                ]
                if candidates:
                    idx = rng.choice(candidates)
                    marker = markers[idx]
                    data = bytearray(marker.data)
                    for _ in range(rng.randint(1, min(4, len(data)))):
                        data[rng.randint(0, len(data) - 1)] ^= 1 << rng.randint(0, 7)
                    marker.data = bytes(data)
                    return bytearray(serialize_jpeg_markers(markers)[: self.f.max_len])

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.gzip import GzipMutator, parse_gzip

        if not hasattr(self.f, "_gzip_mutator"):
            self.f._gzip_mutator = GzipMutator()
        rng = self.f._rand_pool
        if parse_gzip(bytes(buf)):
            mutated = self.f._gzip_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._gzip_mutator._generate_random_gzip(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, data):
        from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp

        if not hasattr(self.f, "_bmp_mutator"):
            self.f._bmp_mutator = BmpMutator()
        self.f._bmp_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        parent_meta = self.f.seed_meta.get(data)
        stride = parent_meta.get("record_stride") if parent_meta else None
        self.f._bmp_mutator.tile_bytes = stride
        rng = self.f._rand_pool
        if parse_bmp(bytes(buf)):
            mutated = self.f._bmp_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._bmp_mutator._generate_random_bmp(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_zlib_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.zlib import ZlibMutator, parse_zlib

        if not hasattr(self.f, "_zlib_mutator"):
            self.f._zlib_mutator = ZlibMutator()
        rng = self.f._rand_pool
        if parse_zlib(bytes(buf)):
            mutated = self.f._zlib_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._zlib_mutator._generate_random_zlib(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_format_lock(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.magic_lock import format_lock_havoc

        rng = self.f._rand_pool
        if buf:
            result = format_lock_havoc(bytes(buf), self.f.max_len, rng=rng)
            if result is not None:
                return bytearray(result[: self.f.max_len])

    def _op_pgs_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.pgs import PgsMutator, parse_pgs_segments

        if not hasattr(self.f, "_pgs_mutator"):
            self.f._pgs_mutator = PgsMutator()
        rng = self.f._rand_pool
        if parse_pgs_segments(bytes(buf)):
            mutated = self.f._pgs_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._pgs_mutator._generate_random_pgs(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_isobmff_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, parse_boxes

        if not hasattr(self.f, "_isobmff_mutator"):
            self.f._isobmff_mutator = IsobmffMutator()
        rng = self.f._rand_pool
        if parse_boxes(bytes(buf)):
            mutated = self.f._isobmff_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._isobmff_mutator._generate_random_isobmff(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_nal_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.nal import NalMutator, parse_nal_units

        if not hasattr(self.f, "_nal_mutator"):
            self.f._nal_mutator = NalMutator()
        rng = self.f._rand_pool
        if parse_nal_units(bytes(buf)):
            mutated = self.f._nal_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._nal_mutator._generate_random_nal_stream(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_protobuf_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.protobuf import ProtobufMutator, parse_protobuf

        if not hasattr(self.f, "_protobuf_mutator"):
            self.f._protobuf_mutator = ProtobufMutator()
        rng = self.f._rand_pool
        if parse_protobuf(bytes(buf)):
            mutated = self.f._protobuf_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._protobuf_mutator._generate_random_protobuf(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_gif_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.gif import GifMutator, parse_gif

        if not hasattr(self.f, "_gif_mutator"):
            self.f._gif_mutator = GifMutator()
        rng = self.f._rand_pool
        if parse_gif(bytes(buf)):
            mutated = self.f._gif_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._gif_mutator._generate_random_gif(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_webp_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.webp import WebpMutator, parse_webp

        if not hasattr(self.f, "_webp_mutator"):
            self.f._webp_mutator = WebpMutator()
        rng = self.f._rand_pool
        if parse_webp(bytes(buf)):
            mutated = self.f._webp_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._webp_mutator._generate_random_webp(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_webm_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.webm import WebmMutator, parse_webm

        if not hasattr(self.f, "_webm_mutator"):
            self.f._webm_mutator = WebmMutator()
        rng = self.f._rand_pool
        if parse_webm(bytes(buf)):
            mutated = self.f._webm_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._webm_mutator._generate_random_webm(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_zip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.zip import ZipMutator, parse_zip

        if not hasattr(self.f, "_zip_mutator"):
            self.f._zip_mutator = ZipMutator()
        rng = self.f._rand_pool
        if parse_zip(bytes(buf)):
            mutated = self.f._zip_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._zip_mutator._generate_random_zip(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_x86_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.x86 import X86Mutator, _decode_insns

        if not hasattr(self.f, "_x86_mutator"):
            self.f._x86_mutator = X86Mutator()
        rng = self.f._rand_pool
        if _decode_insns(bytes(buf)):
            mutated = self.f._x86_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._x86_mutator._generate_random_x86(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_arm_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.arm import ArmMutator, parse_arm

        if not hasattr(self.f, "_arm_mutator"):
            self.f._arm_mutator = ArmMutator()
        rng = self.f._rand_pool
        if parse_arm(bytes(buf)):
            mutated = self.f._arm_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._arm_mutator._generate_random_arm(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_elf_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.elf import ElfMutator

        if not hasattr(self.f, "_elf_mutator"):
            self.f._elf_mutator = ElfMutator()
        rng = self.f._rand_pool
        mutated = self.f._elf_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_recompress_zlib(self, buf, byte_idx, data):
        """Inflate, mutate the plaintext, re-deflate as a valid zlib stream.

        Falls back to havoc when the input is not an inflatable stream: the
        registry's bootstrap trickle can offer this operator before any real
        compressed input has been seen, and emitting the buffer unchanged
        would waste the execution.
        """
        from fuzzer_tool.core.mutations.recompress import recompress_zlib

        rng = self.f._rand_pool
        out = recompress_zlib(bytes(buf), max_len=self.f.max_len, rng=rng)
        if out is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(out[: self.f.max_len])

    def _op_recompress_gzip(self, buf, byte_idx, data):
        """Inflate, mutate the plaintext, re-deflate as a valid gzip member."""
        from fuzzer_tool.core.mutations.recompress import recompress_gzip

        rng = self.f._rand_pool
        out = recompress_gzip(bytes(buf), max_len=self.f.max_len, rng=rng)
        if out is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(out[: self.f.max_len])

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations.png import parse_png_chunks, serialize_png_chunks

        if buf:
            chunks = parse_png_chunks(bytes(buf))
            if chunks and len(chunks) > 1:
                candidates = [i for i, c in enumerate(chunks) if c.chunk_type != b"IEND"]
                if candidates:
                    idx = rng.choice(candidates)
                    chunk = chunks[idx]
                    if chunk.data:
                        data = bytearray(chunk.data)
                        for _ in range(rng.randint(1, min(4, len(data)))):
                            data[rng.randint(0, len(data) - 1)] ^= 1 << rng.randint(0, 7)
                        chunk.data = bytes(data)
                    else:
                        chunk.data = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 32)))
                    return bytearray(serialize_png_chunks(chunks)[: self.f.max_len])

    def _op_redqueen(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        parent_meta = self.f.seed_meta.get(data)
        if not (buf and parent_meta):
            return
        matches = parent_meta.get("redqueen_matches", [])
        offsets = parent_meta.get("redqueen_offsets", [])
        if matches:
            for _ in range(rng.randint(1, min(4, len(matches)))):
                off, op_a, op_b = rng.choice(matches)
                end = off + len(op_a)
                if end <= len(buf) and bytes(buf[off:end]) == op_a:
                    for j, b_val in enumerate(op_b):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets and self.f._cmplog and self.f._cmplog.tokens:
            for _ in range(rng.randint(1, min(4, len(offsets)))):
                off = rng.choice(offsets)
                if off < len(buf):
                    token = rng.choice(self.f._cmplog.tokens)
                    for j, b_val in enumerate(token):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets:
            for _ in range(rng.randint(1, min(4, len(offsets)))):
                off = rng.choice(offsets)
                if off < len(buf):
                    buf[off] ^= 0xFF

    def _op_havoc(self, buf, _byte_idx, data):
        """Havoc mutation with deterministic dedup: retry if fully redundant."""
        original = bytes(buf)
        self.havoc_mutate(buf)
        result = bytes(buf)
        if len(result) == len(original) and self._is_deterministically_redundant(original, result):
            # Fully redundant — apply one more random mutation
            self._apply_single_mutation(buf)
            result = bytes(buf)
        return result

    # ── Dispatch table: op name → handler method ───────────────────────
    def build_dispatch(self):
        return REGISTRY.dispatch(self)

    def havoc_mutate(self, buf: bytearray) -> bytearray:
        """Apply 2-8 random mutations (scaled up during stall recovery).

        During normal operation: 2-8 mutations.
        During stall recovery: 8-16 mutations (honggfuzz-style escalation).
        """
        rng = self.f._rand_pool
        if self.f._stall_recovery_active:
            n = rng.randint_list(8, 16, 1)[0]
        else:
            n = rng.randint_list(2, 8, 1)[0]
        for _ in range(n):
            self._apply_single_mutation(buf)
        return buf

    @staticmethod
    def _is_deterministically_redundant(original: bytes, candidate: bytes) -> bool:
        """Check if all byte diffs are covered by deterministic stages.

        Uses AFL++ duplicate-elimination helpers. If every differing byte
        could be produced by a bitflip, arithmetic, or interest-value
        mutation, the candidate is redundant with deterministic stages.
        """
        if len(original) != len(candidate):
            return False
        for a, b in zip(original, candidate, strict=True):
            if a == b:
                continue
            xor_val = a ^ b
            if could_be_bitflip(xor_val):
                continue
            if could_be_arith(a, b, 1):
                continue
            if could_be_interest(a, b, 1):
                continue
            return False
        return True

    def _apply_single_mutation(self, buf: bytearray):
        rng = self.f._rand_pool
        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.randint(1, 16)))
            return
        # Pre-fetch 4 random values in one vectorized call.
        # Each branch uses 2-4 values from this batch, avoiding N
        # individual randint/randrange Python calls.
        r = self.f._rand_pool.randint_list(0, 1 << 30, 4)
        op = r[0] % 11
        if op == 0:  # bit flip
            buf[r[1] % len(buf)] ^= 1 << (r[2] % 8)
        elif op == 1:  # byte set
            buf[r[1] % len(buf)] = r[2] % 256
        elif op == 2 and len(buf) > 1:  # byte swap
            i = r[1] % len(buf)
            j = (i + 1 + r[2] % (len(buf) - 1)) % len(buf)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.f.max_len:  # insert byte
            idx = r[1] % (len(buf) + 1)
            buf.insert(idx, r[2] % 256)
        elif op == 4 and len(buf) > 1:  # delete block
            idx = r[1] % len(buf)
            size = 1 + r[2] % min(len(buf) - 1, len(buf) - idx)
            del buf[idx : idx + size]
        elif op == 5 and len(buf) >= 4:  # CRC32 repair
            pos = r[1] % max(1, len(buf) - 3)
            buf[pos : pos + 4] = crc32(bytes(buf[:pos])).to_bytes(4, "big")
        elif op == 6 and len(buf) >= 2:  # swap regions
            i = r[1] % (len(buf) - 1)
            j = i + 1 + r[2] % (len(buf) - i - 1)
            size = 1 + r[3] % min(j - i, 8)
            a = buf[i : i + size]
            b = buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a
        elif op == 7 and buf:  # endianness swap
            width = 2 if r[1] % 2 == 0 else 4
            if len(buf) >= width:
                idx = r[2] % (len(buf) - width + 1)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")
        elif op == 8 and buf:  # byte insert
            if len(buf) < self.f.max_len:
                idx = r[1] % (len(buf) + 1)
                buf.insert(idx, r[2] % 256)
        elif op == 9 and buf:  # random byte
            idx = r[1] % len(buf)
            buf[idx] = r[2] % 256
        elif op == 10 and len(buf) >= 2:  # shuffle range
            start = r[1] % (len(buf) - 1)
            end = min(start + 2 + r[2] % 7, len(buf))
            region = buf[start:end]
            rng.shuffle(region)
            buf[start:end] = region

    # ── Operator selection logic ───────────────────────────────────────

    def build_ops(self, data: bytes) -> list[str]:
        """Build the list of available mutation operators from the registry."""
        return REGISTRY.available(self.f, data)

    def select_op(self, ops: list[str]) -> str:
        """Select a mutation operator using the active scheduling strategy."""
        f = self.f

        if f._stall_recovery_active:
            f._meta_strategy = "random_stall"
            return f._rand_pool.choice(ops)

        available = []
        if f._use_replicator and f._replicator:
            available.append("replicator")
        if f.mc and f.mc_bandit:
            available.append("bandit")
        if f._use_mopt and f._mopt:
            available.append("mopt")
        if f.mc and f.mc_cem and f.mc.cem_fitted:
            available.append("cem")
        if f._use_exp3 and f._exp3:
            available.append("exp3")
        if f._use_eps_greedy and f._eps_greedy:
            available.append("eps_greedy")
        if f._use_hierarchical and f._hierarchical:
            available.append("hierarchical")
        if f._use_gp_ucb and f._gp_ucb:
            available.append("gp_ucb")

        if f._use_elo and f._elo and len(available) >= 2:
            # Resolve the meta-strategy once per exec and reuse it for all
            # mutations within it. Elo ratings move once per mutation, so
            # re-sampling the strategy on every select_op call is
            # over-frequent; mutate() resets _meta_strategy_cached each exec.
            strategy = f._meta_strategy_cached
            if strategy is None or strategy not in available:
                strategy = f._elo.select_strategy(available)
                f._meta_strategy_cached = strategy
            f._meta_strategy = strategy
        elif f._use_elo and f._elo and available:
            strategy = available[0]
            f._meta_strategy = strategy
        else:
            strategy = None

        if f._use_elo and f._elo and strategy:
            f._meta_strategy_used.add(strategy)

        if strategy == "replicator" and f._replicator:
            op = f._replicator.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "mopt" and f._mopt:
            op, pid = f._mopt.select_op(ops)
            f._last_mopt_particles.append(pid)
        elif strategy == "bandit" and f.mc and f.mc_bandit:
            op = f.mc.select_op(ops, prev_op=f._prev_bandit_op)
            f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif strategy == "cem" and f.mc and f.mc_cem:
            op = (
                f.mc.select_op(ops, prev_op=f._prev_bandit_op)
                if f.mc_bandit
                else f._rand_pool.choice(ops)
            )
            if f.mc_bandit:
                f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif strategy == "exp3" and f._exp3:
            op = f._exp3.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "eps_greedy" and f._eps_greedy:
            op = f._eps_greedy.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "hierarchical" and f._hierarchical:
            op = f._hierarchical.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "gp_ucb" and f._gp_ucb:
            op = f._gp_ucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_replicator and f._replicator:
            op = f._replicator.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_mopt and f._mopt:
            op, pid = f._mopt.select_op(ops)
            f._last_mopt_particles.append(pid)
        elif f.mc and f.mc_bandit:
            op = f.mc.select_op(ops, prev_op=f._prev_bandit_op)
            f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif f._use_exp3 and f._exp3:
            op = f._exp3.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_eps_greedy and f._eps_greedy:
            op = f._eps_greedy.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_hierarchical and f._hierarchical:
            op = f._hierarchical.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_gp_ucb and f._gp_ucb:
            op = f._gp_ucb.select_op(ops)
            f._last_mopt_particles.append(None)
        else:
            op = f._rand_pool.choice(ops)
            f._last_mopt_particles.append(None)
        return op

    def select_position(self, buf: bytearray, data: bytes) -> int:
        """Select a byte position for mutation using MI/TE/sensitivity/crash-MI/random."""
        f = self.f
        if not buf:
            return 0
        buf_len = len(buf)
        te_pos = f._get_te_weighted_position(buf_len) if f._use_transfer_entropy and f._te else None
        mi_pos = f._mi.weighted_position(buf_len) if f._use_mi and f._mi else None
        # Sensitivity is a per-seed score cache: when disabled the tracker is
        # never populated, so the call would always return None.  Gate it like
        # MI/TE instead of paying the lookup + branches on every mutation.
        sens_pos = (
            f._sensitivity.get_weighted_position(data, buf_len)
            if f._use_sensitivity and f._sensitivity
            else None
        )
        crash_mi_pos = None
        if f._crash_mi and f._crash_mi.total_execs >= f._crash_mi.min_observations:
            crash_mi_pos = f._crash_mi.weighted_position(buf_len)
        candidates = [p for p in [sens_pos, te_pos, mi_pos, crash_mi_pos] if p is not None]
        if candidates:
            return f._rand_pool.choice(candidates)
        return f._rand_pool.randint(0, buf_len - 1)

    # ── Main mutation orchestrator ─────────────────────────────────────

    def mutate(self, data: bytes) -> bytes:
        from fuzzer_tool.core.similarity import hamming_distance

        f = self.f
        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * f._rand_pool.randint_list(1, 32, 1)[0])

        ops = self.build_ops(data)
        f._last_ops_used = []
        f._last_ops_with_sites = []
        f._last_mopt_particles = []
        if not hasattr(f, "_prev_bandit_op"):
            f._prev_bandit_op = None
        f._meta_strategy = None
        f._meta_strategy_cached = None  # reset per-exec elo strategy cache

        n_mutations = f.mutations_per_input
        # Apply seed-level energy multiplier from SeedScorer
        if hasattr(f, "_last_perf_score") and f._last_perf_score != 100.0:
            n_mutations = max(1, int(n_mutations * f._last_perf_score / 100.0))
        if f._stall_recovery_active:
            n_mutations = max(n_mutations, 16)

        # Pre-fetch dictionary indices for dict-aware operators in one
        # vectorized call, replacing N individual random.choice(f.dictionary)
        # calls across _op_dict_* methods.
        if f.dictionary:
            f._dict_scratch = f._rand_pool.randint_list(
                0, len(f.dictionary) - 1, max(n_mutations * 8, 64)
            )
            f._dict_scratch_idx = 0

        # Pre-generate buffer lengths for select_position fallback.
        # select_position is called once per mutation, and when no
        # MI/TE/sensitivity is active it falls back to randrange(len(buf)).
        # Pre-computing positions doesn't work here because len(buf) changes
        # during mutation, so we handle this per-call below.

        for _ in range(n_mutations):
            op = self.select_op(ops)
            f._last_ops_used.append(op)

            byte_idx = self.select_position(buf, data)
            f._last_mutation_offset = byte_idx
            f._last_ops_with_sites.append((op, byte_idx))
            old_len = len(buf)

            result = f._op_dispatch[op](buf, byte_idx, data)
            if result is not None:
                if op == "havoc":
                    if f._frameshift.relations:
                        buf = bytearray(result[: f.max_len])
                        f._frameshift.apply_to_buffer(buf)
                        result = bytes(buf)
                    f._last_hamming_distance = (
                        hamming_distance(data, result) if len(data) == len(result) else -1
                    )
                    return result
                new_len = min(len(result), f.max_len)
                if f._frameshift.relations:
                    if new_len > old_len:
                        f._frameshift.on_insert(byte_idx, new_len - old_len)
                    elif new_len < old_len:
                        f._frameshift.on_delete(byte_idx, old_len - new_len)
                buf = (
                    result[: f.max_len]
                    if isinstance(result, bytearray)
                    else bytearray(result[: f.max_len])
                )

        if f._frameshift.relations:
            f._frameshift.apply_to_buffer(buf)

        result = bytes(buf)
        f._last_hamming_distance = (
            hamming_distance(data, result) if len(data) == len(result) else -1
        )
        return result
