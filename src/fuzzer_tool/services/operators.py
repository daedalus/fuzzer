"""Mutation operator dispatch and execution.

Extracted from Fuzzer class (~lines 1116-1845). Contains:
- All _op_* handler methods
- _build_dispatch() — maps operator names to handlers
- _havoc_mutate() / _apply_single_mutation() — random compound mutations
- _build_ops() — builds list of available operators
- _select_op() — selects operator via scheduling strategy
- _select_position() — selects byte position for mutation
- mutate() — main mutation orchestrator
"""

import logging
import random
import struct

from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    FORMAT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    INTERESTING_UNSIGNED_8,
    INTERESTING_UNSIGNED_16,
    INTERESTING_UNSIGNED_32,
    MUTATIONS,
    splice,
)

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

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 1 << random.randint(0, 7)

    def _op_bit_offset_flip(self, buf, _byte_idx, _data):
        if not buf:
            return
        total_bits = len(buf) * 8
        bit_offset = random.randint(0, total_bits - 1)
        byte_idx = bit_offset >> 3
        bit_idx = bit_offset & 7
        buf[byte_idx] ^= 1 << bit_idx

    def _op_bit_offset_span(self, buf, _byte_idx, _data):
        if not buf:
            return
        total_bits = len(buf) * 8
        span_width = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8], weights=[10, 15, 20, 20, 15, 10, 5, 5]
        )[0]
        start_offset = random.randint(0, max(0, total_bits - span_width))
        for i in range(span_width):
            bit_offset = start_offset + i
            if bit_offset >= total_bits:
                break
            byte_idx = bit_offset >> 3
            bit_idx = bit_offset & 7
            buf[byte_idx] ^= 1 << bit_idx

    def _op_simd_boundary(self, buf, _byte_idx, _data):
        """Resize buffer to SIMD boundary lengths (AVX2: 32, SSE2: 16)."""
        from fuzzer_tool.core.mutations import SIMD_BOUNDARIES

        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.choice(SIMD_BOUNDARIES)))
            return
        target_len = random.choice(SIMD_BOUNDARIES)
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
        from fuzzer_tool.core.mutations import REGEX_BOMBS

        pattern = random.choice(REGEX_BOMBS).encode()
        if len(buf) < len(pattern):
            buf.extend(b"\x00" * (len(pattern) - len(buf)))
        # Insert bomb at random position
        pos = random.randint(0, max(0, len(buf) - len(pattern)))
        buf[pos : pos + len(pattern)] = pattern

    def _op_clone_fixed(self, buf, _byte_idx, _data):
        """Insert a block of repeated constant bytes (AFL++ clone_fixed)."""
        if not buf or len(buf) >= self.f.max_len:
            return
        fill_byte = random.choice([buf[random.randint(0, len(buf) - 1)], 0, 0xFF])
        block_size = random.randint(1, min(32, self.f.max_len - len(buf)))
        ins_pos = random.randint(0, len(buf))
        buf[ins_pos:ins_pos] = bytes([fill_byte] * block_size)

    def _op_overwrite_copy(self, buf, _byte_idx, _data):
        """Overwrite a region with bytes from another position (AFL++ overwrite_copy)."""
        if len(buf) < 2:
            return
        src_len = random.randint(1, min(16, len(buf)))
        src_pos = random.randint(0, len(buf) - src_len)
        dst_pos = random.randint(0, max(0, len(buf) - src_len))
        if src_pos != dst_pos:
            buf[dst_pos : dst_pos + src_len] = buf[src_pos : src_pos + src_len]

    def _op_overwrite_fixed(self, buf, _byte_idx, _data):
        """Overwrite a region with repeated constant bytes (AFL++ overwrite_fixed)."""
        if len(buf) < 2:
            return
        fill_byte = random.choice([buf[random.randint(0, len(buf) - 1)], 0, 0xFF])
        block_len = random.randint(1, min(16, len(buf)))
        dst_pos = random.randint(0, len(buf) - block_len)
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
        import random as _rand

        f = self.f
        if not hasattr(f, "_cmplog") or not f._cmplog or not f._cmplog.pairs:
            return
        if not buf or len(buf) < 2:
            return
        from fuzzer_tool.core.rq_encodings import generate_mutations  # noqa: PLC0415

        # Sample up to 3 pairs from the cmplog pool, prefering shorter pairs
        # (they're more likely to be found in the buffer).
        pairs = [p for p in f._cmplog.pairs if 2 <= len(p[0]) <= len(buf)]
        if not pairs:
            return
        pairs.sort(key=lambda p: len(p[0]))
        sample = _rand.sample(pairs, min(3, len(pairs)))

        for op_a, op_b in sample:
            cmp_size = 512 if len(op_a) > 8 or len(op_b) > 8 else max(len(op_a), len(op_b)) * 8
            cmp_type = "STR" if len(op_a) > 8 else "CMP"

            mutations = generate_mutations(
                op_a, op_b, cmp_size, cmp_type, bytes(buf),
                hammer=True,
                is_hash=f._cmplog.is_hash_candidate if hasattr(f._cmplog, 'is_hash_candidate') else None,
            )
            if mutations:
                offsets, replacements, enc = _rand.choice(mutations)
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
        op_a, _ = _rand.choice(pairs)
        if len(op_a) <= len(buf):
            pos = 0
            candidates = []
            while pos <= len(buf) - len(op_a):
                idx = bytes(buf).find(op_a, pos)
                if idx == -1:
                    break
                candidates.append(idx)
                pos = idx + 1
                if len(candidates) >= 5:
                    break
            if candidates:
                offset = _rand.choice(candidates)
                vb = int.from_bytes(op_a, "little") & 0xFF
                transform = _rand.choice(
                    ["xor", "arithmetic", "boundary", "hex", "toupper", "tolower"]
                )
                if transform == "xor":
                    const = _rand.randint(1, 255)
                    buf[offset] = (vb ^ const) & 0xFF
                elif transform == "arithmetic":
                    delta = _rand.randint(-128, 127)
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
        if len(buf) < 4:
            return
        p1 = random.randint(0, len(buf) - 2)
        p2 = random.randint(0, len(buf) - 2)
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
        f = self.f
        corpus = getattr(f, "corpus", [])
        if not corpus or len(buf) < 3:
            return
        other = bytes(random.choice(corpus))
        if other is data or other == buf:
            others = [c for c in corpus if c is not data]
            if not others:
                return
            other = bytes(random.choice(others))
        if len(other) < 3:
            return
        split_point = min(len(buf), len(other)) // 2
        split_point = random.randint(1, max(1, split_point))
        result = bytearray(buf[:split_point]) + other[split_point:]
        if len(result) <= f.max_len:
            buf[:] = result[: f.max_len]

    def _op_fuse_old(self, buf, _byte_idx, _data):
        """Fuse current input with a remembered block from earlier fuzz runs.

        Radamsa sed-fuse-old: maintains a ring buffer of previously
        mutated data and fuses fragments from it into the current input.
        """
        if len(buf) < 3:
            return
        cls = type(self)
        if not hasattr(cls, "_fuse_memory"):
            cls._fuse_memory = []
            cls._fuse_mem_max = 32
        mem = cls._fuse_memory
        mem.append(bytes(buf))
        if len(mem) > cls._fuse_mem_max:
            mem.pop(0)
        if len(mem) < 2:
            return
        old = random.choice(mem[:-1])
        if len(old) < 3:
            return
        p_cur = random.randint(1, len(buf) - 1)
        p_old = random.randint(1, len(old) - 1)
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

        result = lightweight_tree_mutate(bytes(buf), max_len=self.f.max_len)
        if result != bytes(buf):
            buf[:] = result[: len(buf)]

    # ── UTF-8 confusion mutations (from Radamsa) ────────────────────

    def _op_utf8_widen(self, buf, _byte_idx, _data):
        """Widen 7-bit ASCII bytes into overlong UTF-8 sequences.

        Radamsa ``sed-utf8-widen``: replaces a random ASCII byte with an
        overlong UTF-8 encoding (2-byte sequence).  Exercises UTF-8
        length-checking bugs in parsers.
        """
        if not buf:
            return
        # Find a 7-bit ASCII byte (0x00-0x7F)
        candidates = [i for i, b in enumerate(buf) if b < 0x80]
        if not candidates:
            return
        idx = random.choice(candidates)
        b = buf[idx]
        # Overlong 2-byte encoding: 110xxxxx 10xxxxxx
        buf[idx : idx + 1] = bytes([0xC0 | (b >> 6), 0x80 | (b & 0x3F)])

    def _op_utf8_insert(self, buf, _byte_idx, _data):
        """Insert problematic Unicode byte sequences.

        Radamsa ``sed-utf8-insert``: inserts curated byte sequences that
        have caused security issues: BOMs, right-to-left override,
        zero-width joiners, illegal surrogates, NFKC expansion bombs, etc.
        """
        if not buf or len(buf) >= getattr(self.f, "max_len", 65536):
            return
        from fuzzer_tool.core.mutations import _FUNNY_UNICODE  # noqa: PLC0415

        seq = random.choice(_FUNNY_UNICODE)
        pos = random.randint(0, len(buf))
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
        if not buf or len(buf) < 4:
            return
        # Split on newline
        parts = bytes(buf).split(b"\n")
        if len(parts) < 2:
            return
        mutate = random.choice(["del", "dup", "swap", "perm", "repeat", "clone"])
        if mutate == "del":
            idx = random.randint(0, len(parts) - 1)
            del parts[idx]
        elif mutate == "dup":
            idx = random.randint(0, len(parts) - 1)
            parts.insert(idx + 1, parts[idx])
        elif mutate == "swap" and len(parts) >= 2:
            idx = random.randint(0, len(parts) - 2)
            parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
        elif mutate == "perm" and len(parts) >= 3:
            # Shuffle a subset of lines
            start = random.randint(0, len(parts) - 3)
            end = min(start + random.randint(2, 6), len(parts))
            segment = parts[start:end]
            random.shuffle(segment)
            parts[start:end] = segment
        elif mutate == "repeat" and len(parts) >= 1:
            idx = random.randint(0, len(parts) - 1)
            n = random.randint(1, 32)
            for _ in range(n):
                parts.insert(idx, parts[idx])
        elif mutate == "clone" and len(parts) >= 2:
            src = random.randint(0, len(parts) - 1)
            dst = random.randint(0, len(parts) - 1)
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
        if not buf:
            return
        tbl = _COLORIZE_TBL
        # Try to use cmplog-derived colorization mask for targeted selection
        f = self.f
        cmplog_pairs = getattr(f._cmplog, "pairs", None) if hasattr(f, "_cmplog") else None
        if cmplog_pairs:
            from fuzzer_tool.core.colorizer import CmplogColorizer  # noqa: PLC0415

            colorizer = CmplogColorizer()
            mask = colorizer.colorize_from_cmplog(bytes(buf), cmplog_pairs)
            colorable = [i for i, m in enumerate(mask[: len(buf)]) if m == 0xFF]
            if colorable:
                n_mutate = max(1, min(len(colorable), len(buf) // random.randint(2, 10)))
                indices = random.choices(colorable, k=n_mutate)
                for idx in indices:
                    buf[idx] = tbl[buf[idx]]
                return
        # Fallback: random selection from entire buffer
        n_mutate = max(1, len(buf) // random.randint(2, 10))
        indices = [random.randint(0, len(buf) - 1) for _ in range(n_mutate)]
        for idx in indices:
            buf[idx] = tbl[buf[idx]]

    def _op_skipdet_probe(self, buf, _byte_idx, _data):
        """Block-flip probe to identify inert byte regions.

        AFL++ SkipDet: flip a block of bytes and check if the path changes.
        If not, those bytes are inert. This operator flips a random block
        to help discover which regions affect execution paths.
        """
        if len(buf) < 8:
            return
        # Pick a random block (10-25% of buffer) and flip all bytes
        block_size = random.randint(len(buf) // 10, len(buf) // 4)
        block_size = max(2, min(block_size, len(buf)))
        start = random.randint(0, len(buf) - block_size)
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
        if not buf:
            return
        f = self.f
        # Use dictionary tokens as "auto-extras" if available
        if f.dictionary:
            token = random.choice(f.dictionary)
            if isinstance(token, str):
                token = token.encode()
            if len(token) <= len(buf):
                pos = random.randint(0, len(buf) - len(token))
                buf[pos : pos + len(token)] = token
            return
        # Generate a short sequence of consecutive interesting bytes
        seq_len = random.randint(2, 8)
        seq = bytes(
            random.choice([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF])
            for _ in range(seq_len)
        )
        if len(buf) >= seq_len:
            pos = random.randint(0, len(buf) - seq_len)
            buf[pos : pos + seq_len] = seq

    def _op_byte_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 0xFF

    def _op_interesting_8(self, buf, byte_idx, _data):
        if buf:
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and random.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(byte_idx, k=5)
                if crash_vals:
                    buf[byte_idx] = random.choice(crash_vals) & 0xFF
                    return
            vals = INTERESTING_UNSIGNED_8 if random.random() < 0.5 else INTERESTING_8
            buf[byte_idx] = random.choice(vals) & 0xFF

    def _op_interesting_16(self, buf, _byte_idx, _data):
        if len(buf) >= 2:
            idx = random.randint(0, len(buf) - 2)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and random.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = random.choice(crash_vals)
                    fmt = "<H" if v > 32767 or v < -32768 else "<h"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = random.random() < 0.5
            vals = INTERESTING_UNSIGNED_16 if use_unsigned else INTERESTING_16
            v = random.choice(vals)
            fmt = "<H" if v > 32767 or v < -32768 else "<h"
            struct.pack_into(fmt, buf, idx, v)

    def _op_interesting_32(self, buf, _byte_idx, _data):
        if len(buf) >= 4:
            idx = random.randint(0, len(buf) - 4)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and random.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = random.choice(crash_vals)
                    fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = random.random() < 0.5
            vals = INTERESTING_UNSIGNED_32 if use_unsigned else INTERESTING_32
            v = random.choice(vals)
            fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
            struct.pack_into(fmt, buf, idx, v)

    def _op_arithmetic(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS

        width = random.choice([1, 2, 4, 8])
        if len(buf) >= width:
            max_start = len(buf) - width
            idx = (random.randint(0, max_start) // width) * width
            delta = random.choice(ARITHMETIC_DELTAS)
            if random.random() < 0.5:
                delta = -delta
            if width == 1:
                buf[idx] = (buf[idx] + delta) & 0xFF
            elif width == 2:
                le = random.random() < 0.5
                if le:
                    val = (struct.unpack_from("<H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into("<H", buf, idx, val)
                else:
                    val = (struct.unpack_from(">H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into(">H", buf, idx, val)
            elif width == 4:
                le = random.random() < 0.5
                if le:
                    val = (struct.unpack_from("<I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into("<I", buf, idx, val)
                else:
                    val = (struct.unpack_from(">I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into(">I", buf, idx, val)
            elif width == 8:
                le = random.random() < 0.5
                if le:
                    val = (struct.unpack_from("<Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into("<Q", buf, idx, val)
                else:
                    val = (struct.unpack_from(">Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into(">Q", buf, idx, val)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        if buf:
            buf[random.randint(0, len(buf) - 1)] = random.randint(0, 255)

    def _op_block_insert(self, buf, _byte_idx, _data):
        if len(buf) < self.f.max_len:
            idx = random.randint(0, len(buf))
            size = random.randint(1, min(32, self.f.max_len - len(buf)))
            buf[idx:idx] = bytes(random.randint(0, 255) for _ in range(size))

    def _op_block_delete(self, buf, _byte_idx, _data):
        if len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            max_size = min(32, len(buf) - idx, len(buf) - 1)
            if max_size >= 1:
                del buf[idx : idx + random.randint(1, max_size)]

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        if len(buf) < 2 or len(buf) >= self.f.max_len:
            return
        idx = random.randint(0, len(buf) - 1)
        size = random.randint(1, min(16, len(buf) - idx))
        block = buf[idx : idx + size]
        ins = random.randint(0, len(buf))
        buf[ins:ins] = block

    def _op_dict_insert(self, buf, _byte_idx, _data):
        if self.f.dictionary:
            token = random.choice(self.f.dictionary)
            if len(buf) + len(token) <= self.f.max_len:
                buf[random.randint(0, len(buf)) : 0] = token

    def _op_dict_replace(self, buf, _byte_idx, _data):
        if self.f.dictionary and buf:
            token = random.choice(self.f.dictionary)
            idx = random.randint(0, len(buf) - 1)
            end = min(idx + len(token), len(buf))
            buf[idx:end] = token[: end - idx]

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        if self.f.dictionary:
            return bytearray(random.choice(self.f.dictionary)[: self.f.max_len])

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        if self.f.dictionary:
            token = random.choice(self.f.dictionary)
            if len(buf) + len(token) <= self.f.max_len:
                return bytearray(token) + buf

    def _op_dict_append(self, buf, _byte_idx, _data):
        if self.f.dictionary:
            token = random.choice(self.f.dictionary)
            if len(buf) + len(token) <= self.f.max_len:
                buf.extend(token)

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        import zlib

        if buf and len(buf) >= 4:
            pos = random.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = zlib.crc32(bytes(buf[:pos])).to_bytes(4, "big")

    def _op_token_dup(self, buf, _byte_idx, _data):
        if self.f.dictionary and buf:
            token = random.choice(self.f.dictionary)
            if len(buf) + len(token) <= self.f.max_len:
                buf[random.randint(0, len(buf)) : 0] = token

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        if buf:
            idx = random.randint(0, len(buf) - 1)
            ctx = (
                bytes(buf[max(0, idx - self.f.markov.order) : idx]) if self.f.markov.order else b""
            )
            buf[idx] = self.f.markov.sample_byte(ctx)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        if self.f.mc and self.f.mc.cem_fitted:
            if buf:
                buf[random.randint(0, len(buf) - 1)] = self.f.mc.cem_byte(
                    random.randint(0, len(buf) - 1)
                )
            else:
                return bytearray(self.f.mc.cem_sample(random.randint(1, min(32, self.f.max_len))))

    def _op_splice(self, buf, _byte_idx, data):
        if len(self.f.corpus) >= 2:
            a = random.choice(self.f.corpus)
            b = random.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(splice(a, b)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(splice(bytes(buf), random.choice(others))[: self.f.max_len])

    def _op_crossover(self, buf, _byte_idx, data):
        from fuzzer_tool.core.mutations import crossover

        if len(self.f.corpus) >= 2 and buf:
            a = random.choice(self.f.corpus)
            b = random.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(crossover(a, b)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(crossover(bytes(buf), random.choice(others))[: self.f.max_len])

    def _op_type_replace(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import type_replace

        if buf:
            return bytearray(type_replace(bytes(buf))[: self.f.max_len])

    def _op_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ascii_num_replace

        if buf:
            return bytearray(ascii_num_replace(bytes(buf))[: self.f.max_len])

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_shuffle

        if buf and len(buf) > 1:
            return bytearray(byte_shuffle(bytes(buf))[: self.f.max_len])

    def _op_byte_delete(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_delete

        if buf and len(buf) > 1:
            return bytearray(byte_delete(bytes(buf))[: self.f.max_len])

    def _op_byte_insert(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_insert

        if buf and len(buf) < self.f.max_len:
            return bytearray(byte_insert(bytes(buf), self.f.max_len)[: self.f.max_len])

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import insert_ascii_num

        if buf and len(buf) < self.f.max_len:
            return bytearray(insert_ascii_num(bytes(buf), self.f.max_len)[: self.f.max_len])

    def _op_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 2:
            return bytearray(transpose_bytes(bytes(buf), 2)[: self.f.max_len])

    def _op_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 4:
            return bytearray(transpose_bytes(bytes(buf), 4)[: self.f.max_len])

    def _op_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 8:
            return bytearray(transpose_bytes(bytes(buf), 8)[: self.f.max_len])

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if buf:
            return bytearray(bit_transpose(bytes(buf), 1)[: self.f.max_len])

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 2:
            return bytearray(bit_transpose(bytes(buf), 2)[: self.f.max_len])

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 4:
            return bytearray(bit_transpose(bytes(buf), 4)[: self.f.max_len])

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 8:
            return bytearray(bit_transpose(bytes(buf), 8)[: self.f.max_len])

    def _op_length_grow(self, buf, _byte_idx, _data):
        if buf and len(buf) < self.f.max_len:
            size = random.randint(1, min(64, self.f.max_len - len(buf)))
            if size > 0:
                buf.extend(random.randint(0, 255) for _ in range(size))

    def _op_length_shrink(self, buf, _byte_idx, _data):
        if len(buf) > 2:
            del buf[random.randint(1, len(buf) - 1) :]

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        if buf and len(buf) < self.f.max_len:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = idx + size
            if ins <= len(buf) and len(buf) + len(block) <= self.f.max_len:
                buf[ins:ins] = block

    def _op_truncate(self, buf, _byte_idx, _data):
        if len(buf) > 2:
            del buf[random.randint(2, len(buf)) :]

    def _op_length_boundary(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import LENGTH_BOUNDARIES

        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.randint(1, 32)))
            return
        # 30% chance: bias toward lengths that historically discovered edges
        if hasattr(self.f, "_length_tracker") and self.f._length_tracker and random.random() < 0.3:
            recs = self.f._length_tracker.recommended_lengths(k=5)
            if recs:
                target_len = random.choice(recs)
            else:
                target_len = random.choices(
                    LENGTH_BOUNDARIES,
                    weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
                )[0]
        else:
            target_len = random.choices(
                LENGTH_BOUNDARIES,
                weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
            )[0]
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
        if len(buf) >= 4:
            i = random.randint(0, len(buf) - 3)
            j = random.randint(i + 2, len(buf) - 1)
            size = random.randint(1, min(j - i, 16))
            a, b = buf[i : i + size], buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        if len(buf) >= 2:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        if buf:
            width = random.choice([2, 4, 8])
            if len(buf) >= width:
                idx = random.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        if self.f.grammar:
            return bytearray(
                self.f.grammar.mutate(bytes(buf), max_len=self.f.max_len)[: self.f.max_len]
            )

    def _op_grammar_tree_mutate(self, buf, _byte_idx, _data):
        if self.f.grammar:
            from fuzzer_tool.core.grammar import TreeMutator

            if not hasattr(self.f, "_tree_mutator"):
                self.f._tree_mutator = TreeMutator(self.f.grammar)
            tree = self.f._tree_mutator.parse(bytes(buf))
            return bytearray(
                self.f._tree_mutator.mutate_tree(tree, max_len=self.f.max_len)[: self.f.max_len]
            )

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.png_mutations import PngChunkMutator, parse_png_chunks

        if not hasattr(self.f, "_png_mutator"):
            self.f._png_mutator = PngChunkMutator()
        # Set WFC mode and SMT solver from fuzzer state
        self.f._png_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        self.f._png_mutator.smt_solver = getattr(self.f, "_smt_solver", None)
        if parse_png_chunks(bytes(buf)):
            mutated = self.f._png_mutator.mutate(bytes(buf), max_len=self.f.max_len)
        else:
            mutated = self.f._png_mutator._generate_random_png(self.f.max_len)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.jpeg_mutations import JpegMutator, parse_jpeg_markers

        if not hasattr(self.f, "_jpeg_mutator"):
            self.f._jpeg_mutator = JpegMutator()
        if parse_jpeg_markers(bytes(buf)):
            mutated = self.f._jpeg_mutator.mutate(bytes(buf), max_len=self.f.max_len)
        else:
            mutated = self.f._jpeg_mutator._generate_random_jpeg(max_len=self.f.max_len)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.jpeg_mutations import (
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
                    idx = random.choice(candidates)
                    marker = markers[idx]
                    data = bytearray(marker.data)
                    for _ in range(random.randint(1, min(4, len(data)))):
                        data[random.randint(0, len(data) - 1)] ^= 1 << random.randint(0, 7)
                    marker.data = bytes(data)
                    return bytearray(serialize_jpeg_markers(markers)[: self.f.max_len])

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.gzip_mutations import GzipMutator, parse_gzip

        if not hasattr(self.f, "_gzip_mutator"):
            self.f._gzip_mutator = GzipMutator()
        if parse_gzip(bytes(buf)):
            mutated = self.f._gzip_mutator.mutate(bytes(buf), max_len=self.f.max_len)
        else:
            mutated = self.f._gzip_mutator._generate_random_gzip(max_len=self.f.max_len)
        return bytearray(mutated[: self.f.max_len])

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.bmp_mutations import BmpMutator, parse_bmp

        if not hasattr(self.f, "_bmp_mutator"):
            self.f._bmp_mutator = BmpMutator()
        self.f._bmp_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        if parse_bmp(bytes(buf)):
            mutated = self.f._bmp_mutator.mutate(bytes(buf), max_len=self.f.max_len)
        else:
            mutated = self.f._bmp_mutator._generate_random_bmp(max_len=self.f.max_len)
        return bytearray(mutated[: self.f.max_len])

    def _op_zlib_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.zlib_mutations import ZlibMutator, parse_zlib

        if not hasattr(self.f, "_zlib_mutator"):
            self.f._zlib_mutator = ZlibMutator()
        if parse_zlib(bytes(buf)):
            mutated = self.f._zlib_mutator.mutate(bytes(buf), max_len=self.f.max_len)
        else:
            mutated = self.f._zlib_mutator._generate_random_zlib(max_len=self.f.max_len)
        return bytearray(mutated[: self.f.max_len])

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.png_mutations import parse_png_chunks, serialize_png_chunks

        if buf:
            chunks = parse_png_chunks(bytes(buf))
            if chunks and len(chunks) > 1:
                candidates = [i for i, c in enumerate(chunks) if c.chunk_type != b"IEND"]
                if candidates:
                    idx = random.choice(candidates)
                    chunk = chunks[idx]
                    if chunk.data:
                        data = bytearray(chunk.data)
                        for _ in range(random.randint(1, min(4, len(data)))):
                            data[random.randint(0, len(data) - 1)] ^= 1 << random.randint(0, 7)
                        chunk.data = bytes(data)
                    else:
                        chunk.data = bytes(
                            random.randint(0, 255) for _ in range(random.randint(1, 32))
                        )
                    return bytearray(serialize_png_chunks(chunks)[: self.f.max_len])

    def _op_redqueen(self, buf, _byte_idx, data):
        parent_meta = self.f.seed_meta.get(data)
        if not (buf and parent_meta):
            return
        matches = parent_meta.get("redqueen_matches", [])
        offsets = parent_meta.get("redqueen_offsets", [])
        if matches:
            for _ in range(random.randint(1, min(4, len(matches)))):
                off, op_a, op_b = random.choice(matches)
                end = off + len(op_a)
                if end <= len(buf) and bytes(buf[off:end]) == op_a:
                    for j, b_val in enumerate(op_b):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets and self.f._cmplog and self.f._cmplog.tokens:
            for _ in range(random.randint(1, min(4, len(offsets)))):
                off = random.choice(offsets)
                if off < len(buf):
                    token = random.choice(self.f._cmplog.tokens)
                    for j, b_val in enumerate(token):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets:
            for _ in range(random.randint(1, min(4, len(offsets)))):
                off = random.choice(offsets)
                if off < len(buf):
                    buf[off] ^= 0xFF

    def _op_havoc(self, buf, _byte_idx, data):
        return bytes(self.havoc_mutate(buf))

    # ── Dispatch table: op name → handler method ───────────────────────
    def build_dispatch(self):
        return {
            "bit_flip": self._op_bit_flip,
            "bit_offset_flip": self._op_bit_offset_flip,
            "bit_offset_span": self._op_bit_offset_span,
            "simd_boundary": self._op_simd_boundary,
            "regex_bomb": self._op_regex_bomb,
            "clone_fixed": self._op_clone_fixed,
            "overwrite_copy": self._op_overwrite_copy,
            "overwrite_fixed": self._op_overwrite_fixed,
            "redqueen_xform": self._op_redqueen_xform,
            "fuse_this": self._op_fuse_this,
            "fuse_next": self._op_fuse_next,
            "fuse_old": self._op_fuse_old,
            "tree_mutate": self._op_tree_mutate,
            "utf8_widen": self._op_utf8_widen,
            "utf8_insert": self._op_utf8_insert,
            "line_mutate": self._op_line_mutate,
            "colorization": self._op_colorization,
            "skipdet_probe": self._op_skipdet_probe,
            "auto_extras": self._op_auto_extras,
            "byte_flip": self._op_byte_flip,
            "interesting_8": self._op_interesting_8,
            "interesting_16": self._op_interesting_16,
            "interesting_32": self._op_interesting_32,
            "arithmetic": self._op_arithmetic,
            "random_bytes": self._op_random_bytes,
            "block_insert": self._op_block_insert,
            "block_delete": self._op_block_delete,
            "block_duplicate": self._op_block_duplicate,
            "dict_insert": self._op_dict_insert,
            "dict_replace": self._op_dict_replace,
            "dict_overwrite": self._op_dict_overwrite,
            "dict_prepend": self._op_dict_prepend,
            "dict_append": self._op_dict_append,
            "checksum_repair": self._op_checksum_repair,
            "token_dup": self._op_token_dup,
            "markov_bytes": self._op_markov_bytes,
            "cem_bytes": self._op_cem_bytes,
            "splice": self._op_splice,
            "crossover": self._op_crossover,
            "type_replace": self._op_type_replace,
            "ascii_num": self._op_ascii_num,
            "byte_shuffle": self._op_byte_shuffle,
            "byte_delete": self._op_byte_delete,
            "byte_insert": self._op_byte_insert,
            "insert_ascii_num": self._op_insert_ascii_num,
            "transpose_16": self._op_transpose_16,
            "transpose_32": self._op_transpose_32,
            "transpose_64": self._op_transpose_64,
            "bit_transpose_8": self._op_bit_transpose_8,
            "bit_transpose_16": self._op_bit_transpose_16,
            "bit_transpose_32": self._op_bit_transpose_32,
            "bit_transpose_64": self._op_bit_transpose_64,
            "length_grow": self._op_length_grow,
            "length_shrink": self._op_length_shrink,
            "repeat_clone": self._op_repeat_clone,
            "truncate": self._op_truncate,
            "length_boundary": self._op_length_boundary,
            "swap_regions": self._op_swap_regions,
            "swap_bytes": self._op_swap_bytes,
            "endianness_swap": self._op_endianness_swap,
            "grammar_mutate": self._op_grammar_mutate,
            "grammar_tree_mutate": self._op_grammar_tree_mutate,
            "png_chunk_mutate": self._op_png_chunk_mutate,
            "jpeg_chunk_mutate": self._op_jpeg_chunk_mutate,
            "jpeg_crc_fix": self._op_jpeg_crc_fix,
            "gzip_chunk_mutate": self._op_gzip_chunk_mutate,
            "bmp_chunk_mutate": self._op_bmp_chunk_mutate,
            "zlib_chunk_mutate": self._op_zlib_chunk_mutate,
            "png_crc_fix": self._op_png_crc_fix,
            "redqueen": self._op_redqueen,
            "havoc": self._op_havoc,
        }

    def havoc_mutate(self, buf: bytearray) -> bytearray:
        for _ in range(random.randint(2, 8)):
            self._apply_single_mutation(buf)
        return buf

    def _apply_single_mutation(self, buf: bytearray):
        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.randint(1, 16)))
            return
        op = random.randint(0, 10)
        if op == 0:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] ^= 1 << random.randint(0, 7)
        elif op == 1:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.randint(0, 255)
        elif op == 2 and len(buf) > 1:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.f.max_len:
            idx = random.randint(0, len(buf))
            buf.insert(idx, random.randint(0, 255))
        elif op == 4 and len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(len(buf) - 1, len(buf) - idx))
            del buf[idx : idx + size]
        elif op == 5 and len(buf) >= 4:
            import zlib

            pos = random.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = zlib.crc32(bytes(buf[:pos])).to_bytes(4, "big")
        elif op == 6 and len(buf) >= 2:
            i = random.randint(0, len(buf) - 2)
            j = random.randint(i + 1, len(buf) - 1)
            size = random.randint(1, min(j - i, 8))
            a = buf[i : i + size]
            b = buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a
        elif op == 7 and buf:
            width = random.choice([2, 4])
            if len(buf) >= width:
                idx = random.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")
        elif op == 8 and buf:
            # Byte insert
            if len(buf) < self.f.max_len:
                idx = random.randint(0, len(buf))
                buf.insert(idx, random.randint(0, 255))
        elif op == 9 and buf:
            # Random byte set
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.randint(0, 255)
        elif op == 10 and len(buf) >= 2:
            # Shuffle a short range
            start = random.randint(0, len(buf) - 2)
            end = min(start + random.randint(2, 8), len(buf))
            region = buf[start:end]
            random.shuffle(region)
            buf[start:end] = region

    # ── Operator selection logic ───────────────────────────────────────

    def build_ops(self, data: bytes) -> list[str]:
        """Build the list of available mutation operators from ground truth."""
        f = self.f
        ops = list(MUTATIONS)
        if not getattr(f, "enable_regex_bomb", False):
            ops = [op for op in ops if op != "regex_bomb"]
        if f.dictionary:
            ops.extend(DICT_MUTATIONS)
        if f.markov_trained:
            ops.append("markov_bytes")
        if f.mc and f.mc_cem and f.mc.cem_fitted:
            ops.append("cem_bytes")
        if f.grammar:
            ops.append("grammar_mutate")
            ops.append("grammar_tree_mutate")
        if getattr(f, "_cmplog", None) and f._cmplog.pairs:
            ops.append("redqueen_xform")
        ops.extend(FORMAT_MUTATIONS)
        parent_meta = f.seed_meta.get(data)
        if parent_meta and (
            parent_meta.get("redqueen_matches") or parent_meta.get("redqueen_offsets")
        ):
            ops.append("redqueen")
        return ops

    def select_op(self, ops: list[str]) -> str:
        """Select a mutation operator using the active scheduling strategy."""
        f = self.f

        if f._stall_recovery_active:
            f._meta_strategy = "random_stall"
            return random.choice(ops)

        available = []
        if f._use_replicator and f._replicator:
            available.append("replicator")
        if f.mc and f.mc_bandit:
            available.append("bandit")
        if f._use_mopt and f._mopt:
            available.append("mopt")

        if f._use_elo and f._elo and len(available) >= 2:
            strategy = f._elo.select_strategy(available)
            f._meta_strategy = strategy
        elif f._use_elo and f._elo and available:
            strategy = available[0]
            f._meta_strategy = strategy
        else:
            strategy = None

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
        else:
            op = random.choice(ops)
            f._last_mopt_particles.append(None)
        return op

    def select_position(self, buf: bytearray, data: bytes) -> int:
        """Select a byte position for mutation using MI/TE/sensitivity/crash-MI/random."""
        f = self.f
        if not buf:
            return 0
        te_pos = (
            f._get_te_weighted_position(len(buf)) if f._use_transfer_entropy and f._te else None
        )
        mi_pos = f._mi.weighted_position(len(buf)) if f._use_mi and f._mi else None
        sens_pos = f._sensitivity.get_weighted_position(data, len(buf))
        crash_mi_pos = None
        if f._crash_mi and f._crash_mi.total_execs >= f._crash_mi.min_observations:
            crash_mi_pos = f._crash_mi.weighted_position(len(buf))
        candidates = [p for p in [sens_pos, te_pos, mi_pos, crash_mi_pos] if p is not None]
        if candidates:
            return random.choice(candidates)
        return random.randint(0, len(buf) - 1)

    # ── Main mutation orchestrator ─────────────────────────────────────

    def mutate(self, data: bytes) -> bytes:
        from fuzzer_tool.core.similarity import hamming_distance

        f = self.f
        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * random.randint(1, 32))

        ops = self.build_ops(data)
        f._last_ops_used = []
        f._last_mopt_particles = []
        if not hasattr(f, "_prev_bandit_op"):
            f._prev_bandit_op = None
        f._meta_strategy = None

        n_mutations = f.mutations_per_input
        if f._stall_recovery_active:
            n_mutations = max(n_mutations, 16)

        for _ in range(n_mutations):
            op = self.select_op(ops)
            f._last_ops_used.append(op)

            byte_idx = self.select_position(buf, data)
            f._last_mutation_offset = byte_idx
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
                buf = bytearray(result[: f.max_len])

        if f._frameshift.relations:
            f._frameshift.apply_to_buffer(buf)

        result = bytes(buf)
        f._last_hamming_distance = (
            hamming_distance(data, result) if len(data) == len(result) else -1
        )
        return result
