"""Arithmetic constraint solving for cmplog pairs and PNG computed-field helpers.

Three modulo solving modes selectable via ``--mod-solving``:

- **heuristic**: Try common divisors on ``(remainder, 0)`` pairs (default, no extra infra).
- **trace**: Use PC-correlated DIV/IDIV from static analysis for precise divisors.
- **concolic**: Full constraint model with z3 solver for compound constraints.

Module name preserved for backward compatibility with ``--enable-smt-z3``.
"""

from __future__ import annotations

import logging
import struct
import zlib

log = logging.getLogger(__name__)

_SOLVER_TIMEOUT_MS = 50
_CACHE_MAXSIZE = 1024

# Max delta/mask to consider a relation plausible, keyed by operand width (bytes).
_MAX_DELTA_FOR_WIDTH: dict[int, int] = {1: 256, 2: 65536, 4: 65536, 8: 65536}

# Common divisors to try in heuristic modulo mode.
_COMMON_DIVISORS = [2, 3, 4, 5, 7, 8, 10, 12, 16, 32, 64, 100, 128, 256, 1000]
# In heuristic mode, only try modulo when val_a (the supposed remainder) < this.
_HEURISTIC_MAX_REMAINDER = 256


def _z3_available() -> bool:
    """Check if z3-solver is installed without importing at module level."""
    try:
        import z3  # noqa: F401

        return True
    except ImportError:
        return False


# ── Concolic trace accumulator ──────────────────────────────────────────


class ConcolicTrace:
    """Collects all cmplog pairs from one run for whole-execution constraint solving.

    Stores each CMP entry with operand bytes, width, and optional PC. The
    ``ConcolicEngine`` builds a z3 constraint model from this trace and solves
    for a satisfying input mutation.
    """

    def __init__(self):
        self.entries: list[dict] = []
        self._input_bytes: bytes | None = None

    def set_input(self, data: bytes):
        self._input_bytes = data

    def add_entry(self, op_a: bytes, op_b: bytes,
                  width: int, pc: int | None = None):
        self.entries.append({
            "op_a": op_a,
            "op_b": op_b,
            "width": width,
            "pc": pc,
        })

    def clear(self):
        self.entries.clear()
        self._input_bytes = None

    def has_entries(self) -> bool:
        return len(self.entries) > 0

    def solve(self, timeout_ms: int = _SOLVER_TIMEOUT_MS) -> bytes | None:
        """Build a z3 constraint model from all entries and solve.

        For each entry where one operand appears as a literal byte
        sequence in the input, we infer the relevant offset and add a
        constraint ``input[offset:offset+len] == target_value``.

        Constrains non-overridden bytes to their original values so the
        solution is a minimal mutation, not a wholesale replacement.
        Original-value constraints are skipped for byte positions that
        receive a target override, avoiding contradictions.

        Returns a mutated input if z3 finds a satisfying assignment,
        or ``None`` if the trace is empty or no solution was found.
        """
        if not self.entries or self._input_bytes is None:
            return None
        import z3

        data = self._input_bytes
        z3.set_param("timeout", timeout_ms)

        solver = z3.Solver()
        vars_ = [z3.BitVec(f"b{i}", 8) for i in range(len(data))]
        overridden: set[int] = set()

        for entry in self.entries:
            op_a, op_b = entry["op_a"], entry["op_b"]
            width = entry["width"]
            for candidate, target in [(op_a, op_b), (op_b, op_a)]:
                idx = data.find(candidate)
                if idx != -1 and len(candidate) == width:
                    for j in range(width):
                        solver.add(
                            vars_[idx + j] == z3.BitVecVal(target[j], 8)
                        )
                        overridden.add(idx + j)
                    break

        # Constrain non-overridden bytes to original values
        for i, v in enumerate(vars_):
            if i not in overridden:
                solver.add(v == z3.BitVecVal(data[i], 8))

        if not overridden:
            return None

        if solver.check() == z3.sat:
            model = solver.model()
            result = bytearray(len(data))
            for i in range(len(data)):
                result[i] = model.eval(vars_[i]).as_long()
            return bytes(result)
        return None


# ── PC→divisor map for trace mode ──────────────────────────────────────

# Module-level cache: {pc_address: divisor_value}
# Populated by elf.py's extract_div_constants(), read by Z3Solver.
PC_DIVISOR_MAP: dict[int, int] = {}

# CMP PCs that check a DIV remainder whose divisor is a runtime variable
# (not statically resolvable).  For these, the solver falls back to the
# heuristic common-divisor set instead of skipping the pair entirely.
PC_WEAK_MOD_SET: set[int] = set()


def set_pc_divisor_map(mapping: dict[int, int]):
    """Set the PC→divisor map from static analysis."""
    PC_DIVISOR_MAP.clear()
    PC_DIVISOR_MAP.update(mapping)


def set_weak_mod_set(weak: set[int]):
    """Set the weak modulus PC set from static analysis."""
    PC_WEAK_MOD_SET.clear()
    PC_WEAK_MOD_SET.update(weak)


# ── Solver ──────────────────────────────────────────────────────────────


class Z3Solver:
    """Arithmetic constraint solver for cmplog operand pairs.

    Three modes controlled by ``mod_solving_mode``:

    - ``heuristic``: Direct ADD/XOR/SUB arithmetic + heuristic modulo detection
    - ``trace``: + PC-correlated DIV/IDIV divisor map from static analysis
    - ``concolic``: + whole-execution z3 constraint model

    Results are in the same format as cmplog redqueen_matches, so they can
    be fed into the existing operator pipeline.
    """

    def __init__(self, timeout_ms: int = _SOLVER_TIMEOUT_MS,
                 mod_solving_mode: str = "heuristic"):
        self.timeout_ms = timeout_ms
        self.mod_solving_mode = mod_solving_mode
        self.queries_attempted = 0
        self.queries_solved = 0
        self.queries_failed = 0
        self.batch_attempted = 0
        self.batch_solved = 0
        self.cache_hits = 0
        self._available = _z3_available()
        if not self._available:
            log.info("z3-solver not installed — Z3-dependent modes (concolic) disabled")

        # Concolic trace accumulator
        self.concolic_trace = ConcolicTrace() if mod_solving_mode == "concolic" else None

        # LRU cache
        self._cache: dict[tuple[bytes, bytes], dict | None] = {}
        self._cache_order: list[tuple[bytes, bytes]] = []
        self._cache_maxsize = _CACHE_MAXSIZE

    # ── Public API ──────────────────────────────────────────────────────

    def solve_cmplog_pair(self, op_a: bytes, op_b: bytes,
                          pc: int | None = None) -> dict | None:
        """Interpret a cmplog operand pair as an arithmetic constraint.

        In concolic mode, also records the pair in the trace accumulator.
        In trace mode, uses the optional PC to look up the divisor from
        static analysis.

        Checks the LRU cache first. On miss, tries arithmetic relations
        between *op_a* and *op_b* for widths 1, 2, 4, 8 bytes (little-endian).

        Returns a dict with keys ``solved_bytes``, ``width``,
        ``relation``, ``delta``, or *None* when no relation is found.
        """
        if not self._available:
            return None

        # Concolic mode: accumulate, then solve in batch
        if self.mod_solving_mode == "concolic" and self.concolic_trace is not None:
            for width in (8, 4, 2, 1):
                if len(op_a) == width and len(op_b) == width:
                    self.concolic_trace.add_entry(op_a, op_b, width, pc)
                    break
            return None  # concolic mode returns None per-pair; batch solve later

        key = (op_a, op_b)
        cached = self._cache.get(key)
        if cached is not None or key in self._cache:
            self.cache_hits += 1
            return cached

        for width in (8, 4, 2, 1):
            if len(op_a) == width and len(op_b) == width:
                val_a = int.from_bytes(op_a, "little")
                val_b = int.from_bytes(op_b, "little")
                if val_a == val_b:
                    continue
                result = self._solve_arithmetic(width, val_a, val_b, pc)
                if result is not None:
                    self._cache_set(key, result)
                    return result

        self._cache_set(key, None)
        return None

    def solve_concolic(self, input_data: bytes) -> bytes | None:
        """Run the concolic solver on the accumulated trace.

        Only meaningful in ``concolic`` mode. Returns a mutated input
        or ``None`` if the trace is empty or no solution was found.
        """
        if self.mod_solving_mode != "concolic" or self.concolic_trace is None:
            return None
        self.concolic_trace.set_input(input_data)
        result = self.concolic_trace.solve(timeout_ms=self.timeout_ms)
        self.concolic_trace.clear()
        return result

    def reset_batch(self):
        """Reset per-iteration counters. Called between fuzz iterations."""
        self.batch_attempted = 0
        self.batch_solved = 0

    @property
    def stats(self) -> dict:
        return {
            "queries_attempted": self.queries_attempted,
            "queries_solved": self.queries_solved,
            "queries_failed": self.queries_failed,
        }

    # ── PNG computed-field helpers ─────────────────────────────────────

    @staticmethod
    def solve_png_length(chunk_data: bytes) -> bytes:
        """4-byte big-endian length prefix for a PNG chunk."""
        return struct.pack(">I", len(chunk_data))

    @staticmethod
    def solve_png_crc(chunk_type: bytes, chunk_data: bytes) -> bytes:
        """4-byte CRC32 for a PNG chunk (covers type + data)."""
        crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        return struct.pack(">I", crc)

    def solve_png_chunk_fields(self, chunk_type: bytes, chunk_data: bytes) -> dict:
        """Length prefix + CRC for a PNG chunk."""
        return {
            "length": self.solve_png_length(chunk_data),
            "crc": self.solve_png_crc(chunk_type, chunk_data),
        }

    # ── Core solving ────────────────────────────────────────────────────

    def _solve_arithmetic(self, width: int, val_a: int, val_b: int,
                          pc: int | None = None) -> dict | None:
        """Try common arithmetic relations between *val_a* and *val_b*.

        Uses direct modular arithmetic for ADD, SUB, XOR, and heuristic
        MOD relations. In trace mode, also checks the PC→divisor map.

        Returns a dict with keys ``solved_bytes``, ``width``, ``relation``,
        ``delta``, or *None* when no relation is found.
        """
        self.queries_attempted += 1
        self.batch_attempted += 1
        w = width * 8
        mask = (1 << w) - 1
        max_delta = _MAX_DELTA_FOR_WIDTH.get(width, 65536)

        # ── ADD ──
        delta = (val_b - val_a) & mask
        if 0 < delta < max_delta:
            self.queries_solved += 1
            self.batch_solved += 1
            return {
                "solved_bytes": val_b.to_bytes(width, "little"),
                "width": width, "relation": "add", "delta": delta,
            }

        # ── XOR ──
        xmask = val_a ^ val_b
        if 0 < xmask < max_delta:
            self.queries_solved += 1
            self.batch_solved += 1
            return {
                "solved_bytes": val_b.to_bytes(width, "little"),
                "width": width, "relation": "xor", "delta": xmask,
            }

        # ── SUB ──
        sub_delta = (val_a - val_b) & mask
        if 0 < sub_delta < max_delta:
            self.queries_solved += 1
            self.batch_solved += 1
            return {
                "solved_bytes": val_a.to_bytes(width, "little"),
                "width": width, "relation": "sub", "delta": sub_delta,
            }

        # ── MOD: heuristic mode (A) ──
        if self.mod_solving_mode in ("heuristic", "trace"):
            mod_result = self._try_mod_heuristic(width, val_a, val_b)
            if mod_result is not None:
                return mod_result

        # ── MOD: trace mode (B) — PC-correlated divisor ──
        if self.mod_solving_mode == "trace" and pc is not None:
            divisor = PC_DIVISOR_MAP.get(pc)
            if divisor is not None and width <= 8:
                mod_result = self._try_mod_with_divisor(width, val_a, val_b, divisor)
                if mod_result is not None:
                    return mod_result
            # If the PC is in the weak modulus set, the divisor is a runtime
            # variable — fall back to the heuristic common-divisor set.
            if pc in PC_WEAK_MOD_SET:
                mod_result = self._try_mod_heuristic(width, val_a, val_b)
                if mod_result is not None:
                    return mod_result

        self.queries_failed += 1
        return None

    # ── MOD heuristics ───────────────────────────────────────────────────

    def _try_mod_heuristic(self, width: int, val_a: int, val_b: int) -> dict | None:
        """Heuristic modulo detection: try common divisors.

        Looks for the pattern ``cmp(x % N, expected)`` where cmplog
        captured ``(remainder, expected)``.  Most commonly ``expected``
        is 0 (a modulo-equality check).
        """
        # The remainder is usually val_a (the computed value),
        # and val_b is the comparison target (often 0).
        if val_b != 0 and val_a > _HEURISTIC_MAX_REMAINDER:
            return None

        remainder = val_a if val_b == 0 else val_b
        expected = val_b if val_b == 0 else val_a

        if remainder >= _HEURISTIC_MAX_REMAINDER:
            return None

        for d in _COMMON_DIVISORS:
            if remainder % d == 0:
                solved_val = expected.to_bytes(width, "little")
                self.queries_solved += 1
                self.batch_solved += 1
                return {
                    "solved_bytes": solved_val,
                    "width": width, "relation": "mod", "delta": d,
                }
        return None

    def _try_mod_with_divisor(self, width: int, val_a: int, val_b: int,
                              divisor: int) -> dict | None:
        """Modulo solving with a known divisor from static analysis.

        The CMP at this PC is known to compare ``x % divisor`` against
        ``val_b``.  We solve for a value that makes the modulo produce
        ``val_b``, then return it as the I2S replacement.
        """
        if divisor <= 0 or divisor > (1 << (width * 8)):
            return None
        # The comparison is: (result = x % divisor) == val_b.
        # Any value of the form (val_b + k * divisor) is a solution
        # (for small k we stay in the same byte width).
        for k in range(10):
            candidate = val_b + k * divisor
            if candidate < (1 << (width * 8)):
                self.queries_solved += 1
                self.batch_solved += 1
                return {
                    "solved_bytes": candidate.to_bytes(width, "little"),
                    "width": width, "relation": "mod", "delta": divisor,
                }
        # Fallback: the smallest non-negative solution
        if val_b < (1 << (width * 8)):
            self.queries_solved += 1
            self.batch_solved += 1
            return {
                "solved_bytes": val_b.to_bytes(width, "little"),
                "width": width, "relation": "mod", "delta": divisor,
            }
        return None

    # ── Cache helpers ───────────────────────────────────────────────────

    def _cache_set(self, key: tuple[bytes, bytes], value: dict | None):
        """Insert into LRU cache, evicting oldest entry at capacity."""
        if len(self._cache) >= self._cache_maxsize:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = value
        self._cache_order.append(key)
