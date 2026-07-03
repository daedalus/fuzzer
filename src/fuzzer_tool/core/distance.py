"""AFLGo-style directed distance computation for targeted fuzzing.

Computes call-graph distance from every basic block to a set of target
functions. During fuzzing, seeds whose execution traces have lower
average distance-to-target are prioritized.

Distance computation:
  1. Parse ELF symbol table to extract function addresses
  2. Build an approximate call graph from instruction-level analysis:
     - CALL instructions reference target addresses
     - Distance = shortest path from entry to target function
  3. Map basic block addresses to their containing function
  4. Per-seed average distance = mean distance of all basic blocks hit

The distance signal integrates into scheduling via exponential weighting,
annealed over time from "maximize coverage" to "minimize distance."
"""

import logging
import re
import struct
from pathlib import Path

log = logging.getLogger(__name__)

# Call instruction patterns (x86_64)
# REL32 call: E8 xx xx xx xx (5 bytes)
# INDIRECT call: FF 15 (call [rip+disp32]) — we skip these
_CALL_RE = re.compile(rb"\xe8")  # REL32 call opcode


class TargetDistance:
    """Compute call-graph distances from basic blocks to target functions.

    Args:
        target: Path to the ELF binary.
        targets: List of target function names or addresses (hex strings).
    """

    def __init__(self, target: str, targets: list[str] | None = None):
        self.target = target
        self.target_names: list[str] = targets or []
        self.target_addrs: set[int] = set()

        # Function table: name -> (start_addr, end_addr)
        self.functions: dict[str, tuple[int, int]] = {}
        # Reverse map: address -> function name
        self.addr_to_func: dict[int, str] = {}
        # Call graph: func_name -> set of func_names it calls
        self.call_graph: dict[str, set[str]] = {}
        # Distance cache: func_name -> distance from entry
        self._distances: dict[str, float] = {}
        # BB -> distance cache
        self._bb_distances: dict[int, float] = {}

        self._loaded = False
        self._entry_addr: int = 0
        self._text_start: int = 0
        self._text_end: int = 0
        self._base_addr: int = 0

    def load(self) -> bool:
        """Parse the ELF and compute distances. Returns True on success."""
        try:
            elf_data = Path(self.target).read_bytes()
        except OSError as e:
            log.warning("Cannot read target ELF: %s", e)
            return False

        if len(elf_data) < 64 or elf_data[:4] != b"\x7fELF":
            log.warning("Not an ELF file: %s", self.target)
            return False

        if not self._parse_symbols(elf_data):
            return False
        self._resolve_targets()
        self._build_call_graph(elf_data)
        self._compute_distances()

        self._loaded = True
        log.info(
            "TargetDistance: %d functions, %d targets, entry=0x%x",
            len(self.functions), len(self.target_addrs), self._entry_addr,
        )
        return True

    def _parse_symbols(self, elf_data: bytes) -> bool:
        """Extract function symbols from the ELF symbol table."""
        if elf_data[4] != 2 or elf_data[5] != 1:  # ELF64, little-endian
            log.debug("Only ELF64 little-endian supported")
            return False

        e_entry = struct.unpack_from("<Q", elf_data, 24)[0]
        e_phoff = struct.unpack_from("<Q", elf_data, 32)[0]
        e_shoff = struct.unpack_from("<Q", elf_data, 40)[0]
        e_phentsize = struct.unpack_from("<H", elf_data, 54)[0]
        e_phnum = struct.unpack_from("<H", elf_data, 56)[0]
        e_shentsize = struct.unpack_from("<H", elf_data, 58)[0]
        e_shnum = struct.unpack_from("<H", elf_data, 60)[0]
        e_shstrndx = struct.unpack_from("<H", elf_data, 62)[0]

        # Find text segment for address range
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", elf_data, off)[0]
            if p_type == 1:  # PT_LOAD
                p_vaddr = struct.unpack_from("<Q", elf_data, off + 16)[0]
                p_memsz = struct.unpack_from("<Q", elf_data, off + 40)[0]
                p_flags = struct.unpack_from("<I", elf_data, off + 4)[0]
                if p_flags & 0x1:  # PF_X — executable segment
                    self._text_start = p_vaddr
                    self._text_end = p_vaddr + p_memsz
                    break

        # Find base address (lowest PT_LOAD)
        min_vaddr = float("inf")
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", elf_data, off)[0]
            if p_type == 1:
                p_vaddr = struct.unpack_from("<Q", elf_data, off + 16)[0]
                if p_vaddr < min_vaddr:
                    min_vaddr = p_vaddr
        self._base_addr = min_vaddr if min_vaddr != float("inf") else 0
        self._entry_addr = e_entry

        # Parse symbol table
        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return False

        shstr_off = e_shoff + e_shstrndx * e_shentsize
        shstr_offset = struct.unpack_from("<Q", elf_data, shstr_off + 24)[0]

        symtab_sec = strtab_sec = None
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_type = struct.unpack_from("<I", elf_data, sh + 4)[0]
            sh_name_idx = struct.unpack_from("<I", elf_data, sh)[0]
            name = elf_data[
                shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32
            ].split(b"\x00")[0]
            if sh_type == 2:  # SHT_SYMTAB
                symtab_sec = sh
            elif sh_type == 3 and name == b".strtab":  # SHT_STRTAB
                strtab_sec = sh

        if symtab_sec is None or strtab_sec is None:
            return False

        sym_offset = struct.unpack_from("<Q", elf_data, symtab_sec + 24)[0]
        sym_size = struct.unpack_from("<Q", elf_data, symtab_sec + 32)[0]
        sym_entsize = struct.unpack_from("<Q", elf_data, symtab_sec + 56)[0]
        if sym_entsize == 0:
            return False
        sym_count = sym_size // sym_entsize
        strtab_offset = struct.unpack_from("<Q", elf_data, strtab_sec + 24)[0]

        func_addrs: list[tuple[str, int]] = []
        for i in range(min(sym_count, 50000)):
            sym = sym_offset + i * sym_entsize
            st_info = struct.unpack_from("<B", elf_data, sym + 4)[0]
            st_value = struct.unpack_from("<Q", elf_data, sym + 8)[0]
            st_size = struct.unpack_from("<Q", elf_data, sym + 24)[0]
            st_name_idx = struct.unpack_from("<I", elf_data, sym)[0]
            name = (
                elf_data[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 128]
                .split(b"\x00")[0]
                .decode(errors="replace")
            )
            # STT_FUNC = 2
            if (st_info & 0xf) == 2 and st_value > 0 and st_value >= self._text_start:
                end = st_value + st_size if st_size > 0 else st_value + 1
                func_addrs.append((name, st_value))
                self.functions[name] = (st_value, end)

        # Sort by address for binary search in bb->func mapping
        func_addrs.sort(key=lambda x: x[1])
        for name, addr in func_addrs:
            self.addr_to_func[addr] = name

        log.debug("Parsed %d functions from %s", len(func_addrs), self.target)
        return len(func_addrs) > 0

    def _resolve_targets(self):
        """Resolve target names to addresses."""
        for name in self.target_names:
            # Try as hex address
            try:
                addr = int(name, 16)
                self.target_addrs.add(addr)
                continue
            except ValueError:
                pass
            # Try as function name (exact match)
            if name in self.functions:
                self.target_addrs.add(self.functions[name][0])
                continue
            # Try as substring match
            for fname, (start, _end) in self.functions.items():
                if name in fname:
                    self.target_addrs.add(start)

    def _build_call_graph(self, elf_data: bytes):
        """Build call graph by scanning CALL instructions in each function.

        Also resolves PLT stubs: if a CALL targets a PLT entry, follows
        it to the real function name (PLT names typically match the target).
        """
        # Build PLT stub lookup: plt_address -> plt_name
        plt_addrs: dict[int, str] = {}
        for fname, (start, _end) in self.functions.items():
            if fname.startswith(".plt") or fname.startswith("plt."):
                plt_addrs[start] = fname

        for fname, (start, end) in self.functions.items():
            if end <= start or start < self._text_start or end > self._text_end:
                continue
            if end > len(elf_data):
                continue
            code = elf_data[start:end]
            self.call_graph[fname] = set()

            for m in _CALL_RE.finditer(code):
                offset = m.start()
                if offset + 5 > len(code):
                    continue
                disp = struct.unpack_from("<i", code, offset + 1)[0]
                call_target = start + offset + 5 + disp

                # Check if this calls a PLT stub — if so, resolve the PLT name
                # PLT stubs for functions like "main" are named ".plt.main" or similar
                plt_name = plt_addrs.get(call_target)
                if plt_name:
                    # Extract the real function name from PLT name
                    # e.g., ".plt.main" -> "main", "plt.__libc_start_main" -> "__libc_start_main"
                    real_name = plt_name.replace(".plt.", "").replace("plt.", "")
                    if real_name and real_name != fname:
                        self.call_graph[fname].add(real_name)
                    continue

                target_func = self._addr_to_function(call_target)
                if target_func and target_func != fname:
                    self.call_graph[fname].add(target_func)

    def _addr_to_function(self, addr: int) -> str | None:
        """Map an address to its containing function via binary search."""
        best_name = None
        best_start = -1
        for fname, (start, end) in self.functions.items():
            if start <= addr < end and start > best_start:
                best_name = fname
                best_start = start
        return best_name

    def _reachable_from(self, start_func: str) -> set[str]:
        """BFS from start_func through call graph, return reachable function names."""
        visited = {start_func}
        queue = [start_func]
        while queue:
            current = queue.pop(0)
            for callee in self.call_graph.get(current, set()):
                if callee not in visited:
                    visited.add(callee)
                    queue.append(callee)
        return visited

    def _compute_distances(self):
        """BFS from entry point through call graph to compute distances."""
        entry_func = self._addr_to_function(self._entry_addr)
        if entry_func is None:
            for name in ("main", "_start", "__libc_start_main"):
                if name in self.functions:
                    entry_func = name
                    break
        if entry_func is None:
            log.warning("Cannot find entry function, using distance=1 for all")
            for fname in self.functions:
                self._distances[fname] = 1.0
            return

        # Heuristic: if _start doesn't reach main via call graph,
        # add synthetic edges for common patterns
        if entry_func == "_start" and "main" in self.functions and "main" not in self._reachable_from(entry_func):
                # _start -> __libc_start_main -> main is the standard pattern
                self.call_graph.setdefault("_start", set()).add("main")
        if entry_func is None:
            log.warning("Cannot find entry function, using distance=1 for all")
            for fname in self.functions:
                self._distances[fname] = 1.0
            return

        # BFS from entry
        visited: dict[str, float] = {entry_func: 0.0}
        queue = [entry_func]
        while queue:
            current = queue.pop(0)
            current_dist = visited[current]
            for callee in self.call_graph.get(current, set()):
                if callee not in visited:
                    visited[callee] = current_dist + 1.0
                    queue.append(callee)

        # Assign distance to all functions — unreachable ones get a high penalty
        max_dist = max(visited.values()) if visited else 1.0
        for fname in self.functions:
            self._distances[fname] = visited.get(fname, max_dist + 5.0)

        # Assign distance to target functions (distance 0 by definition)
        for taddr in self.target_addrs:
            tfname = self._addr_to_function(taddr)
            if tfname:
                self._distances[tfname] = 0.0
                log.info("Target function: %s @ 0x%x (distance=0)", tfname, taddr)

    def bb_distance(self, bb_addr: int) -> float:
        """Get the distance of a basic block address to the nearest target.

        Returns 0.0 if the block is in a target function, higher values
        for blocks farther away in the call graph.
        """
        if bb_addr in self._bb_distances:
            return self._bb_distances[bb_addr]

        func_name = self._addr_to_function(bb_addr)
        if func_name is None:
            # Unknown function — assign distance based on address proximity
            # to nearest known function (heuristic)
            dist = self._heuristic_distance(bb_addr)
        else:
            dist = self._distances.get(func_name, 10.0)

        self._bb_distances[bb_addr] = dist
        return dist

    def _heuristic_distance(self, addr: int) -> float:
        """Heuristic distance for addresses not in any known function."""
        # Find nearest function by address
        min_dist = float("inf")
        for fname, (start, end) in self.functions.items():
            if start <= addr < end:
                return self._distances.get(fname, 10.0)
            d = min(abs(addr - start), abs(addr - end))
            if d < min_dist:
                min_dist = d
        # Scale: 1 byte away = 1.0 distance, capped at 20
        return min(min_dist / 64.0 + 2.0, 20.0)

    def seed_distance(self, edge_trace: set[tuple[int, int]]) -> float:
        """Compute average distance-to-target for a seed's execution trace.

        Args:
            edge_trace: Set of (prev_bb, curr_bb) edges hit by this seed.

        Returns:
            Average distance across all basic blocks in the trace.
            Lower is better (closer to target).
        """
        if not edge_trace:
            return 20.0  # unknown trace → high distance

        distances = []
        seen_bbs: set[int] = set()
        for _prev_bb, curr_bb in edge_trace:
            if curr_bb not in seen_bbs:
                seen_bbs.add(curr_bb)
                distances.append(self.bb_distance(curr_bb))

        return sum(distances) / len(distances) if distances else 20.0

    @property
    def max_distance(self) -> float:
        """Maximum distance value (for normalization)."""
        if not self._distances:
            return 10.0
        return max(self._distances.values()) + 1.0

    def is_target(self, bb_addr: int) -> bool:
        """Check if a basic block address is in a target function."""
        func_name = self._addr_to_function(bb_addr)
        if func_name is None:
            return False
        func_start = self.functions[func_name][0]
        return func_start in self.target_addrs
