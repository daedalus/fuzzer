"""AFLGo-style directed distance computation for targeted fuzzing.

Computes call-graph (function-level) and control-flow-graph
(basic-block-level) distances from every basic block to a set of target
locations, following the AFLGo directed greybox fuzzing algorithm
(``distance/distance_calculator/distance.py`` in the AFLGo repo):

  CG distance::

      d_cg(f) = |T_f| / sum_{t in T_f} 1 / (1 + d_bfs(f, t))

  the harmonic mean of (1 + shortest call-graph path) over reachable
  target functions, computed with unweighted BFS on the reverse call
  graph.  Target functions get distance 1, their direct callers 2, and
  functions with no path to any target a penalty distance.

  CFG distance: within a function that contains target basic blocks::

      d_cfg(b) = |T_b| / sum_{t in T_b} 1 / (1 + d_cfg_path(b, t))

  over the function's target blocks (BFS on the reversed
  intra-procedural CFG, built by ``fuzzer_tool.core.cfg``).  Target
  blocks themselves get distance 0.  Blocks in functions without target
  blocks get the 0-based function-level distance ``d_cg(func) - 1`` —
  the cross-function bridge (AFLGo reaches the same effect by seeding
  callsite blocks with their callees' CG distance).

Targets are given as function names, hex addresses, or ``file.c:line``
(``file:line`` is resolved through pure-Python DWARF parsing in
``fuzzer_tool.core.dwarf``).

Seed distance is the mean distance of the *valued* basic blocks a seed's
trace hits (AFLGo counts only blocks that carry a distance).  Seeds
whose trace never reaches a valued block get the maximum distance
(20.0).  When no CFG values were built (e.g. no target function could
be disassembled), the legacy function-level mean is used instead.

The distance signal integrates into scheduling via the ``aflgo`` power
schedule (``core/schedules.py``), annealed over time from "maximize
coverage" to "minimize distance."
"""

import bisect
import logging
import multiprocessing
import re
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fuzzer_tool.core import cfg_cache
from fuzzer_tool.core.cfg import FunctionCFG, build_function_cfg

log = logging.getLogger(__name__)

# Call instruction patterns (x86_64)
# REL32 call: E8 xx xx xx xx (5 bytes)
# INDIRECT call: FF 15 (call [rip+disp32]) — we skip these
_CALL_RE = re.compile(rb"\xe8")  # REL32 call opcode

# Caps on the CFG-based analysis (bounds load time on huge functions).
_MAX_CFG_FUNC_SIZE = 256 * 1024  # skip functions larger than 256 KiB
_MAX_CFG_BLOCKS = 4096  # skip harmonic distance for huge CFGs
_MAX_TARGET_BLOCKS = 512  # cap target-block count for the harmonic BFS

# Distance reported for seeds that never hit a valued block.
_NO_VALUE_DISTANCE = 20.0

_FILE_LINE_RE = re.compile(r"^(.*):(\d+)$")


def _resolve_callee_with(
    addr: int,
    functions: dict[str, tuple[int, int]],
    addr_to_func: dict[int, str],
    func_addrs_sorted: list[tuple[int, str]] | None = None,
) -> str | None:
    """Map a call target address to a function name (PLT-aware).

    Module-level so the parallel decode worker can use it without a
    TargetDistance instance; the method below delegates to this.
    """
    plt_name = addr_to_func.get(addr)
    if plt_name and (plt_name.startswith(".plt") or plt_name.startswith("plt.")):
        real_name = plt_name.replace(".plt.", "").replace("plt.", "")
        if real_name and real_name in functions:
            return real_name
    if func_addrs_sorted:
        idx = bisect.bisect_right(func_addrs_sorted, (addr, "")) - 1
        if idx >= 0:
            fname = func_addrs_sorted[idx][1]
            start = func_addrs_sorted[idx][0]
            if start <= addr < functions[fname][1]:
                return fname
        return None
    best_name = None
    best_start = -1
    for fname, (start, end) in functions.items():
        if start <= addr < end and start > best_start:
            best_name = fname
            best_start = start
    return best_name


def _decode_cfg_worker(task: tuple):
    """Decode one function's CFG from its own file handle.

    Task tuples stay small (name, offsets); the binary path and symbol
    tables ship once per worker via _decode_worker_init, so executor.map
    does not re-pickle the symtab-derived dicts for every function.
    Returns the FunctionCFG, or None for unbuildable/empty functions —
    failures must not cross the process boundary as exceptions.
    """
    name, file_off, nbytes, start_vaddr = task
    path, functions, addr_to_func, func_addrs_sorted = _WORKER_CTX
    try:
        with open(path, "rb") as f:
            f.seek(file_off)
            code = f.read(nbytes)
        if len(code) != nbytes:
            return None

        def _resolve(addr: int) -> str | None:
            name = addr_to_func.get(addr)
            if name and (name.startswith(".plt") or name.startswith("plt.")):
                real_name = name.replace(".plt.", "").replace("plt.", "")
                if real_name and real_name in functions:
                    return real_name
            idx = bisect.bisect_right(func_addrs_sorted, (addr, "")) - 1
            if idx >= 0:
                fname = func_addrs_sorted[idx][1]
                start = func_addrs_sorted[idx][0]
                if start <= addr < functions[fname][1]:
                    return fname
            return None

        cfg = build_function_cfg(
            name,
            code,
            start_vaddr,
            _resolve,
        )
        return cfg if cfg.blocks else None
    except Exception:
        log.debug("CFG build failed for %s", name, exc_info=True)
        return None


# Set by _decode_worker_init in each pool worker; never used in the parent.
_WORKER_CTX: tuple = ()


def _decode_worker_init(
    path: str,
    functions: dict[str, tuple[int, int]],
    addr_to_func: dict[int, str],
    func_addrs_sorted: list[tuple[int, str]],
):
    global _WORKER_CTX
    _WORKER_CTX = (path, functions, addr_to_func, func_addrs_sorted)


class TargetDistance:
    """Compute call-graph and CFG distances from basic blocks to targets.

    Args:
        target: Path to the ELF binary.
        targets: List of target function names, hex addresses, or
            ``file.c:line`` specifications.
    """

    def __init__(
        self,
        target: str,
        targets: list[str] | None = None,
        use_cfg_cache: bool = True,
    ):
        self.target = target
        self.target_names: list[str] = targets or []
        self.target_addrs: set[int] = set()
        self._use_cfg_cache = use_cfg_cache

        # Function table: name -> (start_addr, end_addr)
        self.functions: dict[str, tuple[int, int]] = {}
        # Reverse map: address -> function name
        self.addr_to_func: dict[int, str] = {}
        # Call graph: func_name -> set of func_names it calls
        self.call_graph: dict[str, set[str]] = {}
        # CG distance cache: func_name -> harmonic distance to targets
        self._distances: dict[str, float] = {}
        # Per-BB distance cache (legacy address -> distance)
        self._bb_distances: dict[int, float] = {}
        # Intra-procedural CFGs of target functions (name -> FunctionCFG)
        self._cfgs: dict = {}
        # Valued BB start -> distance (AFLGo BB-level values)
        self._bb_value: dict[int, float] = {}
        # Target BB ranges (start, end) for is_target()
        self._target_bb_ranges: set[tuple[int, int]] = set()

        self._loaded = False
        self._entry_addr: int = 0
        self._text_start: int = 0
        self._text_end: int = 0
        self._base_addr: int = 0
        self._elf_data: bytes = b""
        # PT_LOAD segments: (vaddr, filesz, file_offset) for vaddr→offset
        self._segments: list[tuple[int, int, int]] = []
        # Address of __sanitizer_cov_trace_pc (trace-pc distance builds)
        self._trace_pc_addr: int | None = None
        self._dwarf = None
        # Pre-sorted numpy arrays for vectorized distance lookup
        self._func_starts_np = None
        self._func_ends_np = None
        self._func_dists_np = None
        self._bb_starts_np = None
        self._bb_ends_np = None
        self._bb_dists_np = None

    def load(self) -> bool:
        """Parse the ELF and compute distances. Returns True on success."""
        try:
            self._elf_data = Path(self.target).read_bytes()
        except OSError as e:
            log.warning("Cannot read target ELF: %s", e)
            return False

        if len(self._elf_data) < 64 or self._elf_data[:4] != b"\x7fELF":
            log.warning("Not an ELF file: %s", self.target)
            return False

        t0 = time.perf_counter()
        if not self._parse_symbols(self._elf_data):
            return False
        print(f"[dist] parse_symbols={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._resolve_targets()
        print(f"[dist] resolve_targets={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._build_call_graph(self._elf_data)
        print(f"[dist] build_call_graph={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._compute_distances()
        print(f"[dist] compute_distances={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._build_cfgs()
        print(f"[dist] build_cfgs={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._compute_bb_values()
        print(f"[dist] compute_bb_values={time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        self._build_np_index()
        print(f"[dist] build_np_index={time.perf_counter() - t0:.3f}s")

        self._loaded = True
        log.info(
            "TargetDistance: %d functions, %d targets, %d valued BBs, entry=0x%x",
            len(self.functions),
            len(self.target_addrs),
            len(self._bb_value),
            self._entry_addr,
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

        # Find base address (lowest PT_LOAD) and record vaddr→offset
        # translations (PIE/.so targets have vaddr != file offset).
        min_vaddr = float("inf")
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", elf_data, off)[0]
            if p_type == 1:
                p_vaddr = struct.unpack_from("<Q", elf_data, off + 16)[0]
                p_offset = struct.unpack_from("<Q", elf_data, off + 8)[0]
                p_filesz = struct.unpack_from("<Q", elf_data, off + 32)[0]
                if p_filesz > 0:
                    self._segments.append((p_vaddr, p_filesz, p_offset))
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
            name = elf_data[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(
                b"\x00"
            )[0]
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
            # Elf64_Sym: st_size lives at offset 16 (offset 24 is the
            # NEXT symbol's st_name). Reading it at 24 produced garbage
            # function sizes.
            st_size = struct.unpack_from("<Q", elf_data, sym + 16)[0]
            st_name_idx = struct.unpack_from("<I", elf_data, sym)[0]
            name = (
                elf_data[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 128]
                .split(b"\x00")[0]
                .decode(errors="replace")
            )
            if name == "__sanitizer_cov_trace_pc" and st_value > 0:
                self._trace_pc_addr = st_value
            # STT_FUNC = 2
            if (st_info & 0xF) == 2 and st_value > 0 and st_value >= self._text_start:
                end = st_value + st_size if st_size > 0 else st_value + 1
                func_addrs.append((name, st_value))
                self.functions[name] = (st_value, end)

        # Sort by address for binary search in bb->func mapping
        func_addrs.sort(key=lambda x: x[1])
        for name, addr in func_addrs:
            self.addr_to_func[addr] = name

        log.debug("Parsed %d functions from %s", len(func_addrs), self.target)
        # Preserve the sorted order for O(log n) callee lookups.
        self._func_addrs_sorted = [(addr, name) for name, addr in func_addrs]
        return len(func_addrs) > 0

    def _dwarf_resolver(self):
        """Lazily build the DWARF file:line resolver for this target."""
        if self._dwarf is None:
            from fuzzer_tool.core.dwarf import DwarfLineResolver

            self._dwarf = DwarfLineResolver(self.target)
            self._dwarf.load()
        return self._dwarf

    def _resolve_targets(self):
        """Resolve target names to addresses.

        Order of resolution per target string:
          1. ``file.c:line`` (DWARF) — matches ``^(.*):(\\d+)$``.
          2. Hex address (``0x...``).
          3. Function name (exact, then substring).
        """
        for name in self.target_names:
            # Try as file:line via DWARF
            m = _FILE_LINE_RE.match(name)
            if m:
                resolver = self._dwarf_resolver()
                addrs = resolver.resolve(m.group(1), int(m.group(2)))
                if addrs:
                    self.target_addrs.update(addrs)
                    continue
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

    def _resolve_callee_name(self, addr: int) -> str | None:
        """Map a call target address to a function name (PLT-aware)."""
        name = self.addr_to_func.get(addr)
        if name and (name.startswith(".plt") or name.startswith("plt.")):
            real_name = name.replace(".plt.", "").replace("plt.", "")
            if real_name and real_name in self.functions:
                return real_name
        starts = getattr(self, "_func_addrs_sorted", None)
        if starts:
            idx = bisect.bisect_right(starts, (addr, "")) - 1
            if idx >= 0:
                fname = starts[idx][1]
                start = starts[idx][0]
                if start <= addr < self.functions[fname][1]:
                    return fname
            return None
        best_name = None
        best_start = -1
        for fname, (start, end) in self.functions.items():
            if start <= addr < end and start > best_start:
                best_name = fname
                best_start = start
        return best_name

    def _file_offset(self, vaddr: int) -> int | None:
        """Translate a virtual address to its file offset (or None)."""
        for seg_vaddr, seg_filesz, seg_offset in self._segments:
            if seg_vaddr <= vaddr < seg_vaddr + seg_filesz:
                return seg_offset + (vaddr - seg_vaddr)
        return None

    def _code_slice(self, start: int, end: int) -> bytes | None:
        """Return the file bytes covering [start, end) in virtual memory."""
        off = self._file_offset(start)
        if off is None:
            return None
        return self._elf_data[off : off + (end - start)]

    def _build_call_graph(self, elf_data: bytes):
        """Build call graph by scanning CALL instructions in each function.

        Also resolves PLT stubs: if a CALL targets a PLT entry, follows
        it to the real function name (PLT names typically match the
        target).
        """
        for fname, (start, end) in self.functions.items():
            if end <= start or start < self._text_start or end > self._text_end:
                continue
            code = self._code_slice(start, end)
            if code is None or len(code) != end - start:
                continue
            self.call_graph[fname] = set()

            for m in _CALL_RE.finditer(code):
                offset = m.start()
                if offset + 5 > len(code):
                    continue
                disp = struct.unpack_from("<i", code, offset + 1)[0]
                call_target = start + offset + 5 + disp

                target_func = self._resolve_callee_name(call_target)
                if target_func and target_func != fname:
                    self.call_graph[fname].add(target_func)

    def _addr_to_function(self, addr: int) -> str | None:
        """Map an address to its containing function via binary search."""
        starts = getattr(self, "_func_addrs_sorted", None)
        if starts:
            idx = bisect.bisect_right(starts, (addr, "")) - 1
            if idx >= 0:
                fname = starts[idx][1]
                start = starts[idx][0]
                end = self.functions[fname][1]
                if start <= addr < end:
                    return fname
            return None
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
        """Compute AFLGo harmonic-mean CG distances via reverse BFS.

        For every target function t, BFS the reverse call graph to get
        the shortest path from each function f to t; then
        d_cg(f) = |T_f| / sum_t 1/(1 + d_bfs(f,t)) over reachable
        targets.  Functions with no path to any target get a penalty
        (max reachable distance + 5).  With no targets at all, every
        function gets distance 1.0.
        """
        # Find target functions
        target_names = set()
        for taddr in self.target_addrs:
            tfname = self._addr_to_function(taddr)
            if tfname:
                target_names.add(tfname)

        if not target_names:
            log.warning("No target functions found, using distance=1 for all")
            for fname in self.functions:
                self._distances[fname] = 1.0
            return

        # Build reverse call graph: callee → set of callers
        reverse_graph: dict[str, set[str]] = {}
        for caller, callees in self.call_graph.items():
            for callee in callees:
                reverse_graph.setdefault(callee, set()).add(caller)

        # Per-function accumulation of 1/(1 + d) over reachable targets.
        sum_inv: dict[str, float] = {}
        count: dict[str, int] = {}
        for t in target_names:
            # BFS over the reverse graph from target t.
            visited = {t: 0.0}
            queue = [t]
            while queue:
                current = queue.pop(0)
                current_dist = visited[current]
                sum_inv[current] = sum_inv.get(current, 0.0) + 1.0 / (1.0 + current_dist)
                count[current] = count.get(current, 0) + 1
                for caller in reverse_graph.get(current, set()):
                    if caller not in visited:
                        visited[caller] = current_dist + 1.0
                        queue.append(caller)

        reachable_max = max(visited.values()) if visited else 1.0
        for fname in self.functions:
            if count.get(fname, 0) > 0:
                self._distances[fname] = count[fname] / sum_inv[fname]
            else:
                self._distances[fname] = reachable_max + 5.0

        log.info(
            "Target functions: %s; CG distance range [%.2f, %.2f]",
            sorted(target_names),
            min(self._distances.values()),
            max(self._distances.values()),
        )

    def _build_cfgs(self):
        """Build intra-procedural CFGs for target functions only.

        The harmonic CFG distance only needs CFGs of functions that
        contain target blocks; everything else uses the function-level
        CG distance (bounding the load-time disassembly cost).

        Decoded CFGs go through core/cfg_cache.py: hits skip the decode,
        misses are decoded (parallel above the pool thresholds) and then
        stored, accumulate-merged per function so a later run with
        different --target-functions still hits.
        """
        target_funcs = set()
        for taddr in self.target_addrs:
            tfname = self._addr_to_function(taddr)
            if tfname:
                target_funcs.add(tfname)

        ident = None
        cached: dict = {}
        if self._use_cfg_cache and cfg_cache.env_enabled():
            ident = cfg_cache.identity(self.target)
            if ident is not None:
                cached = cfg_cache.load(ident) or {}
        wanted = sorted(target_funcs)
        for name in wanted:
            hit = cached.get(name)
            if hit is not None and name not in self._cfgs:
                self._cfgs[name] = hit

        specs: list[tuple[int, str, int, int]] = []
        total_bytes = 0
        for name in wanted:
            if name in self._cfgs:
                continue
            start, end = self.functions.get(name, (0, 0))
            if end <= start or end - start > _MAX_CFG_FUNC_SIZE:
                continue
            off = self._file_offset(start)
            if off is None:
                continue
            specs.append((off, name, start, end))
            total_bytes += end - start

        decoded: list[FunctionCFG] = []
        if specs and cfg_cache.should_parallelize(total_bytes, len(specs)):
            tasks = [(name, off, end - start, start) for off, name, start, end in specs]
            with ProcessPoolExecutor(
                max_workers=cfg_cache.MAX_WORKERS,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_decode_worker_init,
                initargs=(
                    self.target,
                    self.functions,
                    self.addr_to_func,
                    getattr(self, "_func_addrs_sorted", []),
                ),
            ) as ex:
                for (_, _name, _, _), cfg in zip(
                    specs, ex.map(_decode_cfg_worker, tasks), strict=True
                ):
                    if cfg is not None:
                        decoded.append(cfg)
        else:
            for _off, name, start, end in specs:
                code = self._code_slice(start, end)
                if code is None or len(code) != end - start:
                    continue
                try:
                    cfg = build_function_cfg(name, code, start, self._resolve_callee_name)
                    if cfg.blocks:
                        decoded.append(cfg)
                except Exception:
                    log.debug("CFG build failed for %s", name, exc_info=True)

        new_cfgs = {cfg.name: cfg for cfg in decoded}
        self._cfgs.update(new_cfgs)
        if ident is not None and new_cfgs:
            cfg_cache.store(ident, new_cfgs)

    def _compute_bb_values(self):
        """Compute AFLGo BB-level distances for the target functions.

        Target blocks get 0; other blocks in a target function get the
        harmonic CFG distance to that function's target blocks; blocks
        elsewhere fall back to the 0-based function-level distance
        (d_cg(func) - 1), looked up lazily in ``bb_distance``.
        """
        if not self._cfgs:
            return

        # Blocks containing a target address are the target blocks.
        target_bbs: dict[str, set[int]] = {}
        for taddr in self.target_addrs:
            func = self._addr_to_function(taddr)
            cfg = self._cfgs.get(func) if func else None
            if cfg is None:
                continue
            blk = cfg.block_containing(taddr)
            if blk:
                target_bbs.setdefault(func, set()).add(blk.start)

        for func, cfg in self._cfgs.items():
            tbbs = target_bbs.get(func)
            if not tbbs or len(tbbs) > _MAX_TARGET_BLOCKS:
                continue
            if len(cfg.blocks) > _MAX_CFG_BLOCKS:
                continue

            # Reverse BFS from each target block: for every block b,
            # accumulate 1/(1 + d(b,t)) and the reachable-target count.
            sum_inv: dict[int, float] = {}
            count: dict[int, int] = {}
            for t in tbbs:
                visited = {t: 0.0}
                queue = [t]
                while queue:
                    current = queue.pop(0)
                    d = visited[current]
                    sum_inv[current] = sum_inv.get(current, 0.0) + 1.0 / (1.0 + d)
                    count[current] = count.get(current, 0) + 1
                    for succ in cfg.blocks[current].successors:
                        if succ not in visited:
                            visited[succ] = d + 1.0
                            queue.append(succ)

            for bs, blk in cfg.blocks.items():
                if bs in tbbs:
                    self._bb_value[bs] = 0.0
                    self._target_bb_ranges.add((blk.start, blk.end))
                elif bs in count:
                    self._bb_value[bs] = count[bs] / sum_inv[bs]

    def bb_distance(self, bb_addr: int) -> float:
        """Get the distance of an address to the nearest target.

        Resolution order: valued CFG block → containing function's CG
        distance → address-proximity heuristic.  Returns 0.0 for target
        blocks, small values for code near the targets.
        """
        if bb_addr in self._bb_distances:
            return self._bb_distances[bb_addr]

        dist = self._bb_value_of(bb_addr)
        if dist is None:
            func_name = self._addr_to_function(bb_addr)
            if func_name is None:
                dist = self._heuristic_distance(bb_addr)
            else:
                dist = self._distances.get(func_name, 10.0)

        self._bb_distances[bb_addr] = dist
        return dist

    def _bb_value_of(self, addr: int) -> float | None:
        """Distance of the CFG block containing *addr*, or None."""
        if not self._cfgs:
            return None
        func = self._addr_to_function(addr)
        cfg = self._cfgs.get(func) if func else None
        if cfg is None:
            return None
        blk = cfg.block_containing(addr)
        if blk is None:
            return None
        return self._bb_value.get(blk.start)

    def pc_distance_table(self) -> dict[int, float]:
        """PC→distance table for the SHM-tail channel.

        Keys are the return addresses of ``call __sanitizer_cov_trace_pc``
        sites (the exact PCs the shim's ``__sanitizer_cov_trace_pc()``
        observes), relative to the object base, restricted to blocks with
        a valued AFLGo distance.  Modern clang does not emit a
        ``__sancov_pcs`` section for trace-pc, so the call sites are
        recovered by scanning the text for REL32 calls to the shim's
        trace_pc symbol.  Empty dict for non-distance builds or when no
        site maps to a valued block.
        """
        if not self._bb_value or self._trace_pc_addr is None:
            return {}
        table: dict[int, float] = {}
        base = self._base_addr or 0
        for start, end in self.functions.values():
            if end <= start or start < self._text_start or end > self._text_end:
                continue
            code = self._code_slice(start, end)
            if code is None or len(code) != end - start:
                continue
            for m in _CALL_RE.finditer(code):
                offset = m.start()
                if offset + 5 > len(code):
                    continue
                disp = struct.unpack_from("<i", code, offset + 1)[0]
                call_target = start + offset + 5 + disp
                if call_target != self._trace_pc_addr:
                    continue
                site = start + offset + 5  # return address after the call
                dist = self._bb_value_of(site)
                if dist is not None:
                    table[site - base] = dist
        return table

    def _build_np_index(self):
        """Build pre-sorted numpy arrays for vectorized distance lookup."""
        if not self.functions:
            return
        try:
            import numpy as _np

            starts = []
            ends = []
            dists = []
            for fname, (start, end) in self.functions.items():
                starts.append(start)
                # Cap end addresses — ELF parsing can produce garbage values
                ends.append(min(end, start + 100_000))
                dists.append(self._distances.get(fname, 10.0))
            order = _np.argsort(starts)
            self._func_starts_np = _np.array(starts, dtype=_np.int64)[order]
            self._func_ends_np = _np.array(ends, dtype=_np.int64)[order]
            self._func_dists_np = _np.array(dists, dtype=_np.float64)[order]

            if self._bb_value:
                b_starts = []
                b_ends = []
                b_dists = []
                for bs, value in self._bb_value.items():
                    func = self._addr_to_function(bs)
                    cfg = self._cfgs.get(func) if func else None
                    blk = cfg.blocks.get(bs) if cfg else None
                    if blk is not None:
                        b_starts.append(bs)
                        b_ends.append(blk.end)
                        b_dists.append(value)
                if b_starts:
                    order = _np.argsort(b_starts)
                    self._bb_starts_np = _np.array(b_starts, dtype=_np.int64)[order]
                    self._bb_ends_np = _np.array(b_ends, dtype=_np.int64)[order]
                    self._bb_dists_np = _np.array(b_dists, dtype=_np.float64)[order]
        except (ImportError, OverflowError):
            pass

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

        With CFG values available, averages only the *valued* blocks a
        trace hits (AFLGo semantics); a trace touching no valued block
        gets ``_NO_VALUE_DISTANCE``.  Otherwise falls back to the mean
        function-level distance over the trace's unique blocks.
        """
        if not edge_trace:
            return _NO_VALUE_DISTANCE

        if self._bb_value:
            return self._seed_distance_aflgo(edge_trace)

        if self._func_starts_np is None and self.functions:
            self._build_np_index()
        if self._func_starts_np is not None:
            return self._seed_distance_numpy(edge_trace)
        return self._seed_distance_python(edge_trace)

    def _seed_distance_aflgo(self, edge_trace: set[tuple[int, int]]) -> float:
        """AFLGo-mode mean over valued blocks (vectorized when possible)."""
        seen_bbs = {curr for _prev, curr in edge_trace}
        if not seen_bbs:
            return _NO_VALUE_DISTANCE

        if self._bb_starts_np is not None:
            import numpy as _np

            addrs = _np.array(sorted(seen_bbs), dtype=_np.int64)
            fs = self._bb_starts_np
            fe = self._bb_ends_np
            fd = self._bb_dists_np
            n = len(fs)
            idxs = _np.clip(_np.searchsorted(fs, addrs, side="right") - 1, 0, n - 1)
            in_range = (fs[idxs] <= addrs) & (addrs < fe[idxs])
            values = _np.where(in_range, fd[idxs], -1.0)
            valued = values[values >= 0.0]
            if len(valued) == 0:
                return _NO_VALUE_DISTANCE
            return float(valued.mean())

        distances = []
        for addr in seen_bbs:
            dist = self._bb_value_of(addr)
            if dist is not None:
                distances.append(dist)
        return sum(distances) / len(distances) if distances else _NO_VALUE_DISTANCE

    def _seed_distance_numpy(self, edge_trace: set[tuple[int, int]]) -> float:
        """Vectorized seed distance using numpy searchsorted (branch-free)."""
        import numpy as _np

        # Collect unique BB addresses
        seen_bbs: set[int] = set()
        bb_list = []
        for _prev, curr in edge_trace:
            if curr not in seen_bbs:
                seen_bbs.add(curr)
                bb_list.append(curr)
        if not bb_list:
            return _NO_VALUE_DISTANCE

        addrs = _np.array(bb_list, dtype=_np.int64)
        fs = self._func_starts_np
        fe = self._func_ends_np
        fd = self._func_dists_np
        n = len(fs)

        # Branch-free vectorized lookup: checks idx-1, idx, idx+1
        idxs = _np.clip(_np.searchsorted(fs, addrs, side="right") - 1, 0, n - 1)
        im = _np.clip(idxs - 1, 0, n - 1)
        ip = _np.minimum(idxs + 1, n - 1)

        s_m, e_m, d_m = fs[im], fe[im], fd[im]
        s_0, e_0, d_0 = fs[idxs], fe[idxs], fd[idxs]
        s_p, e_p = fs[ip], fe[ip]

        in_0 = (s_0 <= addrs) & (addrs < e_0)
        in_m = (s_m <= addrs) & (addrs < e_m)
        in_func = in_0 | in_m

        d1 = _np.minimum(_np.abs(addrs - s_m), _np.abs(addrs - e_m))
        d2 = _np.minimum(_np.abs(addrs - s_0), _np.abs(addrs - e_0))
        d3 = _np.minimum(_np.abs(addrs - s_p), _np.abs(addrs - e_p))
        fallback = _np.minimum(_np.minimum(_np.minimum(d1, d2), d3) / 64.0 + 2.0, 20.0)

        results = _np.where(in_func, _np.where(in_0, d_0, d_m), fallback)

        # Cache results
        for i, a in enumerate(bb_list):
            self._bb_distances[a] = float(results[i])

        return float(results.mean())

    def _seed_distance_python(self, edge_trace: set[tuple[int, int]]) -> float:
        """Pure-Python fallback."""
        distances = []
        seen_bbs: set[int] = set()
        for _prev_bb, curr_bb in edge_trace:
            if curr_bb not in seen_bbs:
                seen_bbs.add(curr_bb)
                distances.append(self.bb_distance(curr_bb))
        return sum(distances) / len(distances) if distances else _NO_VALUE_DISTANCE

    @property
    def max_distance(self) -> float:
        """Maximum distance value (for normalization).

        Cached after first computation — the distance tables are only
        modified during initialization, never during fuzzing.
        """
        if not hasattr(self, "_cached_max_distance"):
            self._cached_max_distance = None
        if self._cached_max_distance is not None:
            return self._cached_max_distance
        if self._bb_value:
            self._cached_max_distance = max(self._bb_value.values()) + 1.0
        elif self._distances:
            self._cached_max_distance = max(self._distances.values()) + 1.0
        else:
            self._cached_max_distance = 10.0
        return self._cached_max_distance

    def is_target(self, bb_addr: int) -> bool:
        """Check if an address is inside a target block/function."""
        for start, end in self._target_bb_ranges:
            if start <= bb_addr < end:
                return True
        func_name = self._addr_to_function(bb_addr)
        if func_name is None:
            return False
        func_start = self.functions[func_name][0]
        return func_start in self.target_addrs
