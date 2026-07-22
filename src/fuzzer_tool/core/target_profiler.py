"""Static analysis of target binaries for fuzzing guidance.

Performs ELF parsing to extract string constants, function boundaries,
magic bytes, and input format hints. The profile feeds into dictionary
population, mutation weighting, and seed selection.
"""

import collections
import logging
import re
import struct
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Known file signatures (magic bytes at offset 0)
MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\x7fELF", "elf"),
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip"),
    (b"\xfd7zXZ", "xz"),
    (b"BZh", "bz2"),
    (b"\x1f\x8b", "gzip"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "riff"),
    (b"\x00\x00\x01\x00", "ico"),
    (b"\x00\x00\x02\x00", "cur"),
    (b"MZ", "pe"),
    (b"<!DOCTYPE", "html"),
    (b"<html", "html"),
    (b"<?xml", "xml"),
    (b"{", "json/text"),
]

# Delimiters and separators found in string constants
DELIMITER_CHARS = set(b":/\n\r\t,;=&?#[]{}<>\\\"'!@$%^*()+|~`")


@dataclass
class FunctionInfo:
    """Metadata for a single function."""

    addr: int
    size: int
    name: str
    bb_count: int = 0
    call_depth: int = 0
    branch_density: float = 0.0


@dataclass
class TargetProfile:
    """Static analysis results for a target binary."""

    # String extraction
    rodata_strings: list[tuple[int, str]] = field(default_factory=list)
    interesting_strings: list[str] = field(default_factory=list)
    magic_bytes: list[bytes] = field(default_factory=list)

    # Capstone-based constant extraction from disassembly
    extracted_constants: list[bytes] = field(default_factory=list)

    # Function analysis
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    hot_functions: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)

    # Input format hints
    input_parsers: list[str] = field(default_factory=list)
    boundary_markers: list[bytes] = field(default_factory=list)
    format_signature: str | None = None

    # Call graph
    call_graph: dict[str, set[str]] = field(default_factory=dict)
    reverse_calls: dict[str, set[str]] = field(default_factory=dict)


# Functions that indicate input parsing
INPUT_PARSER_NAMES = {
    "fread",
    "fgets",
    "read",
    "fgetc",
    "getchar",
    "fscanf",
    "scanf",
    "sscanf",
    "strtol",
    "strtod",
    "atoi",
    "atof",
    "strtok",
    "strsep",
    "getline",
    "png_create_read_struct",
    "png_read_image",
    "png_init_io",
    "inflate",
    "uncompress",
    "BZ2_bzDecompress",
    "XML_Parse",
    "yyparse",
}

# Format string patterns that indicate text input processing
FORMAT_STRING_RE = re.compile(rb"%[0-9]*[diouxXeEfFgGaAcspn%]")
ERROR_KEYWORDS = re.compile(
    rb"(error|invalid|overflow|underflow|corrupt|malformed|bad |failed|unable)",
    re.IGNORECASE,
)
INTERESTING_KEYWORDS = re.compile(
    rb"(NULL|true|false|yes|no|on|off|enable|disable|debug|verbose|quiet)",
    re.IGNORECASE,
)


class TargetProfiler:
    """Static analysis of an ELF target binary.

    Args:
        target: Path to the ELF binary.
    """

    def __init__(self, target: str):
        self.target = target
        self._elf: bytes | None = None
        self._sections: dict[
            str, tuple[int, int, int, int]
        ] = {}  # name -> (type, offset, addr, size)
        self._shstrtab: bytes = b""
        self._symtab: list[tuple[str, int, int, int]] = []  # (name, addr, size, type)
        self._strtab: bytes = b""

    def profile(self) -> TargetProfile:
        """Run all analysis passes and return a TargetProfile."""
        profile = TargetProfile()

        try:
            with open(self.target, "rb") as f:
                self._elf = f.read()
        except OSError:
            log.warning("Cannot read target: %s", self.target)
            return profile

        if not self._parse_elf_header():
            return profile

        self._parse_sections()
        self._parse_symbol_tables()

        # 1. String extraction
        self._extract_strings(profile)

        # 2. Capstone compile-time constant extraction (disassembly immediates)
        self._extract_constants(profile)

        # 3. Function analysis
        self._analyze_functions(profile)

        # 3. Input format inference
        self._infer_format(profile)

        # 4. Boundary detection
        self._detect_boundaries(profile)

        # 5. Build call graph
        self._build_call_graph(profile)

        return profile

    def _parse_elf_header(self) -> bool:
        """Parse ELF header and validate."""
        if self._elf is None or len(self._elf) < 64:
            return False
        if self._elf[:4] != b"\x7fELF":
            return False
        if self._elf[4] != 2 or self._elf[5] != 1:  # ELF64, little-endian
            return False
        return True

    def _parse_sections(self):
        """Parse section headers and build section lookup."""
        elf = self._elf
        if elf is None:
            return

        e_shoff = struct.unpack_from("<Q", elf, 40)[0]
        e_shnum = struct.unpack_from("<H", elf, 60)[0]
        e_shentsize = struct.unpack_from("<H", elf, 58)[0]
        e_shstrndx = struct.unpack_from("<H", elf, 62)[0]

        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return

        # Read section header string table
        shstr_off = e_shoff + e_shstrndx * e_shentsize
        if shstr_off + e_shentsize > len(elf):
            return
        shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]
        shstr_size = struct.unpack_from("<Q", elf, shstr_off + 32)[0]
        self._shstrtab = elf[shstr_offset : shstr_offset + shstr_size]

        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            if sh + e_shentsize > len(elf):
                break
            sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
            sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
            name = (
                self._shstrtab[sh_name_idx : sh_name_idx + 32]
                .split(b"\x00")[0]
                .decode(errors="replace")
            )
            sh_offset = struct.unpack_from("<Q", elf, sh + 24)[0]
            sh_addr = struct.unpack_from("<Q", elf, sh + 16)[0]
            sh_size = struct.unpack_from("<Q", elf, sh + 32)[0]
            self._sections[name] = (sh_type, sh_offset, sh_addr, sh_size)

            # Also read .strtab for symbol names
            if sh_type == 3 and name == ".strtab":
                self._strtab = elf[sh_offset : sh_offset + sh_size]

    def _parse_symbol_tables(self):
        """Parse symtab and dynsym for function entries."""
        elf = self._elf
        if elf is None:
            return

        for sec_name in (".symtab", ".dynsym"):
            if sec_name not in self._sections:
                continue
            _, sym_offset, _, sym_size = self._sections[sec_name]
            if sym_size == 0:
                continue

            # Find the associated string table
            strtab_name = ".strtab" if sec_name == ".symtab" else ".dynstr"
            strtab = self._strtab if sec_name == ".symtab" else b""
            if strtab_name in self._sections:
                _, st_offset, _, st_size = self._sections[strtab_name]
                strtab = elf[st_offset : st_offset + st_size]

            if not strtab:
                continue

            # Determine entry size from section header
            entsize = 24  # default for ELF64 symtab
            if sec_name == ".dynsym" and ".rela.dyn" in self._sections:
                # dynsym entry size = 24 bytes
                entsize = 24

            sym_count = sym_size // entsize if entsize else 0
            for i in range(min(sym_count, 50000)):
                sym = sym_offset + i * entsize
                if sym + entsize > len(elf):
                    break
                st_info = elf[sym + 4]
                st_value = struct.unpack_from("<Q", elf, sym + 8)[0]
                st_size = struct.unpack_from("<Q", elf, sym + 16)[0]
                st_name_idx = struct.unpack_from("<I", elf, sym)[0]
                st_type = st_info & 0xF

                name = (
                    strtab[st_name_idx : st_name_idx + 64]
                    .split(b"\x00")[0]
                    .decode(errors="replace")
                )

                if st_type == 2 and st_value > 0:  # STT_FUNC
                    self._symtab.append((name, st_value, st_size, st_type))

    def _extract_strings(self, profile: TargetProfile):
        """Extract strings from .rodata and detect magic bytes."""
        if self._elf is None:
            return

        # Get .rodata section
        if ".rodata" in self._sections:
            _, rodata_offset, _, rodata_size = self._sections[".rodata"]
            rodata = self._elf[rodata_offset : rodata_offset + rodata_size]
        else:
            # Fallback: scan entire binary for strings
            rodata = self._elf

        # Extract null-terminated strings (min length 4)
        strings = []
        current = []
        start = 0
        for i, b in enumerate(rodata):
            if 32 <= b < 127:  # printable ASCII
                if not current:
                    start = i
                current.append(b)
            else:
                if len(current) >= 4:
                    s = bytes(current).decode("ascii", errors="replace")
                    strings.append((start, s))
                current = []

        if len(current) >= 4:
            s = bytes(current).decode("ascii", errors="replace")
            strings.append((start, s))

        profile.rodata_strings = strings

        # Filter for interesting strings
        interesting = []
        for offset, s in strings:
            sb = s.encode("ascii", errors="replace")
            if FORMAT_STRING_RE.search(sb):
                interesting.append(s)
            elif ERROR_KEYWORDS.search(sb):
                interesting.append(s)
            elif INTERESTING_KEYWORDS.search(sb):
                interesting.append(s)
            elif len(s) >= 8 and not s.startswith(("_", ".")):
                # Long non-mangled strings are often user-visible
                interesting.append(s)

        # Deduplicate while preserving order
        seen = set()
        unique_interesting = []
        for s in interesting:
            if s not in seen:
                seen.add(s)
                unique_interesting.append(s)
        profile.interesting_strings = unique_interesting[:500]

        # Detect magic bytes
        magic = []
        for sig, fmt in MAGIC_SIGNATURES:
            if self._elf[: len(sig)] == sig:
                magic.append(sig)
        # Also scan for magic bytes at common offsets in .rodata
        if ".rodata" in self._sections:
            for sig, fmt in MAGIC_SIGNATURES:
                idx = rodata.find(sig)
                if idx >= 0 and sig not in magic:
                    magic.append(sig)
        profile.magic_bytes = magic

    def _extract_constants(self, profile: TargetProfile):
        """Extract compile-time constants from disassembly via Capstone.

        Disassembles .text and extracts immediate operands from comparison
        instructions (CMP, TEST, AND, OR, XOR, SUB, ADD).  These catch
        inlined magic constants, bitmasks, and boundary values that the
        .rodata string scan misses.
        """
        try:
            from fuzzer_tool.core.elf import extract_capstone_constants

            constants = extract_capstone_constants(self.target)
            if constants:
                profile.extracted_constants = constants
                log.info(
                    "Capstone: extracted %d disassembly constants from %s",
                    len(constants),
                    self.target,
                )
        except Exception as e:
            log.debug("Capstone constant extraction failed: %s", e)

    def _analyze_functions(self, profile: TargetProfile):
        """Analyze functions: sizes, branch density, hot functions."""
        if self._elf is None:
            return

        # Get .text data for disassembly
        text_data = None
        text_vaddr = 0
        if ".text" in self._sections:
            _, text_offset, text_vaddr, text_size = self._sections[".text"]
            text_data = self._elf[text_offset : text_offset + text_size]

        # Try capstone for branch counting
        capstone_available = False
        md = None
        try:
            from capstone import CS_ARCH_X86, CS_MODE_64, Cs
            from capstone.x86_const import X86_GRP_JUMP

            md = Cs(CS_ARCH_X86, CS_MODE_64)
            md.detail = True
            capstone_available = True
        except ImportError:
            log.debug("capstone not available — deep branch density disabled")

        for name, addr, size, st_type in self._symtab:
            if size == 0:
                size = 256  # estimate for symbols without size

            bb_count = max(1, size // 16)  # rough estimate

            # Count conditional branches in this function's code
            branch_density = 0.0
            if capstone_available and text_data and text_vaddr > 0:
                func_start = addr - text_vaddr
                func_end = func_start + size
                if 0 <= func_start < len(text_data) and func_end <= len(text_data):
                    func_bytes = text_data[func_start:func_end]
                    cond_branches = 0
                    try:
                        for insn in md.disasm(func_bytes, addr):
                            if X86_GRP_JUMP in insn.groups:
                                is_jcc = (
                                    insn.bytes[0] == 0x0F
                                    and len(insn.bytes) >= 2
                                    and (insn.bytes[1] & 0xF0) == 0x80
                                ) or insn.bytes[0] in range(0x70, 0x80)
                                if is_jcc:
                                    cond_branches += 1
                    except Exception:
                        log.debug(
                            "Instruction parse failed at %#x in %s",
                            insn.address if "insn" in dir() else 0,
                            name,
                            exc_info=True,
                        )
                    branch_density = (cond_branches / max(size, 1)) * 1024

            profile.functions[name] = FunctionInfo(
                addr=addr,
                size=size,
                name=name,
                bb_count=bb_count,
                branch_density=branch_density,
            )

        # Identify hot functions (highest branch density)
        if profile.functions:
            sorted_funcs = sorted(
                profile.functions.items(),
                key=lambda x: x[1].branch_density,
                reverse=True,
            )
            # Top 20% by branch density, minimum 3
            n_hot = max(3, len(sorted_funcs) // 5)
            profile.hot_functions = [name for name, _ in sorted_funcs[:n_hot]]

    def _infer_format(self, profile: TargetProfile):
        """Infer input format from string constants and function names."""
        if self._elf is None:
            return

        # Check for format-specific function names
        func_names = set(profile.functions.keys())
        func_names_lower = {n.lower() for n in func_names}

        # PNG detection
        png_funcs = {
            "png_create_read_struct",
            "png_read_image",
            "png_init_io",
            "png_set_sig_bytes",
            "png_sig_cmp",
        }
        if png_funcs & func_names_lower:
            profile.format_signature = "png"
            return

        # ELF detection
        if "elf_begin" in func_names_lower or "elf_kind" in func_names_lower:
            profile.format_signature = "elf"
            return

        # XML/HTML detection
        xml_funcs = {"xml_parse", "xml_sax_parse", "xml_read_memory"}
        if xml_funcs & func_names_lower:
            profile.format_signature = "xml"
            return

        # JSON detection
        json_funcs = {"json_parse", "yajl_parse", "cJSON_Parse"}
        if json_funcs & func_names_lower:
            profile.format_signature = "json"
            return

        # Archive detection
        archive_funcs = {"gzread", "gzopen", "BZ2_bzDecompress", "uncompress"}
        if archive_funcs & func_names_lower:
            profile.format_signature = "archive"
            return

        # Heuristic: check string constants for format indicators
        all_strings = " ".join(s for _, s in profile.rodata_strings[:200]).lower()
        if "png" in all_strings and ("chunk" in all_strings or "ihdr" in all_strings):
            profile.format_signature = "png"
            return
        if "xml" in all_strings or "doctype" in all_strings:
            profile.format_signature = "xml"
            return
        if "json" in all_strings:
            profile.format_signature = "json"
            return

        # Heuristic: check for magic bytes
        if profile.magic_bytes:
            sig = profile.magic_bytes[0]
            for magic, fmt in MAGIC_SIGNATURES:
                if sig == magic:
                    profile.format_signature = fmt
                    return

        # Heuristic: check for text-like delimiters
        text_score = 0
        for _, s in profile.rodata_strings[:100]:
            if ":" in s or "\n" in s or "=" in s:
                text_score += 1
        if text_score > 10:
            profile.format_signature = "text"
            return

        profile.format_signature = "unknown"

    def _detect_boundaries(self, profile: TargetProfile):
        """Detect input boundary markers from string constants."""
        boundary_candidates = set()

        for _, s in profile.rodata_strings[:300]:
            # Extract single-char and multi-char delimiters
            for ch in (
                b":",
                b"/",
                b"\n",
                b"\r\n",
                b"\t",
                b",",
                b";",
                b"=",
                b"&",
                b"#",
                b"?",
                b"<",
                b">",
                b"{",
                b"}",
                b"[",
                b"]",
                b"(",
                b")",
            ):
                if ch in s.encode("ascii", errors="replace"):
                    boundary_candidates.add(ch)

            # Multi-char boundaries
            sb = s.encode("ascii", errors="replace")
            for boundary in (
                b"HTTP/",
                b"Content-Type:",
                b"Accept:",
                b"Host:",
                b"User-Agent:",
                b"GET ",
                b"POST ",
                b"PUT ",
                b"DELETE ",
                b"HEAD ",
                b"HTTP/1.0",
                b"HTTP/1.1",
                b"HTTP/2",
            ):
                if boundary in sb:
                    boundary_candidates.add(boundary)

        profile.boundary_markers = sorted(boundary_candidates)

        # Input parsers: functions that read input
        for name in profile.functions:
            name_lower = name.lower()
            for parser in INPUT_PARSER_NAMES:
                if parser.lower() in name_lower:
                    profile.input_parsers.append(name)
                    break

    def _build_call_graph(self, profile: TargetProfile):
        """Build call graph from REL32 call instructions."""
        if self._elf is None or ".text" not in self._sections:
            return

        _, text_offset, text_vaddr, text_size = self._sections[".text"]
        text_data = self._elf[text_offset : text_offset + text_size]

        # Build address -> function name map
        addr_to_func: dict[int, str] = {}
        for name, addr, size, _ in self._symtab:
            addr_to_func[addr] = name

        # Try capstone for call detection
        try:
            from capstone import CS_ARCH_X86, CS_MODE_64, Cs
            from capstone.x86_const import X86_GRP_CALL

            md = Cs(CS_ARCH_X86, CS_MODE_64)
            md.detail = True
        except ImportError:
            # Fallback: scan for E8 (REL32 call) opcode
            self._build_call_graph_raw(profile, text_data, text_vaddr, addr_to_func)
            return

        for name, addr, size, _ in self._symtab:
            func_start = addr - text_vaddr
            func_end = func_start + size
            if func_start < 0 or func_end > len(text_data):
                continue
            func_bytes = text_data[func_start:func_end]

            try:
                for insn in md.disasm(func_bytes, addr):
                    if X86_GRP_CALL in insn.groups and insn.op_str.startswith("0x"):
                        target_addr = int(insn.op_str, 16)
                        if target_addr in addr_to_func:
                            callee = addr_to_func[target_addr]
                            if name not in profile.call_graph:
                                profile.call_graph[name] = set()
                            profile.call_graph[name].add(callee)
            except Exception:
                log.debug("Call graph extraction failed for %s", name, exc_info=True)
                continue

        # Build reverse call graph
        for caller, callees in profile.call_graph.items():
            for callee in callees:
                if callee not in profile.reverse_calls:
                    profile.reverse_calls[callee] = set()
                profile.reverse_calls[callee].add(caller)

        # Compute call depth via BFS from entry points
        self._compute_call_depths(profile)

    def _build_call_graph_raw(self, profile, text_data, text_vaddr, addr_to_func):
        """Fallback call graph using raw E8 opcode scanning."""
        for i in range(len(text_data) - 5):
            if text_data[i] != 0xE8:  # REL32 call opcode
                continue
            offset = struct.unpack_from("<i", text_data, i + 1)[0]
            call_addr = text_vaddr + i
            target_addr = text_vaddr + i + 5 + offset

            # Find caller function
            caller = None
            for name, addr, size, _ in self._symtab:
                if addr <= call_addr < addr + size:
                    caller = name
                    break
            if caller is None:
                continue

            callee = addr_to_func.get(target_addr)
            if callee and callee != caller:
                if caller not in profile.call_graph:
                    profile.call_graph[caller] = set()
                profile.call_graph[caller].add(callee)

        # Build reverse call graph
        for caller, callees in profile.call_graph.items():
            for callee in callees:
                if callee not in profile.reverse_calls:
                    profile.reverse_calls[callee] = set()
                profile.reverse_calls[callee].add(caller)

        self._compute_call_depths(profile)

    def _compute_call_depths(self, profile: TargetProfile):
        """Compute call depth from entry points via BFS."""
        # Find entry points: functions not called by anyone (roots)
        all_funcs = set(profile.functions.keys())
        called_funcs = set()
        for callees in profile.call_graph.values():
            called_funcs.update(callees)

        roots = all_funcs - called_funcs
        if not roots:
            # Fallback: use main or _start
            for name in ("main", "_start", "__libc_start_main"):
                if name in all_funcs:
                    roots.add(name)
            if not roots and all_funcs:
                roots = {min(all_funcs)}

        profile.entry_points = sorted(roots)

        # BFS to compute depths
        queue = collections.deque((r, 0) for r in roots)
        visited: dict[str, int] = {}
        while queue:
            func, depth = queue.popleft()
            if func in visited:
                continue
            visited[func] = depth
            if func in profile.functions:
                profile.functions[func].call_depth = depth
            for callee in profile.call_graph.get(func, set()):
                if callee not in visited:
                    queue.append((callee, depth + 1))

        # Mark live functions (reachable from entry)
        live = set(visited.keys())
        for name in profile.functions:
            if name in live:
                profile.entry_points.append(name)
        # Deduplicate
        seen = set()
        unique = []
        for p in profile.entry_points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        profile.entry_points = unique


# Format signature -> structure-aware mutation operators that are almost
# certainly useful for that format (see core.mutations.FORMAT_MUTATIONS).
_FORMAT_OPERATOR_HINTS: dict[str, tuple[str, ...]] = {
    "png": ("png_chunk_mutate", "png_crc_fix"),
    "jpeg": ("jpeg_chunk_mutate", "jpeg_crc_fix"),
    "gzip": ("gzip_chunk_mutate",),
    "bz2": ("gzip_chunk_mutate",),
    "xz": ("gzip_chunk_mutate",),
    "zlib": ("zlib_chunk_mutate",),
    "riff": ("bmp_chunk_mutate",),
}

# Dictionary/token-aware mutation operators (see core.mutations.DICT_MUTATIONS).
_DICT_OPERATORS: tuple[str, ...] = (
    "dict_insert",
    "dict_replace",
    "dict_overwrite",
    "dict_prepend",
    "dict_append",
    "checksum_repair",
    "token_dup",
)

# Beta prior for operators the profile suggests are relevant: same total
# "pseudo-observation" mass as the uninformative default (1, 1), but shifted
# toward success so Thompson sampling favors them before real evidence
# arrives. Weak enough that a handful of real failures will correct it.
_BOOSTED_PRIOR: tuple[float, float] = (2.0, 1.0)


def format_operator_priors(profile: "TargetProfile") -> dict[str, tuple[float, float]]:
    """Derive informative Beta priors for mutation operators from a profile.

    Static analysis (magic bytes, boundary markers, extracted interesting
    strings, format signature) is prior knowledge about which
    structure-aware mutation operators are likely to be useful *before*
    any executions have happened. This lets the Thompson-sampling bandit
    (:class:`fuzzer_tool.core.montecarlo.MonteCarloScheduler`) start with a
    Beta prior biased toward those operators instead of the uninformative
    Beta(1, 1) used for every arm by default.

    Args:
        profile: A populated :class:`TargetProfile`.

    Returns:
        Mapping of operator name to a (prior_alpha, prior_beta) override.
        Operators not present in the mapping should use the default
        Beta(1, 1) prior.
    """
    priors: dict[str, tuple[float, float]] = {}

    fmt = profile.format_signature
    if fmt:
        for op in _FORMAT_OPERATOR_HINTS.get(fmt, ()):
            priors[op] = _BOOSTED_PRIOR

    # Boost dict-aware operators whenever static analysis extracted at least
    # one usable token (non-empty list is truthy; empty list is falsy).
    if profile.magic_bytes or profile.boundary_markers or profile.interesting_strings:
        for op in _DICT_OPERATORS:
            priors[op] = _BOOSTED_PRIOR

    return priors
