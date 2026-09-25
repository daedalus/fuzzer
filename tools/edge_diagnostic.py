"""Consolidated diagnostic tool for the fuzzgoat edge-coverage determinism
probes (P1-4) and the fuzzer memory/leak profilers.

Consolidates the scratch probes of the P1-4 investigation (originally loose
scripts; the script names are kept in the tables below for provenance) into a
single CLI.

Portability.  ``matrix`` and ``phantom`` are self-contained: they take every
binary and corpus on the command line and run from any clone (the package is
imported from this checkout's ``src/``).  The legacy probe and memory modes
are local-only: they were written against one machine's fixed builds and are
kept because they reproduce the P1-4 trail, not because they are general
tools.  Their inputs resolve from the environment, and a missing input stops
the mode with the variable to set instead of a traceback:

    EDGE_DIAG_SCRATCH      scratch dir (debug targets, cj40 corpus, logs);
                           default /tmp/opencode
    EDGE_DIAG_BUILDS       prebuilt targets; default ~/fuzzing/builds
    EDGE_DIAG_CTX_TARGET   fuzzgoat ctx build; default $EDGE_DIAG_BUILDS/fuzzgoat_read
    EDGE_DIAG_DBG_TARGET   default $EDGE_DIAG_SCRATCH/fuzzgoat_dbg_dist
    EDGE_DIAG_TRACE_TARGET default $EDGE_DIAG_SCRATCH/fuzzgoat_dbg (fire-trace-build)
    EDGE_DIAG_CORPUS       input dir for the probe modes; default $EDGE_DIAG_SCRATCH/cj40
    EDGE_DIAG_LIBPAD       LD_PRELOAD pad library for mappad; default /tmp/libpad.so
    EDGE_DIAG_PROF_TARGET  memory-mode target; default $EDGE_DIAG_BUILDS/test_target_v3
    EDGE_DIAG_PROF_CORPUS  memory-mode corpus; default ~/fuzzing/prof_corpus

``python3 tools/corpus_fuzzgoat.py --out DIR`` builds a usable corpus and the
handover's reference-run recipe builds the ctx target.

Edge-coverage modes reproduce the P1-4 experiment trail (fuzzgoat, clang,
ASLR pinned via disable_aslr()):
    stored-ids       compare_binaries.py   (ctx vs dbg_dist, maps 8192/512)
    per-input-sweep  det_probe.py          (map sweep 512..65536 vs 65536)
    fresh-sweep      fresh_probe.py        (512 vs 8192 per input)
    collect-dump     table_dump.py         (collect-style header dump)
    full-table       full_table_dump4.py   (alloc 65536, view 512/8192)
    trace-stored     full_dump2.py         (dbg_dist + __AFL_EDGE_TRACE=1)
    view-mismatch    view_mismatch.py      (alloc 65536, view 512 vs 8192)
    mixed-map        mixed_map.py          (warmup/current-map interaction)
    env-layout       env_layout_probe.py   (env pad 200B, path_hash)
    env-pad          env_pad_sweep.py      (pad sweep 0..4096 at 512/8192)
    malloc-tunables  heap_probe.py         (MALLOC_* tunables vs id 209)
    mappad           pad_mb_probe.py       (needs $EDGE_DIAG_LIBPAD)
    strace-map       strace_probe.py       (needs strace; shmat/mmap addrs)
    gdb-guards       gdb_guard_capture.py  (needs gdb; __sanitizer_cov stream)
    fire-trace       fire_trace.py         (needs $EDGE_DIAG_TRACE_TARGET)
    fire-trace-build (build recipe for the ___AFL_EDGE_TRACE debug target)

Sub-command modes (deep analysis; each delegates to its own argparse parser):
    matrix          edge_matrix_analysis.py -- full 2-D (edge_id, hit-count)
                    matrix analysis (sections 1-8: axis, y-marginal, seed/edge
                    spectrum, GF(2) rank, integer relations, equivalence classes,
                    placement matrix).
    phantom         phantom_edge_probe.py -- F2 phantom-edge quantification
                    (per-input steady-state vs production runs, ownership, seed
                    weights).

Memory profiler modes drive the fuzzer (test_target_v3 / prof_corpus):
    mem-stats        memprof.py
    mem-native       native_prof.py
    mem-curve        curve.py
    mem-plateau      plateau.py
    mem-snapshot     diff_snaps.py
    mem-heap-attr    heap_attr.py
    mem-long         long_prof.py
    mem-typehist     typehist.py
    mem-smaps        smaps_prof.py
    mem-smaps2       smaps2.py
    mem-trim         trim.py
    mem-shm          shmcount.py
    op-caches        cache_probe.py
    run-sanity       runclean.py
"""

import argparse
import collections
import contextlib
import ctypes
import gc
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tracemalloc
from pathlib import Path

import numpy as np

SRCDIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRCDIR)

logging.disable(logging.INFO)

from fuzzer_tool.adapters import shm as shmmod  # noqa: E402
from fuzzer_tool.adapters.process import disable_aslr  # noqa: E402
from fuzzer_tool.adapters.shm import ShmCoverage  # noqa: E402
from fuzzer_tool.core.gini import gini as _gini  # noqa: E402
from fuzzer_tool.core.lattice import lll_reduce  # noqa: E402
from fuzzer_tool.services.fuzzer import Fuzzer  # noqa: E402

# Inputs of the local-only modes (see the module docstring).  Strings, not
# Paths: several modes build derived names by concatenation.
_ENV = os.environ.get
SCRATCH = _ENV("EDGE_DIAG_SCRATCH", "/tmp/opencode")
BUILDS = _ENV("EDGE_DIAG_BUILDS", str(Path.home() / "fuzzing" / "builds"))
CTX_TARGET = _ENV("EDGE_DIAG_CTX_TARGET", os.path.join(BUILDS, "fuzzgoat_read"))
DBG_TARGET = _ENV("EDGE_DIAG_DBG_TARGET", os.path.join(SCRATCH, "fuzzgoat_dbg_dist"))
EDGE_TRACE_TARGET = _ENV("EDGE_DIAG_TRACE_TARGET", os.path.join(SCRATCH, "fuzzgoat_dbg"))
CORPUS_DIR = _ENV("EDGE_DIAG_CORPUS", os.path.join(SCRATCH, "cj40"))
LIBPAD = _ENV("EDGE_DIAG_LIBPAD", "/tmp/libpad.so")
PROF_TARGET = _ENV("EDGE_DIAG_PROF_TARGET", os.path.join(BUILDS, "test_target_v3"))
PROF_CORPUS = _ENV("EDGE_DIAG_PROF_CORPUS", str(Path.home() / "fuzzing" / "prof_corpus"))
BASE_IDS = [
    13,
    17,
    19,
    23,
    33,
    35,
    47,
    53,
    57,
    59,
    61,
    81,
    87,
    129,
    131,
    135,
    193,
    207,
    213,
    215,
    217,
    219,
    223,
    249,
    251,
    273,
    393,
    455,
    471,
    557,
    651,
    827,
    851,
    857,
    935,
    945,
    979,
    993,
    2065,
    2069,
    2213,
    2301,
    2303,
    4231,
    4323,
    4335,
    6361,
]

FILES = sorted(glob.glob(os.path.join(CORPUS_DIR, "*")))

DEVNULL = open(os.devnull, "w")  # noqa: SIM115


# ── shared helpers ───────────────────────────────────────────────────────
def pin():
    assert disable_aslr(), "could not pin ASLR"


def exec_edge(target, path, cov, map_size, extra_env=None):
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(map_size))
    if extra_env:
        env.update(extra_env)
    cov.reset_edge_map()
    return subprocess.run([target, str(path)], env=env, capture_output=True, timeout=10)


def edge_ids(cov, n=None):
    arr = np.frombuffer(cov._map, dtype=shmmod._ENTRY_DTYPE, count=n or cov.num_entries)
    occ = np.flatnonzero(arr["edge_id"])
    return sorted(int(x) for x in arr["edge_id"][occ])


def build_fuzzer():
    os.makedirs(PROF_CORPUS, exist_ok=True)
    os.makedirs(PROF_CORPUS + "_crashes", exist_ok=True)
    return Fuzzer(
        target=PROF_TARGET,
        corpus_dir=PROF_CORPUS,
        crashes_dir=PROF_CORPUS + "_crashes",
        max_len=1024,
        timeout=1,
        mutations_per_input=4,
        use_coverage=True,
        seed=99,
        cmplog=False,
        stack_heartbeat=None,
        stats_file=None,
    )


def quiet_init(f):
    with contextlib.redirect_stdout(DEVNULL), contextlib.redirect_stderr(DEVNULL):
        f.run(iterations=0)


def quiet_run(f, iterations):
    with contextlib.redirect_stdout(DEVNULL), contextlib.redirect_stderr(DEVNULL):
        f.run(iterations=iterations)


def shm_counts():
    out = subprocess.run(["ipcs", "-m"], capture_output=True, text=True).stdout
    total = att = orph = dest = 0
    for line in out.splitlines()[3:]:
        p = line.split()
        if len(p) < 6 or not p[1].isdigit():
            continue
        total += 1
        n = int(p[5])
        if n == 0:
            orph += 1
        else:
            att += 1
        if "dest" in line:
            dest += 1
    return total, att, orph, dest


def rss_kb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


# ── P1-4 edge-coverage modes ─────────────────────────────────────────────
def stored_ids_mode(args):
    pin()
    for T in (args.target, args.second_target):
        if not os.path.exists(T):
            print("== missing", T)
            continue
        print("== ", T)
        for m in (8192, 512):
            cov = ShmCoverage(size=m)
            for gen in range(3):
                exec_edge(T, FILES[gen], cov, m)
            eids = edge_ids(cov)
            print("  map", m, "live_slots", len(eids), "ids", eids)
            cov.cleanup()


def per_input_sweep_mode(args):
    pin()
    for i in args.inputs:
        f = FILES[i]
        per_map = {}
        for m in (65536, 8192, 4096, 1024, 512):
            cov = ShmCoverage(size=m)
            exec_edge(CTX_TARGET, f, cov, m)
            per_map[m] = set(cov._scan_with_positions()[1].tolist())
            cov.cleanup()
        for m in (512, 1024, 4096, 8192, 65536):
            base = per_map[65536]
            only_m = sorted(per_map[m] - base)
            only_65 = sorted(base - per_map[m])
            print(
                f"input {i} map {m:>5} n={len(per_map[m]):>3} "
                f"only@{m}:{len(only_m)} only@65536:{len(only_65)} "
                f"same:{len(per_map[m] & base)}"
            )


def fresh_sweep_mode(args):
    if args.pin:
        pin()
    for i in args.inputs:
        s512 = fresh_one(512, i)
        s8192 = fresh_one(8192, i)
        print(
            f"input {i} 512={len(s512)} 8192={len(s8192)} "
            f"only512={sorted(s512 - s8192)} only8192={sorted(s8192 - s512)}"
        )


def fresh_one(m, i):
    cov = ShmCoverage(size=m)
    try:
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m))
        with contextlib.suppress(subprocess.TimeoutExpired):
            subprocess.run([CTX_TARGET, str(FILES[i])], env=env, capture_output=True, timeout=10)
        return set(cov._scan_with_positions()[1].tolist())
    finally:
        cov.cleanup()


def collect_dump_mode(args):
    pin()
    for m in args.maps:
        cov = ShmCoverage(size=m)
        try:

            def one(path, idx=None, cov=cov, m=m):  # noqa: B023
                cov.reset_edge_map()
                env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m))
                with contextlib.suppress(subprocess.TimeoutExpired):
                    subprocess.run(
                        [CTX_TARGET, str(path)], env=env, capture_output=True, timeout=10
                    )

            one(FILES[0])
            one(FILES[0])
            one(FILES[args.input])
        except Exception:
            cov.cleanup()
            raise
        d = run_collect_style(cov)
        print(
            f"=== map {m}: gen {d['gen']} edge_count {d['edge_count']} "
            f"path_hash {d['path_hash']} dropped {d['dropped']} slots {d['live_n']}"
        )
        print("   all stored ids:", sorted(e[1] for e in d["entries"]))
        cov.cleanup()


def run_collect_style(cov):
    arr = np.frombuffer(cov._map, dtype=shmmod._ENTRY_DTYPE, count=cov.num_entries)
    occ = np.flatnonzero(arr["edge_id"])
    entries = [
        (
            int(p),
            int(arr["edge_id"][p]),
            int(arr["count"][p] >> 24),
            int(arr["count"][p] & 0xFFFFFF),
        )
        for p in occ
    ]
    return {
        "gen": cov.read_generation(),
        "edge_count": cov.read_edge_count(),
        "path_hash": cov.read_path_hash(),
        "dropped": cov.read_dropped_edges(),
        "live_n": len(occ),
        "entries": entries,
    }


def full_table_mode(args):
    pin()
    for view in (512, 8192):
        cov = ShmCoverage(size=65536)
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(view))
        for g in range(3):
            cov.reset_edge_map()
            subprocess.run([CTX_TARGET, str(FILES[g])], env=env, capture_output=True, timeout=10)
        arr = np.frombuffer(cov._map, dtype=shmmod._ENTRY_DTYPE, count=65536)
        occ = np.flatnonzero(arr["edge_id"])
        rows = [(int(i), int(arr["edge_id"][i]), int(arr["count"][i])) for i in occ]
        print(f"view={view}  occupied={len(rows)}")
        print("  id209 slots:", [(i, e, c) for (i, e, c) in rows if e == 209])
        print("  ids:", sorted(e for _, e, _ in rows)[:60])
        print("  slots 505..530:", [(i, e) for i, e, c in rows if 505 <= i <= 530])
        cov.cleanup()


def trace_stored_mode(args):
    pin()
    T = DBG_TARGET
    if not os.path.exists(T):
        print("missing", T)
        return
    for m in (8192, 512):
        cov = ShmCoverage(size=m)
        env = dict(
            os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m), __AFL_EDGE_TRACE="1"
        )

        def one(p, cov=cov, env=env):  # noqa: B023
            cov.reset_edge_map()
            subprocess.run([T, str(p)], env=env, capture_output=True, timeout=10)

        one(FILES[0])
        one(FILES[0])
        r = subprocess.run([T, str(FILES[1])], env=env, capture_output=True, timeout=10)
        eids = edge_ids(cov)
        tr = r.stderr.decode().splitlines()
        print("map", m, "stored_n", len(eids), "trace_n", len(tr))
        print("  stored", eids)
        print("  sample trace:", tr[:3])
        cov.cleanup()


def view_mismatch_mode(args):
    pin()
    a = run_vm(65536, 512, 512)
    b = run_vm(65536, 8192, 8192)
    print("alloc65536 view512 scan512 n", len(a), "has209:", 209 in a)
    print("alloc65536 view8192 scan8192 n", len(b), "has209:", 209 in b)
    print("extra in view512:", sorted(set(a) - set(b))[:6])
    print("missing in view512:", sorted(set(b) - set(a))[:6])


def run_vm(alloc_entries, view_entries, scan_entries):
    cov = ShmCoverage(size=alloc_entries)
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(view_entries))
    for g in range(3):
        cov.reset_edge_map()
        r = subprocess.run([CTX_TARGET, str(FILES[g])], env=env, capture_output=True, timeout=10)
        if r.returncode != 0:
            print("  child rc", r.returncode, r.stderr.decode()[:60])
    arr = np.frombuffer(cov._map, dtype=shmmod._ENTRY_DTYPE, count=scan_entries)
    occ = np.flatnonzero(arr["edge_id"])
    out = sorted(int(x) for x in arr["edge_id"][occ])
    cov.cleanup()
    return out


def mixed_map_mode(args):
    pin()
    for name, pairs, alloc in [
        ("all 512         ", [("512", 0), ("512", 0), ("512", 1)], 65536),
        ("warmup@512,in1@512 ", [("512", 0), ("512", 1)], 65536),
        ("warmup@512,in1@8192", [("512", 0), ("8192", 1)], 65536),
        ("all 8192        ", [("8192", 0), ("8192", 0), ("8192", 1)], 65536),
        ("warmup@8192,in1@512", [("8192", 0), ("512", 1)], 65536),
    ]:
        cov = ShmCoverage(size=alloc)
        for m, idx in pairs:
            env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m))
            cov.reset_edge_map()
            subprocess.run([CTX_TARGET, str(FILES[idx])], env=env, capture_output=True, timeout=10)
        ids = edge_ids(cov)
        cov.cleanup()
        print(f"{name} n={len(ids)} has209={209 in ids} ids209={[e for e in ids if e == 209]}")
        print("   small band:", [e for e in ids if e < 260])


def env_layout_mode(args):
    pin()
    a, pa = dump_state(8192)
    b, pb = dump_state(8192, {"AFL_DUMMY_PAD": "x" * 200})
    c, _ = dump_state(512)
    print("8192  baseline :", a)
    print("8192  +200B env:", b, " SAME" if a == b else " DIFFERS")
    print("path_hash", pa, "vs", pb, "same" if pa == pb else "DIFFERS")
    print("512            :", c)
    print("8192 vs 512 same:", a == c)


def dump_state(map_size, extra_env=None):
    cov = ShmCoverage(size=map_size)
    try:
        base = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(map_size))
        if extra_env:
            base.update(extra_env)

        def one(path):
            cov.reset_edge_map()
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run([CTX_TARGET, str(path)], env=base, capture_output=True, timeout=10)

        one(FILES[0])
        one(FILES[0])
        one(FILES[1])
        return edge_ids(cov), cov.read_path_hash()
    finally:
        cov.cleanup()


def env_pad_mode(args):
    pin()
    for pad in args.pads:
        ids512, r = pickup(512, pad)
        ids8192, _ = pickup(8192, pad)
        extra = sorted(set(ids512) - set(ids8192))
        missing = sorted(set(ids8192) - set(ids512))
        print(
            f"pad={pad:5d}  512_n={len(ids512):3d} 8192_n={len(ids8192):3d} "
            f"extra512={extra} missing={missing} rc={r.returncode} stderr={r.stderr.decode()[:40]!r}"
        )


def pickup(m, pad):
    cov = ShmCoverage(size=m)
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m), PAD="q" * pad)
    r = None
    for gen in range(3):
        cov.reset_edge_map()
        r = subprocess.run([CTX_TARGET, str(FILES[gen])], env=env, capture_output=True, timeout=10)
    out = edge_ids(cov)
    cov.cleanup()
    return out, r


def malloc_tunables_mode(args):
    pin()
    base512 = ids_at(512, {})
    base8192 = ids_at(8192, {})
    print("512 base   :", base512)
    print("8192 base  :", base8192)
    print("has 209:", 209 in base512, 209 in base8192)
    for name, extra in [
        ("MALLOC_ARENA_MAX=1", {"MALLOC_ARENA_MAX": "1"}),
        ("MALLOC_ARENA_MAX=2", {"MALLOC_ARENA_MAX": "2"}),
        ("MALLOC_MMAP_THRESHOLD_=4096", {"MALLOC_MMAP_THRESHOLD_": "4096"}),
        ("MALLOC_TRIM_THRESHOLD_=4096", {"MALLOC_TRIM_THRESHOLD_": "4096"}),
        ("GLIBC_TUNABLES=glibc.malloc.mxfast=32", {"GLIBC_TUNABLES": "glibc.malloc.mxfast=32"}),
    ]:
        ids = ids_at(512, extra)
        print(f"{name:38s} 209->{209 in ids}  512_n={len(ids)}")
    ids8192x = ids_at(8192, {"MALLOC_ARENA_MAX": "1"})
    print("MALLOC_ARENA_MAX=1 @8192 has 209:", 209 in ids8192x)


def ids_at(m, extra):
    cov = ShmCoverage(size=m)
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m), **extra)
    for g in range(3):
        cov.reset_edge_map()
        subprocess.run([CTX_TARGET, str(FILES[g])], env=env, capture_output=True, timeout=10)
    out = edge_ids(cov)
    cov.cleanup()
    return out


def mappad_mode(args):
    pin()
    if not os.path.exists(LIBPAD):
        print(
            f"missing {LIBPAD} (a local LD_PRELOAD pad library; set "
            "EDGE_DIAG_LIBPAD -- no recipe is kept in the tree)"
        )
    for pad in (0, 2, 8, 64):
        for m in (512, 8192):
            cov = ShmCoverage(size=m)
            env = dict(
                os.environ,
                __AFL_SHM_ID=str(cov.shm_id),
                AFL_MAP_SIZE=str(m),
                LD_PRELOAD=LIBPAD,
                __PAD_MB=str(pad),
            )
            for g in range(3):
                cov.reset_edge_map()
                subprocess.run(
                    [CTX_TARGET, str(FILES[g])], env=env, capture_output=True, timeout=10
                )
            ids = edge_ids(cov)
            cov.cleanup()
            tag = "HAS209" if 209 in ids else "no209 "
            extra = sorted(set(ids) - set(BASE_IDS))[:4]
            print(f"pad={pad:3d}MB map={m:5d} {tag} n={len(ids)} extra_vs_base={extra}")


def strace_map_mode(args):
    def run(m):
        cov = ShmCoverage(size=m)
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m))
        for _i in range(3):
            cov.reset_edge_map()
        log = os.path.join(SCRATCH, f"strace_{m}.log")
        r = subprocess.run(
            ["strace", "-f", "-e", "trace=shmat,mmap,brk", "-o", log, CTX_TARGET, str(FILES[1])],
            env=env,
            capture_output=True,
            timeout=30,
        )
        cov.cleanup()
        addrs = {}
        with open(log) as fh:
            for line in fh:
                m2 = re.search(r"shmat\((\d+).*= (0x[0-9a-f]+|-1)", line)
                if m2:
                    addrs[int(m2.group(1))] = m2.group(2)
        mm = []
        with open(log) as fh:
            for line in fh:
                if "mmap" in line and "= " in line:
                    mm.append(line.strip()[:140])
        return r.returncode, addrs, (len(mm), mm[-4:])

    rc512, a512, ex512 = run(512)
    rc8192, a8192, ex8192 = run(8192)
    print("rc", rc512, rc8192)
    print("shmat addresses split by shm id:")
    print(" 512:", a512)
    print(" 8192:", a8192)
    print("mmap line counts 512/8192:", ex512[0], ex8192[0])
    print("last mmaps 512:", *ex512[1], sep="\n    ")
    print("last mmaps 8192:", *ex8192[1], sep="\n    ")


def gdb_guards_mode(args):
    def gdb_capture(m):
        cov = ShmCoverage(size=m)
        for _g in range(3):
            cov.reset_edge_map()
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(m))
        script = f"""
set pagination off
set disable-randomization on
break __sanitizer_cov_trace_pc_guard
commands
 silent
 printf "%lu\\n", (unsigned long)*($rdi)
 continue
end
run {FILES[1]}
quit
"""
        gdb_cmd = os.path.join(SCRATCH, "gdb_cmd.txt")
        with open(gdb_cmd, "w") as _fh:
            _fh.write(script)
        r = subprocess.run(
            ["gdb", "-batch", "-x", gdb_cmd, CTX_TARGET],
            env=env,
            capture_output=True,
            timeout=120,
        )
        ids = [line for line in r.stdout.decode().splitlines() if line.strip().isdigit()]
        cov.cleanup()
        return r.returncode, ids

    rc512, ids512 = gdb_capture(512)
    rc8192, ids8192 = gdb_capture(8192)
    print("rc", rc512, rc8192, "len", len(ids512), len(ids8192))
    for i, (a, b) in enumerate(zip(ids512, ids8192, strict=False)):
        if a != b:
            print(f"first guard divergence at fire {i}:", a, "vs", b)
            print("local ctx:", ids512[max(0, i - 4) : i + 3])
            print("          ", ids8192[max(0, i - 4) : i + 3])
            break
    else:
        print("guard streams identical, len", len(ids512))


def fire_trace_mode(args):
    T = EDGE_TRACE_TARGET
    if not os.path.exists(T):
        print(f"missing {T} (built from /tmp/afl_shim_dbg.c); fire-trace unavailable")
        return
    pin()
    a = traced_input(8192, 1)
    b = traced_input(512, 1)
    print("trace lines 8192:", len(a), " 512:", len(b))
    for i, (la, lb) in enumerate(zip(a, b, strict=False)):
        if la != lb:
            print(f"first divergence at fire {i}:")
            print("  8192:", la.split("\t"))
            print("  512 :", lb.split("\t"))
            print("  context 8192:", [x.split("\t")[2] for x in a[max(0, i - 3) : i + 3]])
            print("  context 512 :", [x.split("\t")[2] for x in b[max(0, i - 3) : i + 3]])
            break
    else:
        print("traces identical")


def traced_input(map_size, idx):
    cov = ShmCoverage(size=map_size)
    try:
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(map_size),
            __AFL_EDGE_TRACE="1",
        )

        def one(path):
            cov.reset_edge_map()
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run(
                    [EDGE_TRACE_TARGET, str(path)], env=env, capture_output=True, timeout=10
                )

        one(FILES[0])
        one(FILES[0])
        r = subprocess.run(
            [EDGE_TRACE_TARGET, str(FILES[idx])], env=env, capture_output=True, timeout=10
        )
        return r.stderr.decode().splitlines()
    finally:
        cov.cleanup()


def fire_trace_build(args):
    logging.disable(logging.NOTSET)
    print("recipe (documented in /tmp/afl_shim_dbg.c diff): add an env-gated")
    print("write(2) dump of cur_loc/prev/ctx/edge_id in __afl_map_edge guarded by")
    print("__AFL_EDGE_TRACE=1, build the shim into the wrapper TU only:")
    print("  clang -fsanitize-coverage=trace-pc-guard -O2 -g -fno-omit-frame-pointer \\")
    print("        -I$FUZZ_VENDOR_ROOT/fuzzgoat -c fuzzgoat.c -o /tmp/fuzzgoat_dbg2.o")
    print("  clang -O2 -g -fno-omit-frame-pointer -include /tmp/afl_shim_dbg.c \\")
    print("        -I$FUZZ_VENDOR_ROOT/fuzzgoat targets/fuzzgoat_read.c /tmp/fuzzgoat_dbg2.o \\")
    print(f"        -o {EDGE_TRACE_TARGET} -lm")
    print("the shim's own -D__AFL_TRACE_FIRES=1 fire log (P0-1) covers the same")
    print("ground without a patched shim; prefer it for new work.")


# ── memory profiler modes ────────────────────────────────────────────────
def mem_stats_mode(args):
    tracemalloc.start()

    def sample(label, snap=None, prev_snap=None):
        st = shm_counts()
        cur = {
            "rss_kb": rss_kb(),
            "fds": len(os.listdir("/proc/self/fd")),
            "threads": threading.active_count(),
            "shm_total": st[0],
            "shm_nattch0": st[2],
            "shm_attached": st[1],
            "shm_dest": st[3],
        }
        heap = growth = 0
        if snap is not None and prev_snap is not None:
            heap = sum(s.size for s in snap.statistics("filename"))
            growth = sum(s.size_diff for s in snap.compare_to(prev_snap, "filename"))
        print(
            f"{label:>10s}  rss={cur['rss_kb']:>8d}KB  fds={cur['fds']:>3d}  "
            f"thr={cur['threads']:>2d}  shm={cur['shm_total']:>3d} "
            f"(attached={cur['shm_attached']}, nattch0={cur['shm_nattch0']}, "
            f"dest={cur['shm_dest']})  heap={heap:>10,d}B  grow={growth:>10,d}B"
        )

    sample("baseline")
    f = build_fuzzer()
    gc.collect()
    snap_base = tracemalloc.take_snapshot()
    cur_snap = snap_base
    sample("init", snap_base, snap_base)
    steps = 5
    per = max(1, args.iters // steps)
    for step in range(1, steps + 1):
        f.run(iterations=per)
        gc.collect()
        prev_snap = cur_snap
        cur_snap = tracemalloc.take_snapshot()
        sample(f"after{step}", cur_snap, prev_snap)
    print("\n== top 25 live allocations (filename) at end of loop ==")
    for st in cur_snap.statistics("filename")[:25]:
        print(f"{st.size:>12,d}  {st.count:>8d}  {st.traceback}")
    f = None
    gc.collect()
    final_snap = tracemalloc.take_snapshot()
    sample("afterdrop", final_snap, cur_snap)
    f2 = build_fuzzer()
    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    sample("sess2", snap2, final_snap)
    del f2
    gc.collect()
    snap3 = tracemalloc.take_snapshot()
    sample("afterdrop2", snap3, snap2)
    print("\n== heap growth retained after stop (top 30) ==")
    for st in final_snap.compare_to(snap_base, "filename")[:30]:
        if st.size_diff > 0:
            print(f"{st.size_diff:>+12,d}  {st.traceback}")
    print("\n== __AFL* env present ==")
    print({k: v for k, v in os.environ.items() if "__AFL" in k})


def mem_native_mode(args):
    import psutil

    def smaps():
        p = psutil.Process()
        mi = p.memory_full_info()
        priv_dirty = 0
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Private_Dirty"):
                    priv_dirty = int(line.split()[1])
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
        return rss, mi.uss, priv_dirty

    def children_rss():
        me = psutil.Process()
        tot = 0
        n = 0
        for c in me.children(recursive=True):
            try:
                tot += c.memory_info().rss
                n += 1
            except psutil.NoSuchProcess:
                pass
        return tot, n

    f = build_fuzzer()
    gc.collect()
    tracemalloc.start()
    rss, priv, pd = smaps()
    crs, cn = children_rss()
    print(
        f"{'exec':>7} {'rss_kb':>9} {'uss_kb':>9} {'privD':>9} {'child_rs':>9} {'child':>5} {'heap_grow':>10}"
    )
    base = tracemalloc.take_snapshot()
    prev = base
    per = 100
    for i in range(1, 9):
        quiet_run(f, per)
        gc.collect()
        rss, priv, pd = smaps()
        crs, cn = children_rss()
        s = tracemalloc.take_snapshot()
        g = sum(x.size_diff for x in s.compare_to(prev, "filename"))
        print(f"{i * per:>7} {rss:>9} {priv:>9} {pd:>9} {crs:>9} {cn:>5} {g:>10}")
        prev = s
    fs = getattr(f, "_forkserver", None)
    if fs is not None:
        with contextlib.redirect_stdout(DEVNULL), contextlib.redirect_stderr(DEVNULL):
            fs.stop()
    del f
    gc.collect()
    rss, priv, pd = smaps()
    crs, cn = children_rss()
    print(f"{'stop':>7} {rss:>9} {priv:>9} {pd:>9} {crs:>9} {cn:>5} {'--':>10}")


def mem_curve_mode(args):
    import psutil

    f = build_fuzzer()
    quiet_run(f, 6000)
    me = psutil.Process()
    print(f"{'exec':>7} {'rss':>9} {'uss':>9} {'anonPSS':>9} {'pss':>9}")
    for i in range(1, 13):
        gc.collect()
        sm = me.memory_full_info()
        quiet_run(f, 2000)
        gc.collect()
        print(f"{i * 2000:>7} {sm.rss:>9} {sm.uss:>9} {sm.pss:>9} {sm.swap:>9}")


def mem_plateau_mode(args):
    import psutil

    f = build_fuzzer()
    quiet_run(f, 5000)
    me = psutil.Process()
    gc.collect()
    r0 = me.memory_full_info().uss
    o0 = shm_counts_plateau()
    quiet_run(f, 8000)
    gc.collect()
    r1 = me.memory_full_info().uss
    o1 = shm_counts_plateau()
    print(
        f"post-warmup 8000 execs: uss {r0} -> {r1} KB  delta {r1 - r0} KB = {(r1 - r0) / 8:.2f} KB/exec"
    )
    print(f"orphan shm before {o0} after {o1}")


def shm_counts_plateau():
    o = 0
    with open("/proc/sysvipc/shm") as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) < 8:
                continue
            nattch = int(p[5])
            if nattch == 0:
                o += 1
    return o


def mem_snapshot_mode(args):
    f = build_fuzzer()
    tracemalloc.start()
    gc.collect()
    base = tracemalloc.take_snapshot()
    prev = base
    print(f"{'phase':>6} {'rss_kb':>8} {'heap':>10} {'grow':>10}  top-growth (file: bytes)")
    per = 100
    for i in range(1, 9):
        f.run(iterations=per)
        gc.collect()
        s = tracemalloc.take_snapshot()
        diff = s.compare_to(prev, "filename")
        diff.sort(key=lambda st: st.size_diff, reverse=True)
        top = " ".join(
            f"{st.traceback}:{st.size_diff / 1024:.0f}K" for st in diff[:5] if st.size_diff > 0
        )
        print(
            f"{i * per:>6} {rss_kb():>8} {sum(x.size for x in s.statistics('lineno')) / 1024:>9.0f}K "
            f"{sum(x.size_diff for x in diff) / 1024:>9.0f}K  {top}"
        )
        prev = s
    del f


def mem_heap_attr_mode(args):
    f = build_fuzzer()
    quiet_run(f, 500)
    gc.collect()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    quiet_run(f, 2000)
    gc.collect()
    snap1 = tracemalloc.take_snapshot()
    diffs = sorted(snap1.compare_to(snap0, "lineno"), key=lambda s: -s.size_diff)
    print(f"{'size_delta':>10} {'count_delta':>10}  traceback")
    for st in diffs[:20]:
        if st.size_diff <= 0:
            continue
        print(f"{st.size_diff:>10} {st.count_diff:>10}  {st.traceback}")


def mem_long_mode(args):
    import psutil

    f = build_fuzzer()

    def stats():
        me = psutil.Process()
        u = me.memory_full_info().uss
        r = me.memory_info().rss
        s = shm_counts()
        return r, u, s[0], s[1], s[2]

    gc.collect()
    tracemalloc.start()
    r0, u0, t0, a0, z0 = stats()
    p0 = tracemalloc.take_snapshot()
    print(
        f"{'exec':>6} {'rss_kb':>8} {'uss_kb':>8} {'shm':>4} {'attached':>8} {'orphan':>6} {'heapGrow':>9}"
    )
    prev = p0
    per = 500
    for i in range(1, 9):
        quiet_run(f, per)
        gc.collect()
        r, u, t, a, z = stats()
        s = tracemalloc.take_snapshot()
        g = sum(x.size_diff for x in s.compare_to(prev, "filename"))
        print(f"{i * per:>6} {r:>8} {u:>8} {t:>4} {a:>8} {z:>6} {g:>9}")
        prev = s
    t0 = tracemalloc.take_snapshot()
    print("\nnative RSS delta total:", r - r0, "KB over", 8 * per, "execs")
    print("heap delta total:", sum(x.size_diff for x in t0.compare_to(p0, "filename")))


def mem_typehist_mode(args):
    f = build_fuzzer()

    def hist(execs, block):
        gc.collect()
        c0 = collections.Counter(type(o) for o in gc.get_objects())
        quiet_run(f, execs)
        gc.collect()
        c1 = collections.Counter(type(o) for o in gc.get_objects())
        rows = []
        for t, n0 in c0.items():
            d = c1.get(t, 0) - n0
            if d > 0:
                rows.append((d, str(t)))
        rows.sort(reverse=True)
        print(f"--- after warmup, block {block}: {execs} execs, top +retained by type ---")
        for cnt, tn in rows[:20]:
            print(f"{cnt:>8}  {tn}")

    quiet_run(f, 5000)
    for block in (1, 2, 3):
        hist(2000, block)


def smaps_breakdown():
    agg = collections.defaultdict(lambda: [0, 0, 0])
    path = ""
    with open("/proc/self/smaps") as fh:
        for line in fh:
            if "-" in line and line[0] != " ":
                path = ""
            elif line.startswith("Name:"):
                path = line.split(":", 1)[1].strip()
            elif line.startswith("Rss:"):
                agg[path][0] += int(line.split()[1])
            elif line.startswith("Private_Dirty:"):
                agg[path][2] += int(line.split()[1])
    return agg


def mem_smaps_mode(args):
    f = build_fuzzer()
    gc.collect()
    tracemalloc.start()
    a0 = smaps_breakdown()
    snap0 = tracemalloc.take_snapshot()
    quiet_run(f, 2000)
    gc.collect()
    a1 = smaps_breakdown()
    snap1 = tracemalloc.take_snapshot()
    keys = set(a0) | set(a1)
    rows = []
    for k in keys:
        d = a1.get(k, [0, 0, 0])[0] - a0.get(k, [0, 0, 0])[0]
        pd = a1.get(k, [0, 0, 0])[2] - a0.get(k, [0, 0, 0])[2]
        if d or pd:
            rows.append((d, k, pd))
    rows.sort(key=lambda x: -x[0])
    print(f"{'rss_delta':>10} {'privD_delta':>10}  mapping")
    for d, k, pd in rows[:25]:
        print(f"{d:>10} {pd:>10}  {k!r}")
    g = sum(x.size_diff for x in snap1.compare_to(snap0, "filename"))
    print("tracemalloc heap delta:", g)
    print("\ntop heap growth by lineno:")
    for st in sorted(snap1.compare_to(snap0, "filename"), key=lambda s: -s.size_diff)[:15]:
        if st.size_diff > 0:
            print(f"{st.size_diff:>10}  {st.traceback}")


def mem_smaps2_mode(args):
    import resource

    f = build_fuzzer()

    def anon_breakdown():
        agg = collections.defaultdict(int)
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if "-" in line and line[0] != " ":
                    path = ""
                elif line.startswith("Name:"):
                    path = line.split(":", 1)[1].strip()
                elif line.startswith("Rss:"):
                    rss = int(line.split()[1])
                    if path == "":
                        agg["ANON"] += rss
                    else:
                        agg[path] += rss
        return agg

    quiet_run(f, 5000)
    gc.collect()
    a0 = anon_breakdown()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    quiet_run(f, 2000)
    gc.collect()
    a1 = anon_breakdown()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    print("post-warmup 2000-exec block:")
    for k in sorted(set(a0) | set(a1), key=lambda x: -(a1.get(x, 0) - a0.get(x, 0))):
        d = a1.get(k, 0) - a0.get(k, 0)
        if d:
            print(f"{d:>10}  {k}")
    print("maxrss growth KB:", r1.ru_maxrss - r0.ru_maxrss)


def mem_trim_mode(args):
    import psutil

    libc = ctypes.CDLL("libc.so.6")
    f = build_fuzzer()
    me = psutil.Process()

    def show(tag):
        gc.collect()
        info = me.memory_full_info()
        print(f"{tag}: rss={info.rss / 1048576:.1f}MB uss={info.uss / 1048576:.1f}MB")

    quiet_run(f, 6000)
    show("after 6000 warmup")
    quiet_run(f, 20000)
    show("after +20000 execs")
    r = libc.malloc_trim(0)
    print("malloc_trim(0) ->", r)
    show("after malloc_trim")


def mem_shm_mode(args):
    out = subprocess.run(["ipcs", "-m"], capture_output=True, text=True).stdout
    total = att = orph = dest = 0
    for line in out.splitlines()[3:]:
        p = line.split()
        if len(p) < 5:
            continue
        total += 1
        n = int(p[4])
        if n == 0:
            orph += 1
        if "dest" in line:
            dest += 1
        if n > 0:
            att += 1
    print((total, att, orph, dest))


def op_caches_mode(args):
    logging.disable(logging.NOTSET)
    from fuzzer_tool.core.mutations import fractal_voronoi, perlin_noise

    f = build_fuzzer()
    print(
        f"{'exec':>6} {'pnc':>5} {'grad':>6} {'site':>6} {'near':>6} {'root':>6} {'bound':>6} {'rhash':>6} {'plan':>5} {'planMB':>7}"
    )
    per = 500
    for i in range(1, 7):
        quiet_run(f, per)
        gc.collect()
        pnl = perlin_noise._gradient.cache_info()
        fl = (
            fractal_voronoi._site.cache_info(),
            fractal_voronoi._nearest_site.cache_info(),
            fractal_voronoi._root.cache_info(),
        )
        plbytes = sum(sys.getsizeof(t) for t in fractal_voronoi._plan_cache.values())
        print(
            f"{i * per:>6} {len(perlin_noise._noise_cache):>5} {pnl.currsize:>6} {fl[0].currsize:>6} {fl[1].currsize:>6} {fl[2].currsize:>6} {len(fractal_voronoi._boundary_cache):>6} {len(fractal_voronoi._root_hash_cache):>6} {len(fractal_voronoi._plan_cache):>5} {plbytes / 1048576:>7}"
        )


def run_sanity_mode(args):
    f = build_fuzzer()
    quiet_run(f, 1000)
    print("run done")


import enum  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.count_class import classify_single  # noqa: E402

COUNT_MASK = 0xFFFFFF  # the top byte of the SHM count field is the generation tag
MIN_FLAKY_REPEATS = 2  # one repeat has no variance to measure
DEFAULT_BOOTSTRAP = 200  # --all-offline resamples when --bootstrap is not given


# ── Collection ────────────────────────────────────────────────────────


def _run_one(target: Path, cov: ShmCoverage, path: Path, timeout: float):
    """Execute *target* on *path* and return its live (positions, ids, counts)."""
    cov.reset_edge_map()
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(cov.num_entries))
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run([str(target), str(path)], env=env, capture_output=True, timeout=timeout)
    positions, ids, counts = cov._scan_with_positions()
    return (
        positions.astype(np.int64),
        ids.astype(np.int64),
        (counts & COUNT_MASK).astype(np.int64),
    )


def collect(target: Path, inputs: list[Path], map_size: int, timeout: float):
    """Run every input once, returning one (positions, ids, counts) triple per execution.

    The first execution against a *clean* table does not produce the same
    ids as every later execution against a generation-reset one -- measured,
    see the warm-up note in ``measure_stability``.  One discarded warm-up run
    puts the whole collection in the same regime the fuzzer spends its
    campaign in, instead of leaving input 0 in a regime of its own.
    """
    cov = ShmCoverage(size=map_size)
    try:
        _run_one(target, cov, inputs[0], timeout)  # warm-up, discarded
        return [_run_one(target, cov, p, timeout) for p in inputs]
    finally:
        cov.cleanup()


def collect_repeats(target: Path, inputs: list[Path], repeats: int, map_size: int, timeout: float):
    """Run every input *repeats* times: per input, a list of (ids, counts).

    One discarded warm-up first, as in ``collect``, so every repeat runs in the
    steady-state regime the campaign lives in.
    """
    cov = ShmCoverage(size=map_size)
    try:
        _run_one(target, cov, inputs[0], timeout)
        return [[_run_one(target, cov, p, timeout)[1:] for _ in range(repeats)] for p in inputs]
    finally:
        cov.cleanup()


def measure_stability(target: Path, path: Path, repeats: int, map_size: int, timeout: float):
    """Jaccard of the edge-id sets across *repeats* executions of one input.

    1.0 means edge identity is reproducible across processes.  Anything less
    with a deterministic target means the ids themselves are moving, which
    invalidates every per-edge statistic downstream -- the headline case
    being a PIE target under ASLR with ``__AFL_CTX_SENSITIVE=1``, where the
    context term is a hash of a raw return address and therefore differs in
    every process.

    Two regimes are reported separately because they have different causes.
    ``jaccard`` covers the steady state, after one warm-up execution has
    populated the table; ``first_exec_matches`` says whether the very first
    execution against a *clean* table agreed with it.  Measured on fuzzgoat
    with ASLR off: steady state is exact, and the first execution differs --
    same edge count and same total fires, different rolling path hash, and
    under ``__AFL_CTX_SENSITIVE=1`` a handful of different ids.
    """
    cov = ShmCoverage(size=map_size)
    try:
        first = frozenset(_run_one(target, cov, path, timeout)[1].tolist())
        sets = [frozenset(_run_one(target, cov, path, timeout)[1].tolist()) for _ in range(repeats)]
    finally:
        cov.cleanup()
    union = set().union(*sets)
    inter = set(sets[0]).intersection(*sets[1:]) if len(sets) > 1 else set(sets[0])
    return {
        "repeats": repeats,
        "sizes": [len(s) for s in sets],
        "union": len(union),
        "intersection": len(inter),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "first_exec_matches": first == sets[0],
        "first_exec_only": len(first - union),
    }


def save_runs(path: Path, runs, map_size: int | None = None, sizes=None) -> None:
    """Store the ragged per-execution columns as three flat arrays.

    *sizes* (input bytes per execution) rides along for ``--length-confound``.
    """
    offsets = np.cumsum([0] + [len(p) for p, _, _ in runs])
    extra = {} if sizes is None else {"sizes": np.asarray(sizes, dtype=np.int64)}
    np.savez_compressed(
        path,
        positions=np.concatenate([p for p, _, _ in runs]) if runs else np.empty(0, np.int64),
        ids=np.concatenate([i for _, i, _ in runs]) if runs else np.empty(0, np.int64),
        counts=np.concatenate([c for _, _, c in runs]) if runs else np.empty(0, np.int64),
        offsets=offsets,
        map_size=map_size or 0,
        **extra,
    )


def load_sizes(path: Path):
    """Input sizes a collection was saved with, or None (older collections)."""
    with np.load(path) as z:
        return z.get("sizes", None)


def saved_map_size(path: Path):
    """Table size a collection was recorded with, if the file carries it."""
    with np.load(path) as z:
        size = int(z["map_size"]) if "map_size" in z else 0
    return size if size else None


def load_runs(path: Path):
    z = np.load(path)
    off = z["offsets"]
    runs = [
        (z["ids"][off[i] : off[i + 1]], z["counts"][off[i] : off[i + 1]])
        for i in range(len(off) - 1)
    ]
    if "positions" in z:
        return [
            (z["positions"][off[i] : off[i + 1]], ids, counts)
            for i, (ids, counts) in enumerate(runs)
        ]
    return runs


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate(runs):
    """Collapse per-execution columns into the per-edge vectors under study.

    ``total`` mirrors ``EdgeTracker._global_edge_hits`` (execution volume),
    ``owners`` mirrors ``_edge_owner_count`` (incidence), ``first_seen``
    mirrors ``_edge_first_seen`` and ``peak`` mirrors ``_max_counts``.  The
    point of returning all four is that the first is the one most easily
    mistaken for the others.
    """
    total: Counter[int] = Counter()
    owners: Counter[int] = Counter()
    peak: dict[int, int] = {}
    first: dict[int, int] = {}
    for idx, (ids, counts) in enumerate(runs):
        for edge, count in zip(ids.tolist(), counts.tolist(), strict=True):
            total[edge] += count
            owners[edge] += 1
            if count > peak.get(edge, 0):
                peak[edge] = count
            first.setdefault(edge, idx)
    ids = np.array(sorted(total), dtype=np.int64)
    return {
        "ids": ids,
        "total": np.array([total[e] for e in ids], dtype=float),
        "owners": np.array([owners[e] for e in ids], dtype=float),
        "peak": np.array([peak[e] for e in ids], dtype=float),
        "first_seen": np.array([first[e] for e in ids], dtype=float),
    }


# ── Statistics (numpy only; scipy is not a dependency) ────────────────


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum() / denom) if denom else 0.0


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks, so ties (very common in hit counts) do not bias rho."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _lag1(v: np.ndarray) -> float:
    vc = v - v.mean()
    denom = float((vc * vc).sum())
    return float((vc[:-1] * vc[1:]).sum() / denom) if denom else 0.0


# ── Section 1: the x axis ─────────────────────────────────────────────


def axis_structure(ids: np.ndarray, total: np.ndarray, ctx_bits: int, perms: int, seed: int):
    """Test whether the id axis carries anything, and if so, at what scale."""
    rng = np.random.default_rng(seed)
    families = defaultdict(list)
    for edge, count in zip(ids.tolist(), total.tolist(), strict=True):
        families[edge >> ctx_bits].append(count)

    # Nested variance decomposition on log2(1+count): how much of the
    # spread is *between* context-free families rather than within one.
    y = np.log2(1.0 + total)
    grand = y.mean()
    between = 0.0
    for members in families.values():
        fam = np.log2(1.0 + np.asarray(members, dtype=float))
        between += len(fam) * (fam.mean() - grand) ** 2
    ss_total = float(((y - grand) ** 2).sum())

    observed = _lag1(total)
    null = np.array([_lag1(rng.permutation(total)) for _ in range(perms)])
    # Control (Hard Rule 46): shuffling only *within* families must leave the
    # effect standing if the effect is the blocking.  If it collapses, the
    # signal is a gradient along x, which would be a genuine surprise worth
    # chasing rather than the context layout.
    fam_index: dict[int, list[int]] = defaultdict(list)
    for i, edge in enumerate(ids.tolist()):
        fam_index[edge >> ctx_bits].append(i)
    within = total.copy()
    for idxs in fam_index.values():
        within[idxs] = rng.permutation(total[idxs])

    odd = int((ids & 1).sum())
    sizes = sorted((len(v) for v in families.values()), reverse=True)
    return {
        "n_edges": int(len(ids)),
        "id_min": int(ids.min()) if len(ids) else 0,
        "id_max": int(ids.max()) if len(ids) else 0,
        "ctx_bits": ctx_bits,
        "families": len(families),
        "largest_families": sizes[:8],
        # A pre-2026-09-23 shim forced `edge_id |= 1`, which killed tag bit 0;
        # the current one remaps only id 0. All-odd ids identify the old one.
        "ctx_tags_reachable": ((1 << ctx_bits) // (2 if odd == len(ids) else 1)) if ctx_bits else 1,
        "all_ids_odd": odd == len(ids),
        "icc_family": (between / ss_total) if ss_total else 0.0,
        "lag1_observed": observed,
        "lag1_null_mean": float(null.mean()),
        "lag1_null_sd": float(null.std()),
        "lag1_z": float((observed - null.mean()) / null.std()) if null.std() else 0.0,
        "lag1_within_family_shuffle": _lag1(within),
        "spearman_id_vs_count": _spearman(ids.astype(float), total),
    }


# ── Section 2: the y marginal ─────────────────────────────────────────


def y_marginal(total: np.ndarray):
    """Everything about the matrix that survives permuting the id axis."""
    p = total / total.sum()
    entropy = float(-(p * np.log2(p)).sum())
    desc = np.sort(total)[::-1]
    n = len(total)
    # Shared with the live per-tick Gini readouts (seed energy, operator
    # selection, edge hits) and the crash-cluster-size Gini in report.py --
    # see core/gini.py for why this moved out of a local copy.
    gini = _gini(total.tolist())
    rank = np.arange(1, n + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(rank), np.log(desc), 1)
    resid = np.log(desc) - (slope * np.log(rank) + intercept)
    classes = Counter(classify_single(int(c)) for c in total)
    return {
        "mean": float(total.mean()),
        "median": float(np.median(total)),
        "max": float(total.max()),
        "entropy_bits": entropy,
        "entropy_max_bits": math.log2(n) if n else 0.0,
        "effective_edges": 2.0**entropy,
        "gini": gini,
        "top_1pct_share": float(desc[: max(1, n // 100)].sum() / desc.sum()),
        "zipf_slope": float(slope),
        "zipf_resid_sd": float(resid.std()),
        "count_classes": {str(k): v for k, v in sorted(classes.items())},
    }


# ── Section 3: axes that mean something ───────────────────────────────


def substituted_axes(agg):
    """Rank correlation of each meaningful x candidate against total count."""
    total = agg["total"]
    return {
        "spearman_owners_vs_total": _spearman(agg["owners"], total),
        "spearman_first_seen_vs_total": _spearman(agg["first_seen"], total),
        "spearman_peak_vs_total": _spearman(agg["peak"], total),
    }


# ── Sections 4 and 5: the seed x edge matrix ──────────────────────────
#
# The (edge_id, count) columns are not a matrix -- two incommensurable
# columns. The matrix the counts live in is seed x edge, which is what
# core/schedulers/seed_tang.py factorises. Both sections below exist mainly
# as F1 canaries: a spectrum and a GF(2) rank are far more sensitive to
# per-process id drift than an edge count is, because every phantom id is a
# column owned by exactly one seed.

SVD_CELL_BUDGET = 20_000_000  # m*n above which the dense SVD is skipped
GREEDY_CELL_BUDGET = 50_000_000


def seed_edge_matrix(runs):
    """Dense seed x edge hit-count matrix, rows in execution order."""
    ids = sorted({int(e) for eids, _ in runs for e in eids.tolist()})
    col = {e: j for j, e in enumerate(ids)}
    mat = np.zeros((len(runs), len(ids)), dtype=float)
    for r, (eids, counts) in enumerate(runs):
        for edge, count in zip(eids.tolist(), counts.tolist(), strict=True):
            mat[r, col[int(edge)]] = count
    return mat


def spectrum(mat, ks=(1, 3, 10, 20)):
    """Singular spectrum under all three cell semantics.

    Reporting the three together is deliberate: a harness bug upstream of the
    transform once produced *identical* spectra for hit counts, log1p and
    binary (`handover_done_2026-09-06.md` §14, third round), and identical
    rows here are the signature of that class of bug rather than a result.
    """
    if mat.size > SVD_CELL_BUDGET:
        return {"skipped": f"{mat.shape[0]}x{mat.shape[1]} exceeds the dense-SVD budget"}
    out = {}
    for label, cells in (
        ("raw", mat),
        ("log1p", np.log1p(mat)),
        ("binary", (mat > 0).astype(float)),
    ):
        sv = np.linalg.svd(cells, compute_uv=False)
        sq = sv**2
        mass = float(sq.sum())
        out[label] = {
            "dominance": float(sq[0] / mass) if mass else 0.0,
            # participation ratio of the squared spectrum: 1.0 means rank-1,
            # min(m, n) means flat.
            "effective_rank": float(mass**2 / float((sq**2).sum())) if mass else 0.0,
            "rho": {str(k): float(np.sqrt(sq[k:].sum() / mass)) if mass else 0.0 for k in ks},
        }
    return out


def _gf2_reduce(binary):
    """Row-reduce over GF(2) using one big int per row.

    Returns (rank, pivot columns, indices of rows that reduced to zero). A
    row reduces to zero exactly when its coverage is the symmetric difference
    of earlier rows -- which, since XOR can only set bits present in an
    operand, implies its edges are contained in their union. That makes this
    a *sound* redundancy detector; it is not a complete one, because a seed
    whose edges are merely a subset of another's produces no cancellation and
    stays independent.
    """
    pivots: dict[int, int] = {}
    zero_rows: list[int] = []
    for idx, row in enumerate(binary):
        cur = int.from_bytes(np.packbits(row[::-1].astype(np.uint8)).tobytes(), "big")
        cur >>= (-len(row)) % 8  # packbits pads the high end; drop the padding
        while cur:
            bit = cur.bit_length() - 1
            if bit in pivots:
                cur ^= pivots[bit]
            else:
                pivots[bit] = cur
                break
        if cur == 0:
            zero_rows.append(idx)
    return len(pivots), sorted(pivots), zero_rows


def gf2_structure(mat):
    """GF(2) rank of the binary matrix, against the redundancy it is a proxy for."""
    binary = mat > 0
    rank, pivots, zero_rows = _gf2_reduce(binary)
    owners = binary.sum(axis=0)
    # A seed's edges are contained in the union of the others exactly when it
    # owns no singleton edge -- O(m*n) and exact, no pairwise comparison.
    has_singleton = (binary & (owners == 1)).any(axis=1)
    union_redundant = int((~has_singleton).sum())

    greedy = None
    if binary.size <= GREEDY_CELL_BUDGET:
        need = np.ones(binary.shape[1], dtype=bool)
        greedy = 0
        while need.any():
            gains = binary[:, need].sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            need &= ~binary[best]
            greedy += 1

    pivot_cols = list(pivots)
    separated = len({binary[i][pivot_cols].tobytes() for i in range(binary.shape[0])})
    return {
        "rows": int(binary.shape[0]),
        "gf2_rank": rank,
        "real_rank": int(np.linalg.matrix_rank(binary.astype(float))),
        "distinct_rows": len({r.tobytes() for r in binary}),
        "xor_dependent_rows": len(zero_rows),
        "union_redundant_seeds": union_redundant,
        "pivot_edges": len(pivot_cols),
        "distinct_rows_on_pivots": separated,
        "greedy_cover_seeds": greedy,
    }


# ── Section 6: integer relations (opt-in) ─────────────────────────────

LLL_ROW_BUDGET = 100  # LLL here is ~2 minutes at 100 rows and superlinear


_lll_reduce = lll_reduce  # moved to core/lattice.py


def integer_relations(mat, max_rows=LLL_ROW_BUDGET):
    """Integer structure of the raw-count matrix: duplicates, sparse relations, LLL.

    Off by default (``--lll``) because it is the most expensive analysis here
    by three orders of magnitude and, measured on fuzzgoat, the least
    informative: the exact relations LLL recovers are dense (support ~n/2,
    coefficients into the tens) and the only short vectors in the lattice are
    duplicate-row differences, which the hash below finds in O(m).

    The exhaustive scalar-multiple and A = B + C searches are what make a
    negative meaningful: they prove absence of sparse relations at support
    <= 3 outright, where LLL proves nothing -- its approximation factor is
    2^((n-1)/2), vacuous at these dimensions.
    """
    rows = [r for r in mat.astype(int)]
    first: dict[bytes, int] = {}
    distinct_idx = []
    for i, row in enumerate(rows):
        key = row.tobytes()
        if key in first:
            continue
        first[key] = i
        distinct_idx.append(i)
    dist = mat.astype(int)[distinct_idx]
    triples, multiples = _sparse_relations(dist)

    out = {
        "rows": int(mat.shape[0]),
        "distinct_rows": len(dist),
        "duplicate_rows": int(mat.shape[0]) - len(dist),
        "scalar_multiple_pairs": len(multiples),
        "sum_triples": len(triples),
    }

    sub = dist[:max_rows]
    sub = sub[:, sub.any(axis=0)]
    n = sub.shape[0]
    rank = int(np.linalg.matrix_rank(sub.astype(float)))
    rels, elapsed = _lll_relations(sub)
    support = [sum(1 for v in c if v) for c in rels]
    l1 = [sum(abs(v) for v in c) for c in rels]
    peak = [max(abs(v) for v in c) for c in rels]
    out["lll"] = {
        "rows": n,
        "cols": int(sub.shape[1]),
        "rank": rank,
        "kernel_dim": n - rank,
        "relations_found": len(rels),
        "support_median": float(np.median(support)) if rels else None,
        "l1_median": float(np.median(l1)) if rels else None,
        "max_coeff_median": float(np.median(peak)) if rels else None,
        "seconds": elapsed,
    }
    return out


# ── Section 7: edge equivalence classes (transposed orientation) ──────


def duplicate_classes(mat, row_ids, ctx_bits):
    """Group edges whose count profile is identical across every execution.

    Only meaningful on the transposed matrix, where a row is an edge. Two
    edges in one class carry the same information in this corpus: whatever
    distinguishes them, no input here exercised it.

    The split by ``id >> ctx_bits`` is the actionable part. A class confined
    to one family is a base edge whose context tags never once differed --
    map slots context sensitivity bought and did not use, countable exactly
    rather than inferred from the ICC in section [1]. A class spanning
    several families is a straight-line block chain, a property of the target
    rather than of the instrumentation.
    """
    groups: dict[bytes, list[int]] = {}
    for i, row in enumerate(mat.astype(int)):
        groups.setdefault(row.tobytes(), []).append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    within = [g for g in multi if len({int(row_ids[i]) >> ctx_bits for i in g}) == 1]
    across = [g for g in multi if len({int(row_ids[i]) >> ctx_bits for i in g}) > 1]
    largest = max(multi, key=len) if multi else []
    return {
        "rows": int(mat.shape[0]),
        "classes": len(groups),
        "duplicate_rows": sum(len(g) - 1 for g in multi),
        "within_family_classes": len(within),
        "within_family_rows": sum(len(g) - 1 for g in within),
        "across_family_classes": len(across),
        "across_family_rows": sum(len(g) - 1 for g in across),
        "largest_class": len(largest),
        "largest_class_families": sorted({int(row_ids[i]) >> ctx_bits for i in largest})[:4],
    }


# ── Section 8: the (edge_pos, edge_id, count) matrix (opt-in) ─────────
#
# x = edge_pos is the SHM slot index, derived by the shim's linear probe
# from home = edge_id % map_size; y = edge_id; z = hit count.  The 2-D
# projection (pos, id) with the count as its value and the 3-D binary
# tensor (pos, id, count) are the two matrices under study.
#
# Three falsifiable predictions fall straight out of the shim design:
#
# *  a slot is claimed once and never reclaimed, so an id's position must
#    be fixed for the whole run once its edge first fires;
# *  an id's count is a property of the *input*, its position a property of
#    the *insertion* -- so count and position should be independent;
# *  per run the (pos, id) matrix is a partial permutation (one live id per
#    slot, one slot per id), so its singular spectrum is exactly the
#    per-edge count histogram: a fold is a bijection onto its image and
#    cannot add information (handover F12, now made quantitative).
#
# Every correlation is tested against a null that permutes the companion
# margin among the triples (Hard Rule 46), so the marginal distributions
# are fixed under the null.

SVD_CELL_BUDGET_POS = 20_000_000  # trimmed per-run matrix cells


def _placement_rollup(pos_runs):
    """Flatten every run into (run_idx, pos, id, count) plus slot/id membership."""
    pos_of: dict[int, set[int]] = defaultdict(set)
    ids_at_slot: dict[int, set[int]] = defaultdict(set)
    triples = []
    for run_idx, (positions, ids, counts) in enumerate(pos_runs):
        for pos, eid, cnt in zip(positions.tolist(), ids.tolist(), counts.tolist(), strict=True):
            pos_of[eid].add(pos)
            ids_at_slot[pos].add(eid)
            triples.append((run_idx, pos, eid, cnt))
    return pos_of, ids_at_slot, triples


def _placement_overview(arr, pos_of, ids_at_slot, map_size):
    """Placement identity stats; the F14 machinery, end to end."""
    arr_pos, arr_id, arr_cnt = arr[:, 1], arr[:, 2], arr[:, 3]
    with np.errstate(invalid="ignore"):
        home = arr_id % map_size
        disp = (arr_pos - home) % map_size
    n_ids = len(pos_of)
    ids_multi_position = sum(1 for s in pos_of.values() if len(s) > 1)
    slots_multi_id = sum(1 for ids_at_pos in ids_at_slot.values() if len(ids_at_pos) > 1)
    return (
        {
            "unique_ids": n_ids,
            "ids_single_position": n_ids - ids_multi_position,
            "ids_single_position_pct": 100.0 * (n_ids - ids_multi_position) / n_ids,
            "ids_multi_position": ids_multi_position,
            "slots_multi_id": slots_multi_id,
            "displacement_mean": float(disp.mean()),
            "home_hit_frac": float((disp == 0).mean()),
            "beyond_probe_max": int((disp >= ShmCoverage.PROBE_MAX).sum()),
            "disp_hist_top": [int(c) for _k, c in Counter(disp.tolist()).most_common(8)],
        },
        home,
        disp,
        arr_cnt,
    )


def _spearman_null_block(a, b, perms, rng, prefix):
    """Observed rank correlation against a b-preserving permutation null."""
    obs = _spearman(a, b)
    null = np.array([_spearman(a, rng.permutation(b)) for _ in range(perms)])
    return {
        f"{prefix}": obs,
        f"{prefix}_null_mean": float(null.mean()),
        f"{prefix}_null_sd": float(null.std()),
        f"{prefix}_null_z": float((obs - null.mean()) / null.std()) if null.std() else 0.0,
    }


def _spectral_check(pos_runs, budget):
    """F14 quantitative check: first-run (pos, id) spectrum == count histogram."""
    first = pos_runs[0]
    cells = len(first[0]) * len(first[1])
    if not 0 < cells <= budget:
        return {}
    rowset = {int(p) for p in first[0].tolist()}
    colset = {int(i) for i in first[1].tolist()}
    rm = {r: k for k, r in enumerate(sorted(rowset))}
    cm = {c: k for k, c in enumerate(sorted(colset))}
    mat = np.zeros((len(rm), len(cm)), dtype=float)
    for p, i, c in zip(first[0].tolist(), first[1].tolist(), first[2].tolist(), strict=True):
        mat[rm[p], cm[i]] = c
    sv = np.linalg.svd(mat, compute_uv=False)
    ref = np.sort(first[2].astype(float))[::-1]
    return {
        "rank_2d": int(np.linalg.matrix_rank(mat)),
        "single_run_sv_max_relerr": float(
            np.max(np.abs(sv - ref) / ref.max()) if ref.size else 0.0
        ),
    }


def positions_structure(pos_runs, map_size: int, perms: int, seed: int):
    """Placement statistics over the (pos, id, count) triples of every run."""
    if pos_runs is None:
        return {"available": False}
    rng = np.random.default_rng(seed)

    pos_of, ids_at_slot, triples = _placement_rollup(pos_runs)
    if not triples:
        return {"available": True, "runs": len(pos_runs), "triples": 0, "unique_ids": 0}

    arr = np.array(triples, dtype=np.int64)
    first_run = {}
    for run_idx, (_, ids, _) in enumerate(pos_runs):
        for eid in ids.tolist():
            first_run.setdefault(eid, run_idx)
    arr_first = np.array([first_run[e] for e in arr[:, 2].tolist()], dtype=np.int64)

    overview, home, disp, arr_cnt = _placement_overview(arr, pos_of, ids_at_slot, map_size)
    out = {
        "available": True,
        "runs": len(pos_runs),
        "triples": int(len(triples)),
        "table_size": map_size,
        **overview,
        **_spearman_null_block(
            disp.astype(float), arr_cnt.astype(float), perms, rng, "spearman_disp_count"
        ),
        **_spearman_null_block(
            home.astype(float), disp.astype(float), perms, rng, "spearman_home_disp"
        ),
    }

    # Confound check: displacement is decided by table occupancy at the
    # moment of insertion, so any disp-count association should trace to
    # first-seen run, not to the count itself.
    out["spearman_disp_first_run"] = _spearman(disp.astype(float), arr_first.astype(float))
    out["spearman_first_run_count"] = _spearman(arr_first.astype(float), arr_cnt.astype(float))
    out.update(_spectral_check(pos_runs, SVD_CELL_BUDGET_POS))
    return out


# ── Reporting ─────────────────────────────────────────────────────────


def _gt_runs(tracer: Path, inputs: list[Path], timeout: float) -> list[np.ndarray]:
    """One (n, 3) array of tracer records per input, in input order.

    An input that left no log (died before its first coverage event) gets an
    empty array rather than a gap, so row i is always input i -- the flow
    section lines these rows up with the shim's collection.
    """
    import subprocess
    import tempfile

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gt.bin"
        for p in inputs:
            out.unlink(missing_ok=True)
            env = dict(os.environ, GT_OUT=str(out))
            # A hang still leaves the edges logged before it.
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run([str(tracer), str(p)], env=env, capture_output=True, timeout=timeout)
            raw = np.fromfile(out, dtype="<u8") if out.exists() else np.empty(0, "<u8")
            runs.append(raw[: len(raw) - len(raw) % 3].reshape(-1, 3))
    return runs


def ground_truth(tracer: Path, inputs: list[Path], timeout: float, runs=None) -> dict:
    """Real edges per tools/ground_truth_tracer.c, over the collected inputs.

    Returns the distinct context-free edges (prev, cur) and the distinct
    (prev, cur, call site) triples. The tracer shares no code with the shim's
    id function, so these are the reference the collected ids are judged
    against. A non-zero exit is expected (fuzzgoat aborts on its planted
    bugs); the tracer writes unbuffered, so those executions still count.
    Pass *runs* (from ``_gt_runs``) to reuse a collection instead of re-running.
    """
    if runs is None:
        runs = _gt_runs(tracer, inputs, timeout)
    edges: set[tuple[int, int]] = set()
    triples: set[tuple[int, int, int]] = set()
    for rec in runs:
        for prev, cur, site in rec.tolist():
            edges.add((prev, cur))
            triples.add((prev, cur, site))
    return {"edges": len(edges), "triples": len(triples)}


# ── Section 10: flow conservation on the tracer's walk graph (P1-2) ───
#
# A tracer log is a walk, so a run's edge counts form a circulation once the
# walk is closed (last node -> EXIT -> ENTRY, both virtual). Circulations are
# exactly the kernel of the node-edge incidence matrix B, spanned by the
# fundamental cycles Z. Hence a relation r over edge counts holds for every
# run the graph admits iff Z r = 0 (r in rowspan B: a sum of node laws), and
# rank(Z) is the number of independent count coordinates -- the Ball-Larus
# bound: instrument rank(Z) chords, derive the rest.
#
#        ENTRY --> 1 --> 2 --> 3 --> ... --> last --> EXIT
#          ^                                            |
#          +-------------- virtual return --------------+
#
# Everything here is exact: incidence and cycle vectors are small integers,
# and ranks are taken over GF(p) for a 31-bit prime, which equals the rank
# over Q unless p divides a minor -- the counts here are orders of magnitude
# too small for that to be plausible.


class FlowGraph(enum.Enum):
    """Node identity of the walk graph."""

    EDGE = "context-free"  # node = location, column = (prev, cur)
    CALL_SITE = "call-site"  # node = (location, call site), column = (prev, cur, site)


RANK_PRIME = 2**31 - 1
NON_FLOW_SHOWN = 8  # non-structural relations kept per class, for reading
_ENTRY, _EXIT = ("entry",), ("exit",)


def _rank_p(mat) -> int:
    """Rank of an integer matrix over GF(RANK_PRIME), by vectorised elimination."""
    a = np.array([[int(v) % RANK_PRIME for v in row] for row in np.asarray(mat)], dtype=np.int64)
    if a.size == 0:
        return 0
    rank = 0
    for c in range(a.shape[1]):
        nz = np.flatnonzero(a[rank:, c])
        if not len(nz):
            continue
        p = rank + nz[0]
        a[[rank, p]] = a[[p, rank]]
        a[rank] = a[rank] * pow(int(a[rank, c]), RANK_PRIME - 2, RANK_PRIME) % RANK_PRIME

        # Clear column c everywhere else; entries < 2^31, so products fit int64.
        col = a[:, c].copy()
        col[rank] = 0
        a = (a - np.outer(col, a[rank]) % RANK_PRIME) % RANK_PRIME
        rank += 1
        if rank == a.shape[0]:
            break
    return rank


def _cycle_basis(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    """Fundamental cycles of a directed multigraph, one +/-1 row per chord.

    BFS spanning forest over the undirected graph; each non-tree edge closes
    one cycle through the tree. Row count is E - V + components.
    """
    adj = collections.defaultdict(list)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # parent[w] = (tree edge, +1 if it points parent -> w else -1)
    parent: dict[int, tuple[int, int] | None] = {}
    for root in range(n):
        if root in parent:
            continue
        parent[root] = None
        queue = [root]
        for u in queue:
            for v, i in adj[u]:
                if v in parent:
                    continue
                parent[v] = (i, 1 if src[i] == u else -1)
                queue.append(v)

    tree = {p[0] for p in parent.values() if p is not None}
    rows = []
    for i in range(len(src)):
        if i in tree:
            continue
        z = np.zeros(len(src), dtype=np.int64)
        z[i] = 1
        _walk_up(z, int(dst[i]), -1, parent, src, dst)  # dst -> root, against the tree
        _walk_up(z, int(src[i]), 1, parent, src, dst)  # root -> src, along it
        rows.append(z)
    return np.array(rows, dtype=np.int64).reshape(-1, len(src))


def _walk_up(z, w, sign, parent, src, dst) -> None:
    """Add the tree path w -> root to *z*, times *sign* (-1 walks it upward)."""
    while parent[w] is not None:
        i, orient = parent[w]
        z[i] += sign * orient
        w = int(src[i]) if orient == 1 else int(dst[i])


def _flow_state(rec, kind: FlowGraph):
    return rec[1] if kind is FlowGraph.EDGE else (rec[1], rec[2])


def _flow_col(rec, kind: FlowGraph):
    return (rec[0], rec[1]) if kind is FlowGraph.EDGE else (rec[0], rec[1], rec[2])


def _flow_graph(runs: list[np.ndarray], kind: FlowGraph) -> dict:
    """Walk graph of the tracer runs: incidence, per-run counts, cycles."""
    nodes = {_ENTRY: 0, _EXIT: 1}
    edges: dict[tuple, int] = {}
    cols: dict[tuple, int] = {}
    ecol: list[int] = []  # column of each walk edge, -1 for virtual ones
    per_run, exits, discontinuities = [], set(), 0

    def edge(a, b, col) -> int:
        if (a, b) not in edges:
            nodes.setdefault(a, len(nodes))
            nodes.setdefault(b, len(nodes))
            edges[(a, b)] = len(edges)
            ecol.append(-1 if col is None else cols.setdefault(col, len(cols)))
        return edges[(a, b)]

    for rec in runs:
        # Discontinuity: a record whose prev is not where the walk was.
        if len(rec):
            discontinuities += int(rec[0, 0] != 0) + int((rec[1:, 0] != rec[:-1, 1]).sum())
        hits: Counter[int] = Counter()
        prev = _ENTRY
        for r in rec.tolist():
            state = _flow_state(r, kind)
            hits[edge(prev, state, _flow_col(r, kind))] += 1
            prev = state
        exits.add(prev)
        hits[edge(prev, _EXIT, None)] += 1
        hits[edge(_EXIT, _ENTRY, None)] += 1
        per_run.append(hits)

    n_e = len(edges)
    src, dst = np.empty(n_e, dtype=np.int64), np.empty(n_e, dtype=np.int64)
    for (a, b), i in edges.items():
        src[i], dst[i] = nodes[a], nodes[b]
    af = np.zeros((len(runs), n_e), dtype=np.int64)
    for row, hits in enumerate(per_run):
        for e, c in hits.items():
            af[row, e] = c

    # Project walk edges onto columns: several walk edges can share a column
    # in the call-site graph (same (prev, cur, site), different prev site).
    ecol_a = np.array(ecol, dtype=np.int64)
    proj = np.zeros((n_e, len(cols)), dtype=np.int64)
    real = np.flatnonzero(ecol_a >= 0)
    proj[real, ecol_a[real]] = 1
    z = _cycle_basis(src, dst, len(nodes))
    return {
        "src": src,
        "dst": dst,
        "nodes": len(nodes),
        "cols": cols,
        "af": af,
        "ac": af @ proj,
        "z": z,
        "zp": z @ proj,
        "exits": len(exits),
        "discontinuities": discontinuities,
    }


def _is_structural(zp: np.ndarray, r: np.ndarray) -> bool:
    """True iff relation *r* over columns is a sum of node laws."""
    return not np.count_nonzero(zp @ r)


def _flow_summary(g: dict, lll_rows: int) -> dict:
    """Section [10] numbers for one walk graph."""
    n_nodes, n_cols = g["nodes"], len(g["cols"])
    b = np.zeros((n_nodes, len(g["src"])), dtype=np.int64)
    b[g["src"], np.arange(len(g["src"]))] -= 1
    b[g["dst"], np.arange(len(g["src"]))] += 1

    independent = _rank_p(g["zp"])
    count_rank = _rank_p(g["ac"])
    return {
        "runs": int(g["ac"].shape[0]),
        "nodes": n_nodes,
        "columns": n_cols,
        "exits": g["exits"],
        "discontinuities": g["discontinuities"],
        "kirchhoff_violations": int(np.count_nonzero(b @ g["af"].T)),
        "cycle_rank": int(g["z"].shape[0]),
        "independent": independent,
        "derivable": n_cols - independent,
        "count_rank": count_rank,
        "kernel_dims": n_cols - count_rank,
        "non_flow_dims": independent - count_rank,
        "counts_in_cycle_space": _rank_p(np.vstack([g["ac"], g["zp"]])) == independent,
        "relations": _classify_relations(g, lll_rows),
    }


def _classify_relations(g: dict, lll_rows: int) -> dict:
    """Every sparse empirical relation, split into node laws and the rest."""
    keys = {i: k for k, i in g["cols"].items()}
    out = {}
    for cls, rels in _relation_sets(g["ac"].T, lll_rows).items():
        shown, structural = [], 0
        for rel in rels:
            r = np.zeros(len(keys), dtype=np.int64)
            for i, c in rel:
                r[i] += c
            if _is_structural(g["zp"], r):
                structural += 1
                continue
            if len(shown) < NON_FLOW_SHOWN:
                shown.append([[keys[i], c] for i, c in rel])
        out[cls] = {"found": len(rels), "structural": structural, "non_flow": shown}
    return out


def _relation_sets(mat, lll_rows: int) -> dict[str, list[list[tuple[int, int]]]]:
    """Duplicate, A=B+C, scalar-multiple and LLL relations among the rows of *mat*.

    A relation is a list of (row, integer coefficient). Duplicates are paired
    with the first row of their class; the other searches run on distinct rows.
    """
    mat = np.asarray(mat, dtype=np.int64)
    first: dict[bytes, int] = {}
    dups, distinct = [], []
    for i, row in enumerate(mat):
        j = first.setdefault(row.tobytes(), i)
        if j == i:
            distinct.append(i)
        else:
            dups.append([(j, 1), (i, -1)])

    triples, multiples = _sparse_relations(mat[distinct])
    rels, _ = _lll_relations(mat[distinct[:lll_rows]]) if lll_rows else ([], 0.0)
    back = distinct.__getitem__
    return {
        "duplicates": dups,
        "triples": [[(back(k), 1), (back(i), -1), (back(j), -1)] for k, i, j in triples],
        "multiples": [[(back(i), q), (back(j), -p)] for i, j, p, q in multiples],
        "lll": [[(back(i), c) for i, c in enumerate(rel) if c] for rel in rels],
    }


def _sparse_relations(dist):
    """Exhaustive support-3 search over distinct rows.

    Returns (triples, multiples): triples (k, i, j) with row k = row i + row j,
    multiples (i, j, p, q) with q * row i = p * row j and p != q, both exact.
    """
    index = {dist[i].tobytes(): i for i in range(len(dist))}
    triples, multiples = [], []
    for i in range(len(dist)):
        for j in range(i + 1, len(dist)):
            k = index.get((dist[i] + dist[j]).tobytes())
            if k is not None:
                triples.append((k, i, j))
            a, b = dist[i], dist[j]
            if not (np.array_equal(a > 0, b > 0) and (a > 0).any()):
                continue
            nz = np.flatnonzero(a)[0]
            p, q = int(a[nz]), int(b[nz])  # a / b = p / q
            if p != q and np.array_equal(a * q, b * p):
                multiples.append((i, j, p, q))
    return triples, multiples


def _lll_relations(sub):
    """Exact integer relations among the rows of *sub* via LLL, and the time taken."""
    sub = sub[:, sub.any(axis=0)]
    n = sub.shape[0]
    # [I | N*A]: a reduced row whose A-part vanishes carries an exact integer
    # relation in its I-part. N only has to outweigh the coefficients we care
    # about, so that a row keeping any coverage mass cannot look short.
    scale = 10**4
    basis = [
        [1 if j == i else 0 for j in range(n)] + [scale * int(v) for v in sub[i]] for i in range(n)
    ]
    start = time.perf_counter()
    reduced = _lll_reduce(basis)
    return [row[:n] for row in reduced if not any(row[n:])], time.perf_counter() - start


def flow_structure(gt_runs: list[np.ndarray], lll_rows: int = LLL_ROW_BUDGET) -> dict:
    """Section [10]: both walk graphs, keyed by FlowGraph value."""
    return {k.value: _flow_summary(_flow_graph(gt_runs, k), lll_rows) for k in FlowGraph}


def _profile_match(a, b) -> bool:
    """Same multiset of column profiles -- equal up to a column permutation."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return sorted(map(tuple, a.T.tolist())) == sorted(map(tuple, b.T.tolist()))


def _matrix_report(result) -> None:
    coll = result["collection"]
    print(
        f"inputs {coll['inputs']}  median live edges {coll['median_live_edges']}  "
        f"union {coll['union_edges']}  total hits {coll['total_hits']}"
    )
    print(f"matrix orientation for sections [4]-[7]: {coll['orientation']}")
    if "stability" in result:
        s = result["stability"]
        print(
            f"\n[0] cross-process id stability ({s['repeats']} runs of "
            f"{s.get('input', 'one input')}, the input with the most live edges)"
        )
        print(
            f"    sizes {s['sizes']}  union {s['union']}  intersection {s['intersection']}  "
            f"jaccard {s['jaccard']:.3f}"
        )
        if s["jaccard"] < 1.0:
            print("    WARNING: edge ids are not reproducible across processes. Every per-edge")
            print("    statistic below is aggregating ids that do not denote the same edge.")
            print("    First suspect: a PIE target under ASLR with __AFL_CTX_SENSITIVE=1.")
        if not s["first_exec_matches"]:
            print(
                f"    first execution (clean table) disagreed with the steady state; "
                f"{s['first_exec_only']} of its ids never reappear"
            )

    a = result["axis"]
    print(f"\n[1] x axis: {a['n_edges']} distinct ids in [{a['id_min']}, {a['id_max']}]")
    print(
        f"    ctx_bits {a['ctx_bits']}: {a['families']} context-free families "
        f"(id >> {a['ctx_bits']}), largest {a['largest_families']}"
    )
    print(
        f"    reachable ctx tags per family {a['ctx_tags_reachable']} "
        f"(all ids odd: {a['all_ids_odd']} -- True means a legacy `|= 1` shim, "
        "which kills tag bit 0)"
    )
    if a["ctx_bits"] and not a["all_ids_odd"]:
        print(
            "    note: under the hashed-location shim a family is the hashed location's\n"
            "    bits above ctx_bits, so context-free edges that share them count as one\n"
            "    family (fuzzgoat: 323 families for 344 context-free edges). For an exact\n"
            "    count pair this with a __AFL_CTX_SENSITIVE=0 build or --ground-truth."
        )
    degenerate = (
        " (degenerate: one edge per family, nothing to decompose)"
        if a["families"] == a["n_edges"]
        else ""
    )
    print(f"    ICC: family explains {a['icc_family']:.3f} of log2(1+count) variance{degenerate}")
    print(
        f"    lag-1 autocorr along id order {a['lag1_observed']:+.4f}  "
        f"null {a['lag1_null_mean']:+.4f} +/- {a['lag1_null_sd']:.4f}  z={a['lag1_z']:+.2f}"
    )
    print(
        f"    control, shuffled within family: {a['lag1_within_family_shuffle']:+.4f} "
        "(should stay near observed)"
    )
    print(
        f"    Spearman(id, count) {a['spearman_id_vs_count']:+.4f} "
        "-- not interpretable as a trend; reported to show it is not zero either"
    )

    y = result["y_marginal"]
    print(f"\n[2] y marginal: mean {y['mean']:.1f} median {y['median']:.0f} max {y['max']:.0f}")
    print(
        f"    Shannon {y['entropy_bits']:.2f} of {y['entropy_max_bits']:.2f} bits "
        f"-> {y['effective_edges']:.0f} effective edges"
    )
    print(
        f"    Gini {y['gini']:.3f}  top 1% of edges carry {y['top_1pct_share'] * 100:.1f}% of hits"
    )
    print(f"    Zipf log-log slope {y['zipf_slope']:+.3f} (residual sd {y['zipf_resid_sd']:.3f})")
    print(f"    AFL count classes {y['count_classes']}")

    s = result["substituted_axes"]
    print("\n[3] substituted x axes, Spearman vs total count")
    print(f"    owner count (incidence)  {s['spearman_owners_vs_total']:+.3f}")
    print(f"    first-seen index         {s['spearman_first_seen_vs_total']:+.3f}")
    print(f"    per-edge max count       {s['spearman_peak_vs_total']:+.3f}")
    print("    distance-table node_idx: not collected here -- needs a __AFL_DISTANCE_MODE build")

    sp = result["spectrum"]
    print(
        f"\n[4] {result['collection']['orientation']} spectrum "
        "(not the (id, count) columns -- those are not a matrix)"
    )
    if result["collection"]["orientation"] == "edge x seed":
        print("    identical to the seed x edge spectrum by construction, sigma(A) = sigma(A^T);")
        print("    printed as a control -- a difference here means the pipeline is wrong.")
    if "skipped" in sp:
        print(f"    skipped: {sp['skipped']}")
    else:
        for label in ("raw", "log1p", "binary"):
            e = sp[label]
            print(
                f"    {label:7s} dominance {e['dominance']:.3f}  effective rank "
                f"{e['effective_rank']:5.1f}  rho k=1/3/10/20 "
                + "/".join(f"{e['rho'][k]:.3f}" for k in ("1", "3", "10", "20"))
            )
        if abs(sp["raw"]["dominance"] - sp["binary"]["dominance"]) < 1e-6:
            print("    WARNING: raw and binary spectra agree to 1e-6. That is a bug upstream")
            print("    of both transforms, not a result -- check the cells are counts.")

    g = result["gf2"]
    transposed = result["collection"]["orientation"] == "edge x seed"
    row, col = ("edges", "seeds") if transposed else ("seeds", "edges")
    print(
        f"\n[5] GF(2) structure: rank {g['gf2_rank']} (real rank {g['real_rank']}, "
        f"{g['distinct_rows']} distinct rows of {g['rows']} {row})"
    )
    print(f"    XOR-dependent {row} {g['xor_dependent_rows']} -- sound but incomplete: each one")
    print(f"    is union-redundant, but {g['union_redundant_seeds']} {row} actually are")
    if g["greedy_cover_seeds"] is not None:
        print(
            f"    as a minimiser: GF(2) basis {g['gf2_rank']} {row} vs greedy set cover "
            f"{g['greedy_cover_seeds']}"
        )
    print(
        f"    {g['pivot_edges']} pivot {col} separate {g['distinct_rows_on_pivots']} of "
        f"{g['distinct_rows']} distinct profiles"
    )
    if transposed:
        print("    note: union redundancy and greedy cover are seed-side metrics. Transposed")
        print("    they degenerate -- every edge sits inside the union of the others, and the")
        print("    cover is the single edge every input reaches.")

    if "integer_relations" in result:
        r = result["integer_relations"]
        print(
            f"\n[6] integer relations over the raw counts ({r['rows']} rows, "
            f"{r['distinct_rows']} distinct, {r['duplicate_rows']} exact duplicates)"
        )
        print(
            f"    sparse relations, exhaustive: {r['scalar_multiple_pairs']} scalar-multiple "
            f"pairs, {r['sum_triples']} A=B+C triples"
        )
        lll = r["lll"]
        print(
            f"    LLL on {lll['rows']}x{lll['cols']} (rank {lll['rank']}, kernel "
            f"{lll['kernel_dim']}): {lll['relations_found']} exact relations in "
            f"{lll['seconds']:.1f}s"
        )
        if lll["relations_found"]:
            print(
                f"    support median {lll['support_median']:.0f} of {lll['rows']}, "
                f"L1 median {lll['l1_median']:.0f}, max|coeff| median "
                f"{lll['max_coeff_median']:.0f}"
            )
            sparse = lll["support_median"] <= 8 and lll["max_coeff_median"] <= 2
            if sparse:
                print("    sparse, near-unit relations -- diagnostic only: on fuzzgoat most")
                print("    are not node laws (P1-2). --ground-truth --flow says which are.")
            else:
                print("    dense relations are not actionable: the only short vectors here")
                print("    are duplicate-row differences, which the hash above finds in O(m).")

    if "duplicate_classes" in result:
        d = result["duplicate_classes"]
        print(
            f"\n[7] edge equivalence classes: {d['classes']} distinct profiles for "
            f"{d['rows']} edges ({d['duplicate_rows']} exact copies)"
        )
        print(
            f"    within one id>>ctx_bits family: {d['within_family_rows']} edges in "
            f"{d['within_family_classes']} classes -- context tags that never differed"
        )
        print(
            f"    across families: {d['across_family_rows']} edges in "
            f"{d['across_family_classes']} classes -- straight-line block chains"
        )
        print(
            f"    largest class {d['largest_class']} edges, families {d['largest_class_families']}"
        )

    if "positions" in result:
        p = result["positions"]
        print("\n[8] edge_pos placement (SHM slot index), (pos, id, count) matrix")
        if not p["available"]:
            print("    unavailable: this collection has no positions -- re-collect with --save")
            return
        print(
            f"    table {p['table_size']} slots; {p['triples']} triples over {p['runs']} runs, "
            f"{p['unique_ids']} distinct ids"
        )
        print(
            f"    ids at a single position {p['ids_single_position']} "
            f"({p['ids_single_position_pct']:.1f}%)  -- slots are never reclaimed, so any"
        )
        print("    multi-position id contradicts the shim's design (first-fit is final)")
        print(
            f"    slots hosting >1 distinct id across runs: {p['slots_multi_id']}"
            "  (expected: only broken placements)"
        )
        print(
            f"    probe displacement: mean {p['displacement_mean']:.3f}, "
            f"home hits {p['home_hit_frac'] * 100:.1f}%, "
            f"beyond PROBE_MAX {p['beyond_probe_max']}  "
            f"top bins {p['disp_hist_top']}"
        )
        obs = p["spearman_disp_count"]
        print(
            f"    Spearman(displacement, count) {obs:+.4f}  "
            f"null {p['spearman_disp_count_null_mean']:+.4f} +/- "
            f"{p['spearman_disp_count_null_sd']:.4f}  z={p['spearman_disp_count_null_z']:+.2f}"
        )
        obs2 = p["spearman_home_disp"]
        print(
            f"    Spearman(home, displacement)  {obs2:+.4f}  "
            f"null {p['spearman_home_disp_null_mean']:+.4f} +/- "
            f"{p['spearman_home_disp_null_sd']:.4f}  z={p['spearman_home_disp_null_z']:+.2f}"
        )
        print(
            f"    confound: Spearman(disp, first_seen) {p['spearman_disp_first_run']:+.4f}, "
            f"Spearman(first_seen, count) {p['spearman_first_run_count']:+.4f}"
        )
        if "rank_2d" in p:
            print(
                f"    first-run (pos x id) matrix rank {p['rank_2d']}; "
                f"singular spectrum max|rel err| vs the count histogram "
                f"{p['single_run_sv_max_relerr']:.2e}"
            )
            print("    (a fold is a bijection onto its image: the pos x id view is the (id, count)")
            print("     view re-rendered, per handover F12 -- this measures that, F15)")

    if "ground_truth" in result:
        g = result["ground_truth"]
        print("\n[9] ground truth (tools/ground_truth_tracer.c over the same inputs)")
        print(
            f"    real context-free edges {g['edges']}, (edge, call site) triples "
            f"{g['triples']}, ids reported by --target {g['ids']}"
        )
        if g["ctx_bits"] == 0:
            merged = g["edges"] - g["ids"]
            print(
                f"    merged by the id function: {merged} of {g['edges']} "
                f"({merged / max(g['edges'], 1):.1%}) -- exact for a context-free build,\n"
                "    where each id is a function of (prev, cur) alone"
            )
            if merged < 0:
                print(
                    "    WARNING: more ids than real edges -- ids are not a function of the\n"
                    "    edge (unstable ids, or --target and the tracer build differ)"
                )
        else:
            print(
                "    context build: the tracer's call sites are its own binary's, so compare\n"
                "    counts, not triples; ids well below the triple count mean merging, and\n"
                "    a __AFL_CTX_SENSITIVE=0 target gives the exact figure"
            )
    if "flow" in result:
        _flow_report(result["flow"])


def _flow_report(flow: dict) -> None:
    print(f"\n[10] flow conservation on the tracer's walk graph (exact, GF({RANK_PRIME}))")
    for kind, f in flow.items():
        if kind == "profile_match":
            continue
        match = flow.get("profile_match", {}).get(kind)
        print(
            f"    {kind}: {f['nodes']} nodes, {f['columns']} edges, {f['exits']} exit nodes, "
            f"discontinuities {f['discontinuities']}, Kirchhoff violations "
            f"{f['kirchhoff_violations']}"
            + ("" if match is None else f", profiles equal --target's: {match}")
        )
        print(
            f"      independent counts {f['independent']} of {f['columns']} "
            f"(Ball-Larus: {f['derivable']} derivable from the graph); count rank "
            f"{f['count_rank']}, so {f['non_flow_dims']} of {f['kernel_dims']} "
            "empirical kernel dims are not flow conservation"
        )
        rel = f["relations"]
        print(
            "      structural / found: "
            + "  ".join(f"{c} {v['structural']}/{v['found']}" for c, v in rel.items())
        )
        for c, v in rel.items():
            for r in v["non_flow"][:2]:
                print(f"        not a node law, {c}: " + " ".join(f"{k:+d}*{e}" for e, k in r))
    print("    verdict: " + _flow_verdict(flow))


def _flow_verdict(flow: dict) -> str:
    """P1-2's decision rule, applied to the call-site graph (it sees call/return)."""
    f = flow[FlowGraph.CALL_SITE.value]
    if f["discontinuities"] or f["kirchhoff_violations"] or not f["counts_in_cycle_space"]:
        return "INVALID -- the walk graph does not conserve flow; fix the log before reading on"
    found = sum(v["found"] for v in f["relations"].values())
    structural = sum(v["structural"] for v in f["relations"].values())
    if f["non_flow_dims"] == 0 and 2 * structural >= found:
        return (
            "positive -- the empirical relations are node laws; derive edges from "
            "the graph (Ball-Larus)"
        )
    return (
        f"negative -- {found - structural} of {found} sparse relations and "
        f"{f['non_flow_dims']} kernel dims are not node laws; counts cannot tell "
        "the node laws apart, the graph can"
    )


# ── Entry point ───────────────────────────────────────────────────────


STATESTORE_JSON_FILES = frozenset(
    {
        "markov.json",
        "mi.json",
        "elo.json",
        "ga.json",
        "qea.json",
        "state.json",
        "sensitivity.json",
        "crash_mi.json",
        "length_tracker.json",
        "seed_quality.json",
        "edge_tracker.json",
    }
)


def _so_sibling(target: Path) -> Path | None:
    """Find a dlopen-able sibling of a PIE target for the hail-mary campaign.

    Ordered most-to-least usable: the coverage-only .so variants first
    (_noasan/_nosan, a plain .so), then _asan.so last.  An _asan.so can
    double-load libasan in-process (preload + dlopen) and trip an ASAN CHECK
    intermittently; the coverage-only variants avoid it and the campaign only
    needs coverage growth, not sanitizer detection.
    """
    stem = str(target)
    candidates = (
        [f"{stem}_noasan.so", f"{stem}_nosan.so", f"{stem}.so", f"{stem}_asan.so"]
        if Path(stem).suffix not in (".so", ".dylib", ".dll")
        else []
    )
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
    return None


def _hail_mary_grow(target: Path, corpus: Path, iters: int, inprocess_func: str) -> Path:
    """Run the fuzzer CLI in hail-mary mode on a copy of a corpus.

    The fuzzer mutates and grows the corpus in place; the original is left
    untouched per the corpus rules. ``iters`` is an execution budget passed
    to the campaign as ``--max-execs`` (a real exec count, not ``-n``
    iterations -- one iteration runs a whole mutation budget). Returns the
    grown copy so the matrix analysis can be run over the edges the fuzzer
    actually discovered.
    """
    grown = Path(tempfile.mkdtemp(prefix="edge_diag_hm_")) / "corpus"
    grown.mkdir()
    for p in corpus.rglob("*"):
        if p.is_file():
            shutil.copy2(p, grown / p.name)
    proc = "fuzzer-tool"
    cmd = [
        proc,
        "fuzz",
        str(target),
        "-d",
        str(grown),
        "--max-execs",
        str(iters or 1),
        "--hail-mary",
        "--inprocess",
        "--inprocess-func",
        inprocess_func,
        "-o",
        str(grown.parent / "crashes"),
    ]
    print(f"[*] hail-mary campaign: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.expanduser("~"), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"hail-mary campaign failed with rc={result.returncode}")
    for p in grown.glob("*.json"):
        if p.name in STATESTORE_JSON_FILES:
            p.unlink()
    n = sum(1 for p in grown.rglob("*") if p.is_file())
    print(f"[*] hail-mary corpus now has {n} inputs")
    return grown


def _matrix_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--target", type=Path, help="instrumented target taking an input path in argv[1]"
    )
    ap.add_argument("--corpus", type=Path, help="directory of inputs to run (one execution each)")
    ap.add_argument("--load", type=Path, help="re-analyse a collection saved with --save")
    ap.add_argument("--save", type=Path, help="write the collected columns to an .npz")
    ap.add_argument("--json", type=Path, help="write the full result dict as JSON")
    ap.add_argument("--map-size", type=int, default=65536, help="SHM table entries (default 65536)")
    ap.add_argument(
        "--ctx-bits",
        type=int,
        default=8,
        help="__AFL_CTX_BITS the target was built with (default 8; 0 for a "
        "__AFL_CTX_SENSITIVE=0 build)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="executions of one input for the stability check (0 to skip)",
    )
    ap.add_argument(
        "--keep-aslr",
        action="store_true",
        help="do not disable ASLR (raw-address ctx regime; export "
        "FUZZER_KEEP_ASLR=1 too for what a production run with ASLR on sees)",
    )
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--transpose",
        action="store_true",
        help="run sections [4]-[6] on the edge x seed matrix and add section [7]",
    )
    ap.add_argument(
        "--lll",
        action="store_true",
        help="run section [6]: integer relations over the raw counts (slow)",
    )
    ap.add_argument(
        "--lll-rows",
        type=int,
        default=LLL_ROW_BUDGET,
        help=f"distinct rows fed to LLL (default {LLL_ROW_BUDGET}; cost is superlinear)",
    )
    ap.add_argument("--perms", type=int, default=2000, help="permutation null samples")
    ap.add_argument(
        "--ground-truth",
        type=Path,
        metavar="TRACER_BIN",
        help="run section [9]: the same inputs through a build of the target made "
        "with tools/ground_truth_tracer.c in place of the shim, and compare the "
        "real edges it logs with the ids --target reported",
    )
    ap.add_argument(
        "--positions",
        action="store_true",
        help="run section [8]: the (edge_pos, edge_id, count) placement matrix",
    )
    ap.add_argument(
        "--flow",
        action="store_true",
        help="run section [10] (needs --ground-truth): which count relations are flow "
        "conservation on the tracer's walk graph, and which hold on this corpus only",
    )
    ap.add_argument(
        "--flaky",
        type=int,
        default=0,
        metavar="K",
        help="section [11]: rerun every input K times (>= 2, needs --target/--corpus) "
        "and report edges whose presence or count varies across repeats",
    )
    ap.add_argument("--subsumption", action="store_true", help="section [12]: subset order")
    ap.add_argument(
        "--admission-replay",
        action="store_true",
        help="section [13]: replay the corpus through edge / bucket / maxcount admission",
    )
    ap.add_argument(
        "--rarefaction", action="store_true", help="section [14]: accumulation curve, Chao2 check"
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="section [15]: N seed resamples of the headline statistics",
    )
    ap.add_argument(
        "--length-confound",
        action="store_true",
        help="section [16]: is hit volume input size (needs sizes: re-collect with --save)",
    )
    ap.add_argument(
        "--prefix-fold", action="store_true", help="section [17]: fold the ctx tag bits away"
    )
    ap.add_argument(
        "--score-audit", action="store_true", help="section [18]: screen seed scores (kills only)"
    )
    ap.add_argument(
        "--all-offline",
        action="store_true",
        help="sections [12]-[18]: everything that needs only the collected matrix",
    )
    ap.add_argument(
        "--resamples",
        type=int,
        default=100,
        help="random orders for --admission-replay and --rarefaction (default 100)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--hail-mary",
        action="store_true",
        help="before collecting, run the fuzzer CLI in hail-mary mode against a "
        "copy of --corpus for --iters executions and analyze the grown corpus",
    )
    ap.add_argument(
        "--hm-target",
        type=Path,
        help="target for the hail-mary campaign. Must be a .so loadable by the "
        "fuzzer's in-process mode (hail-mary force-enables it); for a PIE build "
        "pass its _asan.so sibling. Defaults to --target when --target ends in "
        ".so",
    )
    ap.add_argument(
        "--hm-inprocess-func",
        default="fuzz_shm_run",
        help="symbol the campaign's in-process mode calls (default fuzz_shm_run)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=0,
        help="target executions for the hail-mary fuzzer campaign (0 = unlimited; "
        "a real --max-execs budget, not -n iterations; keep it bounded)",
    )
    args = ap.parse_args(argv)

    if args.load is None and (args.target is None or args.corpus is None):
        ap.error("either --load, or both --target and --corpus")

    if args.flow and args.ground_truth is None:
        ap.error("--flow needs --ground-truth")

    if args.hail_mary and args.load is not None:
        ap.error("--hail-mary grows a corpus; it is incompatible with --load")

    if args.hail_mary and args.hm_target is None and str(args.target).endswith(".so") is False:
        sibling = _so_sibling(args.target)
        if sibling is not None:
            args.hm_target = sibling
        else:
            ap.error(
                "--hail-mary needs a .so campaign target: pass --hm-target "
                "fuzzgoat_read_nosan.so (hail-mary force-enables in-process mode, "
                "which cannot dlopen a PIE executable, and an ASAN .so double-loads "
                "libasan)"
            )

    if args.all_offline:
        args.subsumption = args.admission_replay = args.rarefaction = True
        args.length_confound = args.prefix_fold = args.score_audit = True
        args.bootstrap = args.bootstrap or DEFAULT_BOOTSTRAP

    if args.flaky and (args.load is not None or args.flaky < MIN_FLAKY_REPEATS):
        ap.error(f"--flaky needs a live --target/--corpus and K >= {MIN_FLAKY_REPEATS}")

    if args.load is not None:
        runs_all = load_runs(args.load)
        sizes = load_sizes(args.load)
        stability = None
    else:
        if args.hail_mary:
            hm_target = args.hm_target or args.target
            args.corpus = _hail_mary_grow(
                hm_target, args.corpus, args.iters, args.hm_inprocess_func
            )
        if not args.keep_aslr:
            disable_aslr()
        inputs = sorted(p for p in args.corpus.rglob("*") if p.is_file())
        if not inputs:
            ap.error(f"no inputs under {args.corpus}")
        runs_all = collect(args.target, inputs, args.map_size, args.timeout)
        sizes = np.array([p.stat().st_size for p in inputs], dtype=np.int64)
        # Probe the input with the most live edges, not sorted()[0]: on the
        # fuzzgoat corpus that is the empty file, whose 2 edges (both from the
        # wrapper, no parser code) made a Jaccard of 1.000 vacuous.
        richest = max(range(len(inputs)), key=lambda i: len(runs_all[i][-2]))
        stability = (
            measure_stability(
                args.target, inputs[richest], args.repeats, args.map_size, args.timeout
            )
            if args.repeats > 1
            else None
        )
        if stability is not None:
            stability["input"] = inputs[richest].name
        if args.save is not None:
            save_runs(args.save, runs_all, args.map_size, sizes)

    if runs_all and len(runs_all[0]) == 3:
        pos_runs = runs_all
        runs = [(ids, counts) for _, ids, counts in runs_all]
    else:
        pos_runs = None
        runs = runs_all
    collected_map = saved_map_size(args.load) if args.load is not None else args.map_size

    agg = aggregate(runs)
    mat = seed_edge_matrix(runs)
    smat = mat  # sections [11]-[18] always read seed x edge, whatever --transpose does
    # The spectrum and both ranks are transpose-invariant; everything derived
    # from them is not. See the handover's F10 for the comparison.
    if args.transpose:
        mat = mat.T
    if len(agg["ids"]) == 0:
        print("no edges recorded -- is the target instrumented and linked against the shim?")
        return 1

    live = sorted(len(ids) for ids, _ in runs)
    result = {
        "collection": {
            "inputs": len(runs),
            "median_live_edges": live[len(live) // 2],
            "union_edges": int(len(agg["ids"])),
            "total_hits": int(agg["total"].sum()),
            "aslr_disabled": (not args.keep_aslr) if args.load is None else None,
            "orientation": "edge x seed" if args.transpose else "seed x edge",
        },
        "axis": axis_structure(agg["ids"], agg["total"], args.ctx_bits, args.perms, args.seed),
        "y_marginal": y_marginal(agg["total"]),
        "substituted_axes": substituted_axes(agg),
        "spectrum": spectrum(mat),
        "gf2": gf2_structure(mat),
    }
    if args.transpose:
        result["duplicate_classes"] = duplicate_classes(mat, agg["ids"], args.ctx_bits)
    if args.lll:
        result["integer_relations"] = integer_relations(mat, args.lll_rows)
    if args.positions:
        result["positions"] = positions_structure(pos_runs, collected_map, args.perms, args.seed)
    if stability is not None:
        result["stability"] = stability
    if args.ground_truth is not None:
        if args.load is not None:
            ap.error("--ground-truth needs the inputs; it is incompatible with --load")
        if not os.access(args.ground_truth, os.X_OK):
            ap.error(f"--ground-truth {args.ground_truth}: not an executable tracer build")
        gt_runs = _gt_runs(args.ground_truth, inputs, args.timeout)
        gt = ground_truth(args.ground_truth, inputs, args.timeout, runs=gt_runs)
        gt["ids"] = int(len(agg["ids"]))
        gt["ctx_bits"] = args.ctx_bits
        result["ground_truth"] = gt
        if args.flow:
            result["flow"] = flow_structure(gt_runs, args.lll_rows if args.lll else 0)
            shim = seed_edge_matrix(runs)
            result["flow"]["profile_match"] = {
                k.value: _profile_match(shim, _flow_graph(gt_runs, k)["ac"]) for k in FlowGraph
            }
    if args.flaky:
        if args.load is not None:
            ap.error("--flaky needs a live --target/--corpus; it is incompatible with --load")
        reps = collect_repeats(args.target, inputs, args.flaky, args.map_size, args.timeout)
        result["flaky"] = emm.flaky_edges(reps)
    result.update(_offline_sections(args, smat, agg["ids"], sizes))
    _matrix_report(result)
    emm.print_sections(result)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True, default=_np_default))
        print(f"\nwrote {args.json}")
    return 0


def _headline(mat):
    """Headline statistics of one seed resample, for ``emm.bootstrap_ci``."""
    live = mat[:, mat.any(axis=0)]
    if live.size == 0:
        return dict.fromkeys(HEADLINE_KEYS, 0.0)
    g = gf2_structure(live)
    p = live.sum(axis=0) / live.sum()
    greedy = g["greedy_cover_seeds"]
    return {
        "union_edges": float(live.shape[1]),
        "effective_edges": float(2.0 ** -(p * np.log2(p)).sum()),
        "gf2_rank": float(g["gf2_rank"]),
        "real_rank": float(g["real_rank"]),
        "distinct_rows": float(g["distinct_rows"]),
        "greedy_cover": math.nan if greedy is None else float(greedy),
        "edge_classes": float(len({c.tobytes() for c in live.T})),
    }


HEADLINE_KEYS = (
    "union_edges", "effective_edges", "gf2_rank", "real_rank", "distinct_rows",
    "greedy_cover", "edge_classes",
)  # fmt: skip


def _np_default(obj):
    """json.dumps hook: numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _offline_sections(args, smat, ids, sizes) -> dict:
    """The analyses that need only the collected matrix, keyed by result section."""
    out = {}
    if args.subsumption:
        out["subsumption"] = emm.subsumption(smat, ids)
    if args.admission_replay:
        out["admission"] = emm.admission_replay(smat, args.resamples, args.seed)
    if args.rarefaction:
        out["rarefaction"] = emm.rarefaction(smat, args.resamples, args.seed)
    if args.bootstrap:
        out["bootstrap"] = emm.bootstrap_ci(smat, _headline, args.bootstrap, args.seed)
    if args.length_confound:
        out["length_confound"] = emm.length_confound(smat, ids, sizes)
    if args.prefix_fold:
        out["prefix_fold"] = emm.prefix_fold(smat, ids, args.ctx_bits)
    if args.score_audit:
        out["score_audit"] = emm.score_audit(smat, ids)
    return out


import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import edge_matrix_modes as emm  # noqa: E402

from fuzzer_tool.adapters.shm import ShmCoverage  # noqa: E402
from fuzzer_tool.core.edge_tracker import EdgeTracker  # noqa: E402
from fuzzer_tool.services.seed_picker import RARE_EDGE_OWNERS, SeedPicker  # noqa: E402

STEADY_RERUNS = 3
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAP_SIZE = 65536
DEFAULT_SHUFFLES = 2
DEFAULT_SEED = 3


def _ids(target: Path, cov: ShmCoverage, path: Path, timeout: float) -> frozenset[int]:
    return frozenset(_run_one(target, cov, path, timeout)[0].tolist())


def steady_state(target: Path, path: Path, timeout: float) -> frozenset[int]:
    """Id set of *path* on a warmed table; asserts the reruns agree."""
    cov = ShmCoverage(size=DEFAULT_MAP_SIZE)
    try:
        _run_one(target, cov, path, timeout)
        runs = [_ids(target, cov, path, timeout) for _ in range(STEADY_RERUNS)]
    finally:
        cov.cleanup()
    if len(set(runs)) != 1:
        raise SystemExit(f"{path.name}: reruns disagree; target is not deterministic")
    return runs[0]


def production(target: Path, files: list[Path], order: list[int], timeout: float):
    """One persistent table, one discarded warm-up, inputs in *order*."""
    cov = ShmCoverage(size=DEFAULT_MAP_SIZE)
    try:
        _run_one(target, cov, files[order[0]], timeout)
        return [_ids(target, cov, files[i], timeout) for i in order]
    finally:
        cov.cleanup()


def success_events(obs, steady) -> dict[str, int]:
    """Executions that add a new id, observed vs steady, in arrival order."""
    seen_o: set[int] = set()
    seen_s: set[int] = set()
    out = Counter()
    for o, s in zip(obs, steady, strict=True):
        new_o, new_s = o - seen_o, s - seen_s
        out["observed"] += bool(new_o)
        out["steady"] += bool(new_s)
        out["phantom_only"] += bool(new_o) and not new_s
        out["carry_phantoms"] += bool(o - s)
        seen_o |= o
        seen_s |= s
    out["phantom_ids"] = len(seen_o - seen_s)
    out["ids"] = len(seen_o)
    return dict(out)


def rare_ownership(obs, steady) -> dict[str, int]:
    """How many of the rarest edges exist only as phantoms."""
    own_o = Counter(e for o in obs for e in o)
    own_s = Counter(e for s in steady for e in s)
    phantom = set(own_o) - set(own_s)
    return {
        "singletons": sum(v == 1 for v in own_o.values()),
        "singletons_phantom": sum(own_o[e] == 1 for e in phantom),
        "rare": sum(v <= RARE_EDGE_OWNERS for v in own_o.values()),
        "rare_phantom": sum(own_o[e] <= RARE_EDGE_OWNERS for e in phantom),
    }


def _weights(sets) -> list[float]:
    """Per-seed weights through the production rarity code path."""
    tracker = EdgeTracker(map_size=DEFAULT_MAP_SIZE)
    for i, s in enumerate(sets):
        tracker.record_edges(f"s{i}", set(s))
    fake = SimpleNamespace(_edge_tracker=tracker, _recent_seed_edges=None, _rng=None)
    picker = SeedPicker(fake)
    return [picker._weight_edge_penalties(f"s{i}", 1.0, 0, fake) for i in range(len(sets))]


def weight_shift(obs, steady) -> dict[str, float]:
    """Seeds whose weight the phantoms raise, and by how much at most."""
    w_o, w_s = _weights(obs), _weights(steady)
    up = [o / s for o, s in zip(w_o, w_s, strict=True) if o > s + 1e-9]
    return {"boosted": len(up), "max_boost": max(up, default=1.0)}


def _phantom_report(name: str, obs, steady) -> None:
    n = len(obs)
    ev, rare, wt = (
        success_events(obs, steady),
        rare_ownership(obs, steady),
        weight_shift(obs, steady),
    )
    print(f"[{name}]")
    print(f"  execs carrying phantom ids   {ev['carry_phantoms']}/{n}")
    print(
        f"  new-edge successes           observed {ev['observed']}  steady {ev['steady']}  phantom-only {ev['phantom_only']}"
    )
    print(f"  phantom ids in cumulative    {ev['phantom_ids']} of {ev['ids']}")
    print(
        f"  singleton edges              {rare['singletons']}  phantom {rare['singletons_phantom']}"
    )
    print(
        f"  owner<={RARE_EDGE_OWNERS} edges              {rare['rare']}  phantom {rare['rare_phantom']}"
    )
    print(f"  seeds boosted by phantoms    {wt['boosted']}/{n}  max boost x{wt['max_boost']:.2f}")


def _phantom_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--shuffles", type=int, default=DEFAULT_SHUFFLES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    disable_aslr()
    files = sorted(p for p in args.corpus.expanduser().iterdir() if p.is_file())
    steady = [steady_state(args.target, p, args.timeout) for p in files]
    rng = np.random.default_rng(args.seed)
    orders = {"corpus order": list(range(len(files)))}
    for k in range(args.shuffles):
        orders[f"shuffle {k}"] = rng.permutation(len(files)).tolist()

    for name, order in orders.items():
        obs = production(args.target, files, order, args.timeout)
        _phantom_report(name, obs, [steady[i] for i in order])
    return 0


MODES = {
    "stored-ids": stored_ids_mode,
    "per-input-sweep": per_input_sweep_mode,
    "fresh-sweep": fresh_sweep_mode,
    "collect-dump": collect_dump_mode,
    "full-table": full_table_mode,
    "trace-stored": trace_stored_mode,
    "view-mismatch": view_mismatch_mode,
    "mixed-map": mixed_map_mode,
    "env-layout": env_layout_mode,
    "env-pad": env_pad_mode,
    "malloc-tunables": malloc_tunables_mode,
    "mappad": mappad_mode,
    "strace-map": strace_map_mode,
    "gdb-guards": gdb_guards_mode,
    "fire-trace": fire_trace_mode,
    "fire-trace-build": fire_trace_build,
    "mem-stats": mem_stats_mode,
    "mem-native": mem_native_mode,
    "mem-curve": mem_curve_mode,
    "mem-plateau": mem_plateau_mode,
    "mem-snapshot": mem_snapshot_mode,
    "mem-heap-attr": mem_heap_attr_mode,
    "mem-long": mem_long_mode,
    "mem-typehist": mem_typehist_mode,
    "mem-smaps": mem_smaps_mode,
    "mem-smaps2": mem_smaps2_mode,
    "mem-trim": mem_trim_mode,
    "mem-shm": mem_shm_mode,
    "op-caches": op_caches_mode,
    "run-sanity": run_sanity_mode,
}


_EDGE_PROBE_MODES = frozenset(
    {
        "per-input-sweep",
        "fresh-sweep",
        "collect-dump",
        "full-table",
        "view-mismatch",
        "mixed-map",
        "env-layout",
        "env-pad",
        "malloc-tunables",
        "mappad",
        "strace-map",
        "gdb-guards",
    }
)


def _mode_requirements(mode, args):
    """What a local-only mode reads, as (description, env var, present?) triples.

    Checked before dispatch so a fresh clone gets the variable to set rather
    than an IndexError on an empty corpus glob or a FileNotFoundError from a
    subprocess.  ``fire-trace-build`` only prints a recipe and needs nothing.
    """
    need = []
    corpus = ("probe corpus (>= 6 inputs)", "EDGE_DIAG_CORPUS", len(FILES) >= 6)
    if mode in _EDGE_PROBE_MODES:
        need += [("ctx target", "EDGE_DIAG_CTX_TARGET", os.path.exists(CTX_TARGET)), corpus]
    if mode == "stored-ids":
        need.append(corpus)  # the binaries are --target/--second-target; it skips missing ones
    if mode == "trace-stored":
        need += [("debug target", "EDGE_DIAG_DBG_TARGET", os.path.exists(DBG_TARGET)), corpus]
    if mode == "fire-trace":
        need += [
            ("fire-trace target", "EDGE_DIAG_TRACE_TARGET", os.path.exists(EDGE_TRACE_TARGET)),
            corpus,
        ]
    if mode == "mappad":
        need.append(("pad library", "EDGE_DIAG_LIBPAD", os.path.exists(LIBPAD)))
    if mode in ("strace-map", "gdb-guards"):
        tool = "strace" if mode == "strace-map" else "gdb"
        need.append((f"{tool} on PATH", "PATH", shutil.which(tool) is not None))
        need.append(("scratch dir", "EDGE_DIAG_SCRATCH", os.path.isdir(SCRATCH)))
    if mode.startswith("mem-") or mode in ("op-caches", "run-sanity"):
        need.append(("memory-mode target", "EDGE_DIAG_PROF_TARGET", os.path.exists(PROF_TARGET)))
    return need


def _preflight(mode, args):
    missing = [(d, v) for d, v, ok in _mode_requirements(mode, args) if not ok]
    for desc, var in missing:
        print(f"edge_diagnostic {mode}: missing {desc} -- set {var}", file=sys.stderr)
    return not missing


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("matrix", "phantom"):
        if sys.argv[1] == "matrix":
            return _matrix_main(sys.argv[2:])
        return _phantom_main(sys.argv[2:])
    parser = argparse.ArgumentParser(
        prog="edge_diagnostic",
        description="Consolidated fuzzgoat coverage-determinism + fuzzer memory diagnostics. "
        "matrix/phantom: deep analysis, portable.  The modes below are local-only; "
        "their inputs come from EDGE_DIAG_* (see the module docstring).",
    )
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--target", default=CTX_TARGET, help="stored-ids first binary")
    parser.add_argument("--second-target", default=DBG_TARGET, help="stored-ids second binary")
    parser.add_argument(
        "--inputs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="per-input-sweep / fresh-sweep input indices",
    )
    parser.add_argument(
        "--maps", type=int, nargs="+", default=[8192, 512], help="collect-dump maps"
    )
    parser.add_argument("--input", type=int, default=1, help="collect-dump input index")
    parser.add_argument(
        "--pads", type=int, nargs="+", default=[0, 8, 64, 512, 4096], help="env-pad sweep sizes"
    )
    parser.add_argument("--items", type=int, default=200, help="mem-snapshot per phase iterations")
    parser.add_argument("--iters", type=int, default=5000, help="mem-stats total iterations")
    parser.add_argument(
        "--pin", action="store_true", help="fresh-sweep: pin ASLR (matches det_probe)"
    )
    args = parser.parse_args()

    if not _preflight(args.mode, args):
        return 2
    MODES[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
