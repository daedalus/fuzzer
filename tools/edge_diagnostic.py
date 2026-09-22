"""Consolidated diagnostic tool for the fuzzgoat edge-coverage determinism
probes (P1-4) and the fuzzer memory/leak profilers.

Consolidates every scratch probe formerly living at /tmp/opencode/*.py into a
single CLI.  The originals are preserved untouched at /tmp/opencode/*.py; this
file is the only new artifact.

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
    mappad           pad_mb_probe.py       (needs /tmp/libpad.so)
    strace-map       strace_probe.py       (needs strace; shmat/mmap addrs)
    gdb-guards       gdb_guard_capture.py  (needs gdb; __sanitizer_cov stream)
    fire-trace       fire_trace.py         (needs /tmp/opencode/fuzzgoat_dbg)
    fire-trace-build (build recipe for the ___AFL_EDGE_TRACE debug target)

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
import subprocess
import sys
import threading
import tracemalloc

import numpy as np

SRCDIR = "/home/dclavijo/my_code/fuzzer-new/src"
sys.path.insert(0, SRCDIR)

logging.disable(logging.INFO)

from fuzzer_tool.adapters import shm as shmmod  # noqa: E402
from fuzzer_tool.adapters.process import disable_aslr  # noqa: E402
from fuzzer_tool.adapters.shm import ShmCoverage  # noqa: E402
from fuzzer_tool.services.fuzzer import Fuzzer  # noqa: E402

CTX_TARGET = "/home/dclavijo/fuzzing/builds/fuzzgoat_read"
DBG_TARGET = "/tmp/opencode/fuzzgoat_dbg_dist"
EDGE_TRACE_TARGET = "/tmp/opencode/fuzzgoat_dbg"
PROF_TARGET = "/home/dclavijo/fuzzing/builds/test_target_v3"
PROF_CORPUS = "/home/dclavijo/fuzzing/prof_corpus"
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

FILES = sorted(glob.glob("/tmp/opencode/cj40/*"))

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
    if not os.path.exists("/tmp/libpad.so"):
        print(
            "missing /tmp/libpad.so (build from /tmp/pad.c: "
            "clang -shared -fPIC -O1 /tmp/pad.c -o /tmp/libpad.so)"
        )
    for pad in (0, 2, 8, 64):
        for m in (512, 8192):
            cov = ShmCoverage(size=m)
            env = dict(
                os.environ,
                __AFL_SHM_ID=str(cov.shm_id),
                AFL_MAP_SIZE=str(m),
                LD_PRELOAD="/tmp/libpad.so",
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
        log = f"/tmp/opencode/strace_{m}.log"
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
        with open("/tmp/opencode/gdb_cmd.txt", "w") as _fh:
            _fh.write(script)
        r = subprocess.run(
            ["gdb", "-batch", "-x", "/tmp/opencode/gdb_cmd.txt", CTX_TARGET],
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
    print("        -o /tmp/opencode/fuzzgoat_dbg -lm")


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


def main():
    parser = argparse.ArgumentParser(
        prog="edge_diagnostic",
        description="Consolidated fuzzgoat coverage-determinism + fuzzer memory diagnostics. "
        "Consolidates /tmp/opencode/*.py; originals untouched.",
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

    MODES[args.mode](args)


if __name__ == "__main__":
    main()
