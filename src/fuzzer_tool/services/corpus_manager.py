"""Corpus persistence, state management, and minimization.

Extracted from Fuzzer class (~lines 648-783, 1845-2231). Contains:
- load_corpus() — load corpus from disk
- init_seed_metadata() — initialize per-seed tracking
- seed_key() — hash a seed for tracking
- save_state() — persist fuzzer state for resume
- load_state() — restore fuzzer state
- save_crash() — save a crash with metadata
- save_to_corpus() — add a new seed to corpus

Signal name mapping for crash return codes.
"""

import hashlib
import logging
import math
import os
import shutil
import struct
from array import array
from enum import Enum
from pathlib import Path

import xxhash

from fuzzer_tool.adapters import seed_zip
from fuzzer_tool.adapters.filesystem import (
    load_corpus,
    save_crash,
    save_irreplaceable,
    save_timeout_seed,
    save_to_corpus,
)
from fuzzer_tool.core.byte_entropy import CumulativeByteEntropy
from fuzzer_tool.core.clock import clock_of
from fuzzer_tool.core.cost_ledger import seed_exec_us
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.periodicity import estimate_record_size
from fuzzer_tool.core.pool_drift import PoolDrift
from fuzzer_tool.core.rate_distortion import RateDistortionCorpus
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.set_cover import min_cover
from fuzzer_tool.core.size_bloat import seed_size_bloat
from fuzzer_tool.services.crash_explain import explain_static
from fuzzer_tool.services.operators import HAVOC_SUB_OPS

log = logging.getLogger(__name__)


class PoissonAdmissionDecision(Enum):
    """Outcome of Poisson-disk admission check."""

    ADMIT = "admit"
    ADMIT_NEAR_DUP = "admit_near_dup"
    REJECT_NEAR_DUP = "reject_near_dup"


class PoissonDiskAdmission:
    """Proactive corpus diversity gate via MinHashLSH disk-exclusion.

    Bridges Bridson/Mitchell Poisson-disk sampling into the fuzzer's save_to_corpus
    path.  Instead of admitting everything and pruning near-duplicates reactively,
    we query MinHashLSH.find_similar() at admission time and reject seeds whose
    edge-signature is closer than the Jaccard radius to any already-admitted
    corpus member.

    Rare-edge safety valve: a seed is admitted even if it is Jaccard-similar when it
    contributes at least one edge not owned by any of its similar neighbors.  This
    prevents rare/frontier edge coverage from being discarded under redundancy pressure.
    """

    def __init__(self, fuzzer, min_jaccard: float = 0.25):
        self._fuzzer = fuzzer
        self._min_jaccard = min_jaccard

    def _minhash(self):
        return self._fuzzer._edge_tracker._minhash

    def _admitted_keys(self) -> set[str]:
        return self._fuzzer._admitted_keys

    def check(self, data: bytes, seed_key: str) -> PoissonAdmissionDecision:
        """Determine whether *seed_key* may be admitted to the corpus.

        Must be called after the seed's edge signature has already been registered
        in the MinHashLSH index (record_edges runs before save_to_corpus in fuzz_one).
        The find_similar result is filtered against the admitted_keys set so the seed
        does not match itself.
        """
        admitted = self._admitted_keys()
        minhash = self._minhash()

        # Step 1: LSH similarity query — returns all seeds sharing ≥1 band.
        # Filtering against admitted_keys is essential: record_edges already
        # registered this seed's signature before save_to_corpus is called, so
        # an unfiltered find_similar would always match the seed against itself.
        all_similar = minhash.find_similar(seed_key, min_jaccard=self._min_jaccard)
        similar_neighbors = all_similar & admitted
        if not similar_neighbors:
            # Disk empty — admit normally and track this seed's LSH occupancy.
            self._fuzzer._admitted_keys.add(seed_key)
            self._track_occupancy(seed_key, minhash)
            return PoissonAdmissionDecision.ADMIT

        # Step 2: Similar neighbors exist.  Check rare-edge safety valve:
        # admit if this seed owns any edge not already covered by neighbors.
        if self._has_unique_edges(seed_key, similar_neighbors):
            self._fuzzer._admitted_keys.add(seed_key)
            self._track_occupancy(seed_key, minhash)
            return PoissonAdmissionDecision.ADMIT_NEAR_DUP

        # No unique edges and within disk radius of admitted members → reject.
        return PoissonAdmissionDecision.REJECT_NEAR_DUP

    def _has_unique_edges(self, seed_key: str, neighbor_keys: set[str]) -> bool:
        """Return True if *seed_key* owns at least one edge no neighbor covers."""
        edge_tracker = self._fuzzer._edge_tracker
        seed_edges = edge_tracker.seed_edges.get(seed_key, set())
        if not seed_edges:
            return False
        neighbor_edges: set[int] = set()
        for nk in neighbor_keys:
            neighbor_edges |= edge_tracker.seed_edges.get(nk, set())
        return bool(seed_edges - neighbor_edges)

    def _track_occupancy(self, seed_key: str, minhash) -> None:
        """Update LSH bucket occupancy for maximality detection."""
        sig = minhash.signatures.get(seed_key)
        if sig is None:
            return
        f = self._fuzzer
        for band_idx in range(minhash.num_bands):
            start = band_idx * minhash.band_size
            end = start + minhash.band_size

            band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
            from fuzzer_tool.core.edge_tracker import crc32_ieee

            band_hash = crc32_ieee(band_bytes)
            bucket_key = (band_idx, band_hash)
            if bucket_key not in f._poisson_occupied_buckets:
                f._poisson_occupied_buckets.add(bucket_key)
                f._poisson_last_new_bucket_exec = f.exec_count

    def occupancy_ratio(self) -> float:
        """Fraction of admitted seeds that touch distinct LSH buckets (grid coverage)."""
        admitted = self._admitted_keys()
        if not admitted:
            return 0.0
        # Each admitted seed touches num_bands buckets.  Distinct bucket count /
        # (admitted_count * num_bands) is the bucket utilisation ratio.
        minhash = self._minhash()
        buckets_touched: set[tuple[int, int]] = set()
        for sk in admitted:
            sig = minhash.signatures.get(sk)
            if sig is None:
                continue

            for band_idx in range(minhash.num_bands):
                start = band_idx * minhash.band_size
                end = start + minhash.band_size
                band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
                from fuzzer_tool.core.edge_tracker import crc32_ieee

                band_hash = crc32_ieee(band_bytes)
                buckets_touched.add((band_idx, band_hash))
        total_slots = len(admitted) * minhash.num_bands
        if total_slots == 0:
            return 0.0
        return len(buckets_touched) / total_slots


def _gdb_crash_replay(f, data: bytes, returncode: int) -> str:
    """Best-effort GDB crash replay text for the report sidecar; '' when unavailable.

    Runs the crashing input once under GDB (only when gdb is installed and the
    target is traceable) so the final crash report carries a real backtrace,
    registers, and fault address. Cost is ~1s on the rare crashing input.
    """
    from fuzzer_tool.core.analyzers.analyzer_trace import CrashTracer

    try:
        tracer = CrashTracer(f.target, timeout=max(5, int(getattr(f, "timeout", 5))))
        if not tracer._has_gdb:
            return ""
        return tracer.gdb_replay(data, returncode).sidecar_block()
    except Exception:
        log.debug("GDB crash replay failed", exc_info=True)
        return ""


_use_xxhash = True

# Extra seed_key memo entries allowed beyond 2x the corpus before a clear.
SEED_KEY_MEMO_SLACK = 1024

# Protected copies kept under corpus/seeds/crashing/ per crash signature.
# Every distinct crashing input used to be written there and marked
# irreplaceable, so a single easily-hit bug grew that directory without bound
# and no pruning path could reclaim it. Keeping a bounded sample per signature
# preserves the reason the directory exists -- triage material for each
# distinct crash -- without letting one signature own the disk.
CRASHING_SEEDS_PER_SIG = 64

SIGNAL_NAMES = {
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    11: "SIGSEGV",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
}


def _returncode_to_signal(returncode: int) -> tuple[str | None, int | None]:
    """Map a subprocess return code to (signal_name, signal_number).

    Returns (None, None) for non-signal exit codes.
    Handles both WIFSIGNALED-style codes (-signum) and
    exit-with-signal codes (128+signum).
    """
    if returncode < 0:
        signum = -returncode
        if signum in SIGNAL_NAMES:
            return SIGNAL_NAMES[signum], signum
    elif returncode >= 128:
        signum = returncode - 128
        if signum in SIGNAL_NAMES:
            return SIGNAL_NAMES[signum], signum
    return None, None


def current_coverage_contract(f) -> dict:
    """The coverage semantics this session runs under. Resume states are
    only valid when every part matches (edge ids change with k and with the
    shim's id scheme; node-channel state is meaningless without the bitmap)."""
    from fuzzer_tool.core.elf import detect_edge_id_scheme, detect_ngram_k

    return {
        "ngram_k": detect_ngram_k(f.target),
        "node_channel": getattr(f, "_katz_channel", None) is not None,
        "edge_ids": detect_edge_id_scheme(f.target),
    }


def check_coverage_contract(saved: dict | None, current: dict) -> None:
    """Refuse resume on any contract mismatch. States written before
    contracts existed carry no section and resume freely."""
    if not saved:
        return
    for key in ("ngram_k", "node_channel"):
        if saved.get(key) != current.get(key):
            raise RuntimeError(
                f"coverage contract mismatch on {key!r}: "
                f"state={saved.get(key)!r} vs current={current.get(key)!r} — "
                "start a fresh corpus dir or rebuild the target to match the saved run"
            )
    # edge_ids joined the contract after the scheme change, so a contract
    # without it was written by a scheme-1 run (or by a scheme-2 build in the
    # short window before this key existed -- refusing that is the safe
    # error). An unreadable target reports None: unknown, not a mismatch.
    cur_ids = current.get("edge_ids")
    saved_ids = saved.get("edge_ids", 1)
    if cur_ids is not None and saved_ids is not None and saved_ids != cur_ids:
        raise RuntimeError(
            f"coverage contract mismatch on 'edge_ids': state={saved_ids!r} vs "
            f"current={cur_ids!r} — the target's shim assigns different edge ids "
            "than the one that wrote this state; start a fresh corpus dir (the "
            "seed files themselves are fine to re-import) or rebuild the target "
            "with the matching shim"
        )


def seed_key(data: bytes) -> str:
    """Content hash used as the persisted ``seed_meta`` key.

    Kept at module level because three call sites outside CorpusManager
    read persisted seed_meta -- cli/commands.py and services/tmin.py among
    them -- and they must agree on the key scheme. They previously did not:
    the entries were keyed by ``seed.hex()`` while the ``parent_key`` values
    stored inside them were already these hashes, so tmin's lineage walk
    looked up a 16-char hash in a map keyed by full seed content and could
    never resolve a chain for any seed longer than 8 bytes.
    """
    if _use_xxhash:
        return xxhash.xxh64(data).hexdigest()[:16]
    return hashlib.sha256(data).hexdigest()[:16]


def _retire_seed_file(corpus_dir: Path, h: str) -> bool:
    """Move the on-disk record for content hash *h* into the ``pruned/`` subtrees.

    Retire rather than unlink: ``rehydrate_by_hash()`` searches
    ``seeds/pruned/`` and ``deltas/pruned/``, so a child seed stored as a
    delta against this one can still be reconstructed afterwards. This is the
    same move ``auto_minimize_corpus()`` performs; that one walks every file
    against a kept-set, this one looks up a single hash.

    Returns True if something was moved.
    """
    if not corpus_dir:
        return False
    corpus_dir = Path(corpus_dir)
    moved = False

    full = corpus_dir / "seeds" / h[:2] / f"id_{h}"
    if full.is_file():
        dest = corpus_dir / "seeds" / "pruned" / h[:2]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(full), str(dest / full.name))
        moved = True

    deltas_dir = corpus_dir / "deltas"
    for delta in (deltas_dir / f"delta_{h}.json", deltas_dir / h[:2] / f"delta_{h}.json"):
        if delta.is_file():
            dest = deltas_dir / "pruned" / h[:2]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(delta), str(dest / delta.name))
            moved = True

    store = seed_zip.lookup(corpus_dir)
    if store is not None and store.retire(h):
        moved = True

    return moved


def _wire_lifetimes(f) -> None:
    """Single-target SHM: date each edge's last hit from the table's
    generation tags (one fold per 256 execs) instead of per-edge writes."""
    shm = getattr(f, "shm_cov", None)
    if shm is None or getattr(f, "multi_targets", None):
        return
    f._edge_tracker.attach_generations(shm.read_generation, shm.entry_tags)
    shm.on_table_loss = f._edge_tracker.fold_tags


class CorpusManager:
    """Manages corpus persistence, state, and minimization.

    Holds a reference to the Fuzzer instance for accessing shared state.

    - trim_new_coverage() — minimize inputs hitting new edges
    - edges_subset_of() — check edge coverage containment
    - auto_minimize_corpus() — hash dedup + subsumption pruning
    - deprioritize_near_duplicates() — merge near-identical seeds
    """

    def __init__(self, fuzzer):
        self.f = fuzzer
        # Corpus seed -> content key. Corpus seeds are long-lived bytes with
        # a cached hash, so a hit skips the xxh64 pass over the seed.
        self._key_memo: dict[bytes, str] = {}

    def _entropy_add(self, data: bytes) -> None:
        """Fold an admitted seed into the byte-entropy readout.

        Absent tracker (state from before it existed) stays absent.
        """
        tracker = getattr(self.f, "_corpus_entropy", None)
        if tracker is None:
            return

        tracker.add(data)

    def _entropy_remove(self, data: bytes) -> None:
        """Unfold a seed that left the corpus from the readout."""
        tracker = getattr(self.f, "_corpus_entropy", None)
        if tracker is None:
            return

        tracker.remove(data)

    def rebuild_entropy(self) -> None:
        """Refold the readout from ``f.corpus`` after a wholesale swap."""
        f = self.f
        if getattr(f, "_corpus_entropy", None) is None:
            return

        f._corpus_entropy = CumulativeByteEntropy()
        for seed in f.corpus:
            f._corpus_entropy.add(seed)

    def load_corpus(self):
        f = self.f
        # Fresh tracker each load: a resume/reload re-reads every seed from
        # disk, so the running totals are rebuilt from that same pass
        # rather than double-counting against whatever was accumulated in
        # a previous process.
        f._corpus_entropy = CumulativeByteEntropy()
        f.corpus, f.seen_hashes, f.irreplaceable_hashes = load_corpus(
            f.corpus_dir, f.bloom, entropy_tracker=f._corpus_entropy
        )
        # Ensure the irreplaceable/ directory exists inside seeds/ so seeds can be
        # promoted to irreplaceable without a late mkdir.
        (f.corpus_dir / "seeds" / "irreplaceable").mkdir(parents=True, exist_ok=True)
        # Same for crashing/: inputs observed to crash the target are stored
        # here and marked irreplaceable, so the dir must exist before the
        # first crash.
        (f.corpus_dir / "seeds" / "crashing").mkdir(parents=True, exist_ok=True)
        # Same for timeouts/: inputs that timed out are stored here and
        # marked irreplaceable, so the dir must exist before the first timeout.
        (f.corpus_dir / "seeds" / "timeouts").mkdir(parents=True, exist_ok=True)

    def init_seed_metadata(self):
        f = self.f
        now = clock_of(f).time()
        f.seed_meta: dict[bytes, dict] = {}
        for seed in f.corpus:
            f.seed_meta[seed] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": now,
                "record_stride": estimate_record_size(seed),
                "seed_passed_det": False,
            }
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        morris_mode = os.environ.get("AFL_MORRIS", "1") != "0"
        f._edge_tracker = EdgeTracker(map_size=f.map_size, morris_mode=morris_mode)
        _wire_lifetimes(f)
        f._corpus_size_history: array = array("I")
        f._seed_size_moments = RunningMoments(window=200)

        # Freeze the seed byte distribution after transforms/boost, before
        # any admission; status and report compare the live pool to it.
        f._pool_drift = None
        if getattr(f, "_use_pool_drift", False):
            f._pool_drift = PoolDrift()
            f._pool_drift.sync(f.corpus)

        if f.resume:
            self.load_state()

    def seed_key(self, data: bytes) -> str:
        """Content key of *data*; corpus seeds are memoized, mutants are not.

        Bounded at twice the corpus plus slack: seeds that left the corpus
        are dropped by clearing when the bound is hit.
        """
        if type(data) is not bytes:
            return seed_key(data)

        memo = self._key_memo
        key = memo.get(data)
        if key is not None:
            return key

        key = seed_key(data)
        meta = getattr(self.f, "seed_meta", None)
        if meta is None or data not in meta:
            return key

        if len(memo) >= 2 * len(meta) + SEED_KEY_MEMO_SLACK:
            memo.clear()
        memo[data] = key
        return key

    def save_state(self):
        f = self.f
        state = {
            "exec_count": f.exec_count,
            "crash_count": f.crash_count,
            "timeout_count": f.timeout_count,
            "crash_sigs": f.crash_sigs,
            "op_counts": f.op_counts,
            "op_success": f.op_success,
            # getattr, unlike the counters above it: save_state() is called
            # with partially-built stand-ins in several tests, and a new
            # required attribute here turns "this counter is new" into
            # "state cannot be saved at all".
            "op_applicable": getattr(f, "op_applicable", {}),
            "op_success_applicable": getattr(f, "op_success_applicable", {}),
            "op_edges": f.op_edges,
            # Havoc sub-mutation credit. Kept as plain lists (the state file
            # is a sanitized pickle, but array("d") would still pin the
            # branch count into the on-disk format); load_state re-pairs
            # them with HAVOC_SUB_OPS by name so adding a branch does not
            # silently shift every count by one slot.
            "havoc_subop_stats": {
                name: (f._operators._havoc_hits[i], f._operators._havoc_trials[i])
                for i, name in enumerate(HAVOC_SUB_OPS)
            },
            "corpus_size_history": list(f._corpus_size_history[-500:]),
            "checksum_learner": getattr(f, "checksum_learner", None)
            and f.checksum_learner.to_dict()
            or None,
            "prng_state_learner": getattr(f, "prng_state_learner", None)
            and f.prng_state_learner.to_dict()
            or None,
            "seed_meta": {},
            "crash_frames": f.crash_frames,
            "crash_min_sizes": f.crash_min_sizes,
            # The command that STARTED this session, not the one running now.
            # cmd_fuzz assigns f.invocation = current argv after the Fuzzer is
            # built (and therefore after load_state has run), so a resumed run
            # holds the --resume command in f.invocation and the original in
            # f.original_invocation. Preferring the latter keeps the first
            # command across a chain of resumes; falling back to the former
            # covers the first save, when no original has been restored yet.
            "invocation": getattr(f, "original_invocation", "") or getattr(f, "invocation", ""),
        }
        for seed, meta in f.seed_meta.items():
            # Keyed by the content hash, not by seed.hex().
            #
            # 289c85f skipped keys >= 256 chars to drop corrupted tracker
            # JSON that had been loaded as corpus seeds, on the stated
            # assumption that "seed keys should be hex hashes (< 256
            # chars)". They were not hashes: seed.hex() is the hex of the
            # seed's whole content, so the guard actually dropped every
            # seed larger than 128 bytes -- and with it the entire meta
            # entry, not just the suspect part. Measured on a live png
            # campaign: the corpus held seeds up to 2858 bytes while the
            # longest persisted key was 234 chars (117 bytes), and no key
            # >= 256 existed in the state at all. Across --resume that
            # lost fuzz_count, coverage_edges, added_at, momentum,
            # lineage_depth, redqueen offsets/matches and the cost ledger
            # for essentially every realistic seed.
            #
            # Hashing makes the assumption true instead of removing the
            # guard and re-admitting the bloat it was reaching for: keys
            # are now 16 chars regardless of seed size, which bounds the
            # state file far more tightly than the 128-byte cliff did.
            # It also makes these keys agree with the parent_key values
            # stored *inside* each entry, which are already seed_key()
            # hashes -- see the lineage walk in services/tmin.py.
            key = self.seed_key(seed)
            rm = meta.get("redqueen_matches", [])
            rm_ser = [[m[0], m[1].hex(), m[2].hex()] for m in rm]
            state["seed_meta"][key] = {
                "fuzz_count": meta["fuzz_count"],
                # The cost ledger. total_time was not persisted before, so a
                # resumed seed carried its restored fuzz_count against a zero
                # numerator and read as the cheapest seed in the corpus for
                # the rest of the campaign. cost_samples travels with it: the
                # two are only meaningful as a pair.
                "total_time": meta.get("total_time", 0.0),
                "cost_samples": meta.get("cost_samples", 0),
                "coverage_edges": meta["coverage_edges"],
                "momentum": meta.get("momentum", 0.0),
                "redqueen_offsets": meta["redqueen_offsets"],
                "redqueen_matches": rm_ser,
                "added_at": meta["added_at"],
                "lineage_depth": meta.get("lineage_depth", 0),
                "hamming_distance": meta.get("hamming_distance", -1),
                "child_count": meta.get("child_count", 0),
                "timed_out": meta.get("timed_out", False),
                "parent_key": meta.get("parent_key"),
                "parent_ops": meta.get("parent_ops", []),
                "parent_sites": meta.get("parent_sites", []),
                "new_edge_count": meta.get("new_edge_count", 0),
                "coverage_edges_baseline": meta.get("coverage_edges_baseline", 0),
                "record_stride": meta.get("record_stride", None),
            }
        # Seeds before state: state must never name seeds not yet on disk.
        seed_zip.flush_all()
        store = f._state_store
        store.set("corpus", state)
        store.set("edge_tracker", f._edge_tracker.to_dict())
        if f._use_elo and f._elo:
            store.set("elo", f._elo.to_dict())
        store.set("sensitivity", f._sensitivity.save())
        store.set("crash_mi", f._crash_mi.save())
        if hasattr(f, "_seed_quality"):
            store.set("seed_quality", f._seed_quality.state_dict())
        store.set("coverage_contract", current_coverage_contract(f))
        if getattr(f, "_katz_channel", None) is not None:
            store.set("katz", f._katz_channel.state_dict())
        # Coverage regime detector state (percolation phase classification)
        if hasattr(f, "_regime"):
            store.set("regime", f._regime.save())
        store.save()

    def load_state(self):
        f = self.f
        check_coverage_contract(
            f._state_store.get("coverage_contract"), current_coverage_contract(f)
        )
        state = f._state_store.get("corpus")
        if not state:
            return
        f.exec_count = state.get("exec_count", 0)
        f._resume_baseline_exec = f.exec_count
        f._last_eps_count = f.exec_count
        f.crash_count = state.get("crash_count", 0)
        f.timeout_count = state.get("timeout_count", 0)
        f.crash_sigs = state.get("crash_sigs", {})
        f.crash_frames = state.get("crash_frames", {})
        f.crash_min_sizes = state.get("crash_min_sizes", {})
        f.op_counts = state.get("op_counts", {})
        f.op_success = state.get("op_success", {})
        # Absent in states written before applicability was tracked. Left
        # empty rather than backfilled from op_counts: the report treats
        # "no entry" as unknown and falls back to the raw count, which is
        # honest, whereas copying op_counts would assert every historic
        # selection was applicable.
        f.op_applicable = state.get("op_applicable", {})
        f.op_success_applicable = state.get("op_success_applicable", {})
        f.op_edges = state.get("op_edges", {})
        # Restored onto a DIFFERENT attribute than f.invocation, which
        # cmd_fuzz overwrites with the current argv immediately after the
        # Fuzzer is constructed. Writing f.invocation here would be clobbered
        # a moment later and the original lost again. Absent in states written
        # before this was persisted, hence the "" default.
        f.original_invocation = state.get("invocation", "")
        self._restore_havoc(state)
        f._operators._rebuild_havoc_table()
        f._corpus_size_history = array("I", state.get("corpus_size_history", []))
        self._restore_seed_meta(state)
        self._restore_subsystems(state)
        if f.resume:
            print(
                f"[*] Resumed: {f.exec_count} execs, {f.crash_count} crashes, {len(f.corpus)} seeds"
            )
        sq_data = f._state_store.get("seed_quality")
        if sq_data is not None and hasattr(f, "_seed_quality"):
            f._seed_quality.load_state_dict(sq_data)
        katz_data = f._state_store.get("katz")
        if katz_data is not None and getattr(f, "_katz_channel", None) is not None:
            f._katz_channel.load_state_dict(katz_data)
        log.info(
            "Fuzzer state loaded: execs=%d, crashes=%d, corpus=%d",
            f.exec_count,
            f.crash_count,
            len(f.corpus),
        )

    def _restore_havoc(self, state: dict) -> None:
        """Restore persisted havoc sub-op hit/trial counts (skip corrupt entries)."""
        f = self.f
        havoc_stats = state.get("havoc_subop_stats") or {}
        for i, name in enumerate(HAVOC_SUB_OPS):
            saved = havoc_stats.get(name)
            if not saved:
                continue
            hits, trials = saved
            # Trials must stay >= 1 or the ratio divides by zero; a corrupt
            # or hand-edited state file should degrade to the prior, not
            # crash the resume.
            if trials >= 1.0 and hits >= 0.0:
                f._operators._havoc_hits[i] = int(hits)
                f._operators._havoc_trials[i] = int(trials)

    def _restore_seed_meta(self, state: dict) -> None:
        """Merge persisted per-seed metadata into f.seed_meta for loaded seeds."""
        f = self.f
        saved_meta = state.get("seed_meta", {})
        for seed in f.corpus:
            # Hash key first, then the legacy seed.hex() key so state files
            # written before the change still restore. Old files only ever
            # held seeds under 128 bytes -- larger ones were dropped at save
            # time -- so the fallback recovers exactly what is there and
            # nothing is silently reinterpreted.
            key = self.seed_key(seed)
            if key not in saved_meta:
                legacy = seed.hex()
                key = legacy if legacy in saved_meta else key
            if key in saved_meta:
                sm = saved_meta[key]
                f.seed_meta[seed].update(
                    {
                        "fuzz_count": sm.get("fuzz_count", 0),
                        # A state file written before the ledger was persisted
                        # has neither key. Restoring 0/0 is correct: zero
                        # samples means "no measurement", which the readers
                        # substitute the corpus mean for, rather than the
                        # 1 microsecond floor that a zero numerator produced.
                        "total_time": sm.get("total_time", 0.0),
                        "cost_samples": sm.get("cost_samples", 0),
                        "coverage_edges": sm.get("coverage_edges", 0),
                        "momentum": sm.get("momentum", 0.0),
                        "redqueen_offsets": sm.get("redqueen_offsets", []),
                        "added_at": sm.get("added_at", f.seed_meta[seed]["added_at"]),
                        "lineage_depth": sm.get("lineage_depth", 0),
                        "hamming_distance": sm.get("hamming_distance", -1),
                        "child_count": sm.get("child_count", 0),
                        "timed_out": sm.get("timed_out", False),
                        "parent_key": sm.get("parent_key"),
                        "parent_ops": sm.get("parent_ops", []),
                        "parent_sites": sm.get("parent_sites", []),
                        "new_edge_count": sm.get("new_edge_count", 0),
                        "coverage_edges_baseline": sm.get("coverage_edges_baseline", 0),
                        "record_stride": sm.get("record_stride", None),
                    }
                )
                rm_ser = sm.get("redqueen_matches", [])
                if rm_ser:
                    f.seed_meta[seed]["redqueen_matches"] = [
                        (m[0], bytes.fromhex(m[1]), bytes.fromhex(m[2])) for m in rm_ser
                    ]

    def _restore_subsystems(self, state: dict) -> None:
        """Restore edge tracker, regime, checksum/PRNG learners and sensitivity."""
        f = self.f
        et_data = f._state_store.get("edge_tracker")
        if et_data is not None:
            f._edge_tracker.from_dict(et_data)
        # Restore coverage regime detector state
        regime_data = f._state_store.get("regime")
        if regime_data is not None and hasattr(f, "_regime"):
            f._regime.load(regime_data)
        # Restore checksum learner state
        cl_data = state.get("checksum_learner")
        if cl_data and hasattr(f, "checksum_learner") and f.checksum_learner is not None:
            from fuzzer_tool.core.analyzers.analyzer_checksum_learner import ChecksumLearner

            f.checksum_learner = ChecksumLearner.from_dict(f, cl_data)
        # Restore PRNG state learner state
        prng_data = state.get("prng_state_learner")
        if prng_data and hasattr(f, "prng_state_learner") and f.prng_state_learner is not None:
            from fuzzer_tool.core.analyzers.analyzer_prng_state_learner import PRNGStateLearner

            f.prng_state_learner = PRNGStateLearner.from_dict(f, prng_data)
        sens_data = f._state_store.get("sensitivity")
        if sens_data is not None:
            f._sensitivity.load(sens_data)

    def save_crash(self, data: bytes, returncode: int, stderr: str) -> str | None:
        f = self.f
        from fuzzer_tool.adapters.filesystem import classify_crash
        from fuzzer_tool.core.crash_metadata import CrashMetadata

        fault_addr = getattr(f, "_last_fault_addr", None)
        # Drain on EVERY crash, not just novel ones: the sink is append-only,
        # so a skipped drain would hand the next crash this one's record.
        sink = getattr(f, "_crash_sym_sink", None)
        shim_sym = sink.drain() if sink is not None else None

        # Decide FIRST whether this crash is one we will keep. Enrichment below
        # (nearest-corpus search over the whole corpus, GDB replay, target
        # hashing) costs from tens of milliseconds to a second per call, and
        # save_crash() throws all of it away for a crash whose signature is
        # already on disk. Crashes are rare only until the first bug is found;
        # after that every mutation of the crashing seed lands here, so paying
        # triage cost per crashing execution collapses throughput exactly when
        # the fuzzer is producing results.
        verdict = classify_crash(
            data,
            returncode,
            stderr,
            f.crash_hashes,
            f.crash_sigs,
            fault_addr=fault_addr,
            crash_blocklist=f.crash_blocklist if f.crash_blocklist else None,
            crash_allowlist=f.crash_allowlist if f.crash_allowlist else None,
        )

        # Preserve the crashing input as corpus material: stored under
        # seeds/crashing/ and marked irreplaceable so no pruning path can drop
        # it. The writer dedups to a stat() for repeats of the same input, but
        # a signature that is trivially reachable produces a fresh input on
        # every execution, so the count is bounded per signature: a novel
        # crash is always preserved, and repeats stop once the signature has
        # CRASHING_SEEDS_PER_SIG samples on disk. Without the bound the
        # directory grows forever and, being irreplaceable, cannot be pruned.
        #
        # The budget is keyed by the signature this crash is COUNTED under,
        # which for a fuzzy match is the existing signature it folds into, not
        # its own: keying by verdict.signature would leave every fuzzy-matched
        # crash reading a count of zero and so exempt from the bound.
        counted_sig = verdict.matched_signature or verdict.signature
        self._keep_crashing_seed(data, verdict, counted_sig)

        self._record_frames(verdict)

        meta: CrashMetadata | None = None
        if verdict.novel:
            meta = self._novel_crash_meta(data, returncode, fault_addr, shim_sym)

        result = save_crash(
            data,
            returncode,
            stderr,
            f.crashes_dir,
            f.crash_hashes,
            f.crash_sigs,
            metadata=meta,
            fault_addr=fault_addr,
            crash_blocklist=f.crash_blocklist if f.crash_blocklist else None,
            crash_allowlist=f.crash_allowlist if f.crash_allowlist else None,
            crash_min_sizes=f.crash_min_sizes if f.save_smaller else None,
            verdict=verdict,
        )
        # Novel crashes return a base name string; duplicates return False.
        #
        # Publish the signature this crash was counted under, and the base
        # name it was written as, for the replay scheduler. Both are already
        # known exactly here; the scheduler used to re-derive the key with
        # `crash_sigs.get(crash_name, crash_name)`, feeding a FILENAME into a
        # signature-keyed dict, so the lookup always missed and the fallback
        # made the filename itself the key (finding #22). _prune_crash_data()
        # pops _crash_replays by signature and so never matched either, and
        # the reproducibility report printed filenames where it labels
        # signatures.
        if result:
            f._last_crash_signature = counted_sig
            # getattr rather than a bare subscript, matching how this function
            # already reaches email_on_crash/_last_regs: several call sites
            # pass a partial fuzzer-like, and a crash write must not fail on a
            # bookkeeping map.
            crash_files = getattr(f, "_crash_files", None)
            if crash_files is not None:
                crash_files[counted_sig] = str(result)
        else:
            f._last_crash_signature = None
        if result and getattr(f, "email_on_crash", None) is not None:
            self._mail_crash(result, returncode, stderr)
        return result

    def _record_frames(self, verdict) -> None:
        """Remember the first valid stack frames seen for each crash signature."""
        f = self.f
        report = verdict.report
        if report and report.is_valid() and verdict.signature not in f.crash_frames:
            f.crash_frames[verdict.signature] = report.frames

    def _keep_crashing_seed(self, data: bytes, verdict, counted_sig: str) -> None:
        """Preserve the crashing input as irreplaceable corpus, bounded per signature."""
        f = self.f
        from fuzzer_tool.adapters.filesystem import save_crashing_seed

        if f.corpus_dir and (
            verdict.novel or f.crash_sigs.get(counted_sig, 0) < CRASHING_SEEDS_PER_SIG
        ):
            save_crashing_seed(data, f.corpus_dir, f.seen_hashes, f.irreplaceable_hashes, f.bloom)

    def _novel_crash_meta(self, data: bytes, returncode: int, fault_addr, shim_sym=None):
        """Build the CrashMetadata sidecar for a novel crash (enrichment is novel-only)."""
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data
        from fuzzer_tool.core.crash_metadata import CrashMetadata, find_nearest_corpus

        meta = CrashMetadata()
        meta.exec_count = f.exec_count
        meta.corpus_size = len(f.corpus)
        meta.target = f.target
        meta.mutation_ops = list(f._last_ops_used)
        meta.parent_sites = [s for _, s in getattr(f, "_last_ops_with_sites", [])]
        meta.elapsed = f._stats.format_elapsed()

        if f.corpus:
            parent = f._last_parent_seed if hasattr(f, "_last_parent_seed") else None
            if parent:
                meta.parent_seed_hash = hash_data(parent)

        if not hasattr(f, "_target_sha256"):
            try:
                f._target_sha256 = hashlib.sha256(Path(f.target).read_bytes()).hexdigest()[:16]
            except Exception:
                f._target_sha256 = "unknown"
        meta.target_sha256 = f._target_sha256

        if f.corpus:
            label, sim, diffs, _ = find_nearest_corpus(data, f.corpus)
            meta.nearest_corpus_file = label
            meta.nearest_similarity = sim
            meta.diff_bytes = diffs

        if hasattr(f, "_last_regs") and (f.ptrace_cov or f._last_regs):
            meta.rip = f._last_regs.get("rip", 0)
            meta.rsp = f._last_regs.get("rsp", 0)
            meta.rbp = f._last_regs.get("rbp", 0)
        if fault_addr is not None:
            meta.fault_addr = f"0x{fault_addr:x}"

        # Shim-side symbolization (--crash-symbolize). Fills only what the
        # ptrace/sanitizer paths left empty. Best-effort: never loses the crash.
        if shim_sym is not None:
            try:
                from fuzzer_tool.core.crash_symbols import hydrate

                hydrate(meta, shim_sym)
            except Exception:
                log.warning("shim crash symbol hydration failed", exc_info=True)

        # Name the crashing input's fields and mark those changed against
        # the parent seed. Cheap (one alignment) and novel-only; the causal
        # search that says which field triggers the crash is a later step.
        # A bug here must not lose the crash, so it is logged, not raised.
        try:
            explain_static(
                meta,
                data,
                parent=getattr(f, "_last_parent_seed", None),
                parent_hash=meta.parent_seed_hash,
                crash_hashes=f.crash_hashes,
                corpus_dir=f.corpus_dir,
                corpus=f.corpus,
                nearest_label=meta.nearest_corpus_file,
            )
        except Exception:
            log.warning("crash field explanation failed", exc_info=True)

        # Which of those fields the crash needs (FIC replay, <= 64 execs, novel
        # crashes only). Opt-in; isolate_for_fuzzer swallows its own errors.
        if getattr(f, "isolate_crash_fields", False):
            from fuzzer_tool.services.crash_isolate import isolate_for_fuzzer

            isolate_for_fuzzer(f, meta, data, returncode)

        # Populate error_type from return code for subprocess/inprocess
        # mode where ptrace isn't available and sanitizer reports are absent.
        if not meta.error_type:
            sig_name, _sig_num = _returncode_to_signal(returncode)
            if sig_name is not None:
                meta.error_type = sig_name

        # Embed the GDB crash replay in the report sidecar (best-effort).
        meta.gdb_replay = _gdb_crash_replay(f, data, returncode)
        return meta

    def _mail_crash(self, result, returncode: int, stderr: str) -> None:
        """Send the crash e-mail; a mail failure never kills the campaign."""
        f = self.f
        try:
            from fuzzer_tool.services.sendmail import send_crash_email

            send_crash_email(
                f.email_on_crash,
                target=str(f.target),
                base_name=str(result),
                crashes_dir=f.crashes_dir,
                returncode=returncode,
                exec_count=getattr(f, "exec_count", 0),
                stderr=stderr or "",
            )
        except Exception as exc:  # never let mail failure kill the campaign
            print(f"[!] crash email failed: {exc}")

    def save_timeout(self, data: bytes) -> None:
        f = self.f
        save_timeout_seed(data, f.corpus_dir, f.seen_hashes, f.irreplaceable_hashes, f.bloom)

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        f = self.f
        parent_depth = self._link_parent(parent)

        f._total_corpus_attempts += 1
        # Compute seed_key early: the Poisson-disk admission check (and the
        # downstream GA population block below) both need it, but the old
        # placement at line 800 was after the check that consumed it.
        seed_key = self.seed_key(data)
        # Set by the Poisson-disk gate below; applied only once seed_meta[data]
        # exists. The entry is created further down by a fresh dict literal,
        # so writing the flag at the gate would KeyError on a new seed (and,
        # for a re-admitted one, be clobbered by that literal anyway).
        is_near_duplicate = False
        if save_to_corpus(
            data,
            f.corpus_dir,
            f.seen_hashes,
            f.bloom,
            parent=parent,
            lineage_depth=parent_depth,
        ):
            # Purely observational bookkeeping for CorpusFlux (P4-T6); must
            # never be able to break a save.
            f._corpus_flux.record_addition()
            # Mix the newly-admitted seed's bytes into the campaign's shared
            # RandPool. This is deliberately gated on save_to_corpus()'s own
            # novelty check (seen_hashes/bloom) above, not on the Poisson-disk
            # near-duplicate check below: "new bytes we hadn't stored before"
            # is the raw-entropy property we want, independent of whether a
            # fuzzy similarity heuristic later decides not to keep it around.
            # Determinism is preserved: for a fixed --seed, the sequence of
            # admitted seeds is itself a deterministic function of the run,
            # so this injection point never introduces run-to-run variance
            # that wasn't already present in which inputs get admitted.
            f._rng.inject_entropy(data)
            if (
                # Under Elo arbitration the corpus-based seed strategies
                # (weighted/pareto/bayesian/boltzmann) read f.corpus; if QEA's
                # bypass froze it, those strategies starve on the initial seeds
                # and the run stalls.  Keep the bypass only for standalone QEA,
                # where the QEA population is the sole seed source.
                not f.qea or getattr(f, "_use_elo", False)
            ):
                # Poisson-disk admission check (proactive near-duplicate rejection).
                # Runs after the bloom/seen-hash novelty gate but before corpus
                # insertion and heavy bookkeeping.  This prevents redundant seeds
                # from entering the pipeline at all.
                decision = self._poisson_gate(data, seed_key)
                if decision == PoissonAdmissionDecision.REJECT_NEAR_DUP:
                    # Record that this seed was rejected for redundancy — but
                    # don't remove edges already tracked by record_edges; those
                    # are part of corpus coverage history.  Just skip corpus entry.
                    return
                if decision == PoissonAdmissionDecision.ADMIT_NEAR_DUP:
                    is_near_duplicate = True

                f.corpus.append(data)
                self._entropy_add(data)
            if f.ga:
                seed_key = self._ga_admit(data)
            f.seed_meta[data] = {
                "fuzz_count": 0,
                "coverage_edges": 0,  # will update below from edge tracker
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": clock_of(f).time(),
                "lineage_depth": parent_depth + 1 if parent else 0,
                "hamming_distance": f._last_hamming_distance,
                "record_stride": estimate_record_size(data),
                "input_size": len(data),
            }
            if is_near_duplicate:
                f.seed_meta[data]["_is_near_duplicate"] = True
            self._inherit_parent(data, parent)
            # Propagate actual coverage_edges from EdgeTracker — when called
            # from fuzz_one, the seed's edges were already recorded by
            # record_edges before save_to_corpus.  For a parentless insert
            # with no prior fuzz_one, edge_count stays 0, which is correct.
            edge_count = len(f._edge_tracker.seed_edges.get(seed_key, set()))
            if edge_count > 0:
                f.seed_meta[data]["coverage_edges"] = edge_count
            # Seed stability calibration (handover item D). Here rather than
            # in fuzz_one so the n_runs re-execution cost is a one-time
            # per-accepted-seed tax instead of a per-iteration one. Runs
            # after the seed is committed, so a seed is never rejected on
            # stability grounds -- unstable edges get masked, the seed stays.
            if getattr(f, "_calibrate_stability", 0):
                f._calibrate_seed_stability(data, n_runs=f._calibrate_stability)
            self._after_admit(data)
        else:
            f._duplicate_reject_count += 1

    def _link_parent(self, parent: bytes | None) -> int:
        """Bump the parent's child_count; return its lineage depth (0 if unknown)."""
        f = self.f
        parent_depth = 0
        if parent is not None:
            parent_meta = f.seed_meta.get(parent)
            if parent_meta is not None:
                parent_depth = parent_meta.get("lineage_depth", 0)
                parent_meta["child_count"] = parent_meta.get("child_count", 0) + 1
        return parent_depth

    def _poisson_gate(self, data: bytes, seed_key: str) -> PoissonAdmissionDecision | None:
        """Poisson-disk near-duplicate check; updates counters. None when disabled."""
        f = self.f
        if not getattr(f, "_use_poisson_disk_admission", False):
            return None

        # Lazy-init PoissonDiskAdmission on first use; the _edge_tracker
        # ._minhash reference is only valid after fuzzer construction completes.
        if f._poisson_admission is None:
            f._poisson_admission = PoissonDiskAdmission(f, f._poisson_disk_min_jaccard)
        decision = f._poisson_admission.check(data, seed_key)
        if decision == PoissonAdmissionDecision.REJECT_NEAR_DUP:
            f._duplicate_reject_count += 1
            f._poisson_reject_count += 1
            f._corpus_flux.record_rejection()
        elif decision == PoissonAdmissionDecision.ADMIT_NEAR_DUP:
            # Admit normally but flag as near-duplicate for deprioritized
            # weighting.  This preserves the seed's edges while signaling
            # it should be weighted lower in seed_key/population selection.
            f._poisson_near_dup_admit_count += 1
            # Drives deprioritize_near_duplicates(): under Poisson
            # admission that reactive scan only runs after 50 of
            # these, so without the increment it never runs.
            f._redundant_admission_count += 1
        return decision

    def _ga_admit(self, data: bytes) -> str:
        """Add *data* to the GA population; return the GA seed key used."""
        f = self.f
        import hashlib as _hashlib

        from fuzzer_tool.core.ga import Individual

        if _use_xxhash:
            seed_key = xxhash.xxh64(data).hexdigest()[:16]
        else:
            seed_key = _hashlib.sha256(data).hexdigest()[:16]
        edge_count = len(f._edge_tracker.seed_edges.get(seed_key, set()))
        ind = Individual(
            data=data,
            edge_count=edge_count,
            generation=f.ga.generation,
            seed_key=seed_key,
        )
        f.ga.add_to_population(ind)
        return seed_key

    def _inherit_parent(self, data: bytes, parent: bytes | None) -> None:
        """Record the lineage edge and Weizz tags inherited from *parent*."""
        f = self.f
        # Lineage edge: parent key + the ops/sites that produced this seed.
        # Only recorded when a real parent exists (interesting/Metropolis
        # paths in fuzz_one). Every in-tree caller now passes one; the
        # parentless branch survives because `parent` is optional on the
        # public Fuzzer.save_to_corpus, so an embedder can still insert a
        # root. Gated on the flag so default runs stay byte-identical.
        if f._use_lineage and parent is not None:
            f.seed_meta[data].update(
                {
                    "parent_key": self.seed_key(parent),
                    "parent_ops": list(getattr(f, "_last_ops_used", [])),
                    "parent_sites": [s for _, s in getattr(f, "_last_ops_with_sites", [])],
                    "new_edge_count": getattr(f, "_last_new_edge_count", 0),
                    "coverage_edges_baseline": 0,
                }
            )
        # Weizz P5: derived-tag inheritance. Length-preserving children
        # reuse the parent StructureMap; length-changing ones inherit
        # a dirty map so P2/P3 skip until the next collector pass.
        if getattr(f, "weizz_tags", False) and parent is not None:
            try:
                from fuzzer_tool.core.weizz_tags import inherit_tags_from_parent

                parent_meta = f.seed_meta.get(parent)
                inherited = inherit_tags_from_parent(parent_meta, parent, data)
                if inherited:
                    f.seed_meta[data].update(inherited)
            except Exception:  # noqa: BLE001 — never block corpus save
                pass

    def _after_admit(self, data: bytes) -> None:
        """Post-admission bookkeeping: Markov, size stats, bloat and max_len tracking."""
        f = self.f
        f.markov.train(data)
        f.markov_trained = f.markov.is_trained()
        # Mutator feedback hook (e.g. wfc_reorder_learned's adjacency tables).
        REGISTRY.notify_new_coverage(data, getattr(f, "_last_new_edge_count", 0))
        if f.markov.snapshot_and_check_plateau():
            log.info(
                "Markov plateau detected (JS=%.4f) — reducing generation rate",
                f.markov.last_js_divergence,
            )
        f._corpus_size_history.append(len(data))
        # The contextual schedulers' corpus-size percentile feature reads
        # this; nothing fed it before, so its guard (count >= 5) never
        # passed and the feature was pinned at the neutral 0.5 for entire
        # runs -- a constant column in a 14-dimensional LinUCB context,
        # which is a second intercept direction rather than a no-op.
        log_size_moments = getattr(f, "_corpus_log_size_stats", None)
        if log_size_moments is not None:
            log_size_moments.update(math.log1p(len(data)))
        seed_moments = getattr(f, "_seed_size_moments", None)
        if seed_moments is not None:
            seed_moments.update(float(len(data)))
        # Bloat early-warning on the *location* of recent sizes (seeds
        # pinned at max_len, or the median doubling across the window).
        # It used to be seed-size skewness > 2, which reads the shape of
        # a heavy-tailed distribution rather than growth: it fired on
        # nearly every check of a stationary lognormal corpus and never
        # on real bloat, which piles sizes at the cap and skews left.
        # See core/size_bloat.py. Rate-limited to once per 500 execs.
        if f.exec_count - f._last_bloat_warn_exec >= 500:
            reason = seed_size_bloat(f._corpus_size_history, f.max_len)
            if reason is not None:
                f._last_bloat_warn_exec = f.exec_count
                log.warning("Corpus bloat warning: %s — minimizing", reason)
                f._defer_minimize()
        if len(f._corpus_size_history) > 1000:
            f._corpus_size_history = f._corpus_size_history[-500:]
        # Display-only (P2-4): the report reads the rule, nothing acts on it.
        if f._corpus_secretary:
            f._corpus_secretary.observe(f._stats.discovery_rate())
        if f.max_corpus > 0 and len(f.corpus) > f.max_corpus:
            f._defer_minimize()
        if len(f._corpus_size_history) >= 100:
            sorted_sizes = sorted(f._corpus_size_history)
            p90 = sorted_sizes[-len(sorted_sizes) // 10]
            # Track the p90 of recent seed sizes in both directions. This
            # was max(f.max_len, ...), a one-way ratchet: once a handful of
            # large seeds pushed p90 up, max_len never came back down, so
            # mutation kept producing larger seeds, which kept p90 up. That
            # is a positive feedback loop into exactly the bloat the
            # warning above (core/size_bloat.py) reports, and minimizing
            # the corpus could not undo it. The configured max_len is the floor.
            f.max_len = min(max(p90 * 2, f._max_len_floor), 65536)

    def _edge_snapshot(self) -> set[int] | None:
        """Edge ids of the last run from SHM or ptrace; None without coverage."""
        f = self.f
        if f.shm_cov:
            return f.shm_cov.get_edge_ids()
        if f.ptrace_cov:
            bm = bytes(f.ptrace_cov.edge_map)
            return {i for i, v in enumerate(bm) if v}
        return None

    def trim_new_coverage(self, data: bytes, parent: bytes) -> None:
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data

        # Crashing/irreplaceable seeds are never trimmed: their exact bytes
        # reproduce the crash, and a halved input may not.
        if hash_data(data) in f.irreplaceable_hashes:
            return

        if len(data) <= 16:
            return

        current_edges = self._edge_snapshot()
        if current_edges is None:
            return

        trimmed = data[: len(data) // 2]

        # Trimmed bytes already a seed (e.g. the parent): swapping would
        # clobber its meta with ours, parent_key == own key -> lineage cycle.
        if trimmed in f.seed_meta:
            return

        rc, _ = f._runner.run_target(trimmed)
        if rc in (-2, -1):
            return
        trimmed_dist = f._exec_distance()

        trimmed_edges = self._edge_snapshot()
        if trimmed_edges is None:
            return

        # AFL rule: keep the trim only if the trace is unchanged.
        if trimmed_edges != current_edges:
            return

        seed_key = self.seed_key(data)
        orig_meta = f.seed_meta.get(data, {})
        if data in f.seed_meta:
            f.seed_meta.pop(data, None)
            f._agg_cache_valid = False  # corpus structure changed
        if data in f.corpus:
            idx = f.corpus.index(data)
            f.corpus[idx] = trimmed
            self._entropy_remove(data)
            self._entropy_add(trimmed)
            # Persist the swap. The in-memory replacement alone lost the seed
            # outright (finding #27): the trimmed bytes were never written, and
            # auto_minimize_corpus() builds its kept-set from f.corpus, so the
            # ORIGINAL file -- whose hash is no longer in that set -- was moved
            # to pruned/ on the next minimize pass. After a resume the corpus
            # held neither. Persistence belongs inside the mutating function,
            # which is the rule this violated.
            #
            # parent=None on purpose: a delta would be encoded against the
            # original, which is being retired in the same breath.
            from fuzzer_tool.adapters.filesystem import save_to_corpus as _save_to_corpus

            _save_to_corpus(trimmed, f.corpus_dir, f.seen_hashes, f.bloom)
            # Retire, not unlink, so any child stored as a delta against the
            # original can still be rehydrated out of pruned/.
            _retire_seed_file(f.corpus_dir, hash_data(data))
            # hash_data(data) deliberately STAYS in seen_hashes. The original
            # was not lost, it was replaced by a smaller input covering the
            # same edges; re-admitting it later would undo the trim on every
            # regeneration.
            f.seed_meta[trimmed] = {
                "fuzz_count": 0,
                "coverage_edges": f._edge_tracker.get_seed_edge_count(seed_key),
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": clock_of(f).time(),
                "lineage_depth": orig_meta.get("lineage_depth", 0) + 1,
            }
            # Directed distance, as admission tags it: the trimmed run's own
            # measurement, else the original's (same trace, same blocks).
            if trimmed_dist is None:
                trimmed_dist = orig_meta.get("avg_distance")
            if trimmed_dist is not None:
                f.seed_meta[trimmed]["avg_distance"] = trimmed_dist
            # The trimmed seed inherits the original's lineage edge so the
            # crash-path chain stays intact across the trim point, with a
            # synthetic ("trim", cut_point) operation appended.
            if f._use_lineage:
                f.seed_meta[trimmed].update(
                    {
                        "parent_key": orig_meta.get("parent_key"),
                        "parent_ops": list(orig_meta.get("parent_ops", [])) + ["trim"],
                        "parent_sites": list(orig_meta.get("parent_sites", [])) + [len(data) // 2],
                        "new_edge_count": orig_meta.get("new_edge_count", 0),
                        "coverage_edges_baseline": orig_meta.get("coverage_edges_baseline", 0),
                    }
                )
            log.debug("Trimmed %d -> %d bytes", len(data), len(trimmed))

    def _mds_select_optional(
        self, scored: list[tuple[float, bytes]], target_size: int, mandatory_count: int
    ) -> list[bytes]:
        """Value-weighted MDS local search over the optional (non-mandatory)
        seed pool, replacing flat top-K-by-score (see core/mds_local_search.py
        for the geometric framing). Falls back to top-K when there's nothing
        for the Jaccard-signature index to work with.
        """
        from fuzzer_tool.core.mds_local_search import disk_radius, local_search_mds

        budget = target_size - mandatory_count
        if budget <= 0 or not scored:
            return []

        minhash = self.f._edge_tracker._minhash
        score_by_key: dict[str, float] = {}
        seed_by_key: dict[str, bytes] = {}
        for score, seed in scored:
            sk = self.seed_key(seed)
            score_by_key[sk] = score
            seed_by_key[sk] = seed

        if not score_by_key:
            keep = min(budget, len(scored))
            return [s for _, s in scored[:keep]]

        smin, smax = min(score_by_key.values()), max(score_by_key.values())
        radius = {k: disk_radius(s, smin, smax) for k, s in score_by_key.items()}
        result = local_search_mds(
            keys=list(score_by_key),
            weight=score_by_key,
            radius=radius,
            jaccard_fn=minhash.approximate_jaccard,
            c=2,
            max_rounds=4,
        )

        picked = result.selected
        if len(picked) > budget:
            picked = sorted(picked, key=lambda k: score_by_key[k], reverse=True)[:budget]
        elif len(picked) < budget:
            # MDS under-filled the budget (radii left slack unused) -- top
            # off with the highest-scoring seeds not already selected,
            # same as the plain top-K path would for the remaining slots.
            picked_set = set(picked)
            leftover = [k for k in score_by_key if k not in picked_set]
            leftover.sort(key=lambda k: score_by_key[k], reverse=True)
            picked = picked + leftover[: budget - len(picked)]

        return [seed_by_key[k] for k in picked]

    def auto_minimize_corpus(self):
        f = self.f
        # GA feeds f.corpus/f.seed_meta the same as the default path (see
        # save_to_corpus: the `if f.ga:` block runs alongside, not instead
        # of, the corpus.append) -- GALifecycle's own population is a
        # separate list of Individuals keyed on seed bytes/seed_key, not on
        # f.corpus indices or f.seed_meta, so pruning f.corpus never touches
        # it. The old blanket `if f.ga: return` predates that and just
        # silently disabled --minimize-every-execs for every GA run.
        #
        # QEA is different only in standalone mode: save_to_corpus() skips
        # the corpus.append() there (QEA's own population is the sole seed
        # source), so f.corpus stays frozen at the initial seed set and
        # there's nothing live to minimize. Under `--elo all` (or any
        # _use_elo path), QEA's bypass is lifted and f.corpus grows and is
        # read by the corpus-based seed strategies -- same condition
        # save_to_corpus() itself uses to decide whether to append.
        if f.qea and not getattr(f, "_use_elo", False):
            return
        if not f.corpus:
            return

        from fuzzer_tool.adapters.filesystem import hash_data

        seen: set[str] = set()
        unique: list[bytes] = []
        for seed in f.corpus:
            h = hash_data(seed)
            if h not in seen:
                seen.add(h)
                unique.append(seed)
        del seen  # free intermediate seed-hash set

        irreplaceable_seeds, fresh_seeds = self._set_aside(unique)
        stale_ratio = self._stale_ratio(unique)
        target_size = self._base_target(unique, stale_ratio)
        mandatory, target_size = self._cover_mandatory(unique, target_size)

        if self._over_budget(unique, target_size):
            unique = self._select_budget(unique, mandatory, target_size)

        # Save set-cover mandatory seeds to irreplaceable/ so they survive
        # future pruning cycles. Remove the original from seeds/ to avoid
        # duplicates on disk.
        if mandatory and f.corpus_dir:
            self._promote_mandatory(unique, mandatory)

        # Re-add fresh seeds that were set aside before minimization.
        # They haven't been fuzzed yet and need a chance to prove their value.
        if fresh_seeds:
            unique = fresh_seeds + unique

        # Re-add irreplaceable seeds that were set aside before minimization.
        # They are never pruned.
        if irreplaceable_seeds:
            unique = unique + irreplaceable_seeds

        self._prune_lineage(unique, fresh_seeds + irreplaceable_seeds, mandatory)

        self._recover_uncovered(unique, mandatory)

        # Bootstrap percolation post-pass: capture transitive redundancy that
        # single-pass greedy set-cover leaves behind. Disabled by default.
        if getattr(f, "_use_bootstrap", False) and len(unique) > 1:
            unique = self._bootstrap_pass(unique)

        removed = len(f.corpus) - len(unique)
        if removed > 0:
            self._commit_minimize(unique, removed, stale_ratio)

    def _over_budget(self, unique: list[bytes], target_size: int) -> bool:
        """True when *unique* exceeds the seed-count or byte budget."""
        f = self.f
        return len(unique) > target_size or (
            f.max_corpus_bytes > 0 and sum(len(s) for s in unique) > f.max_corpus_bytes
        )

    def _promote_mandatory(self, unique: list[bytes], mandatory: set[int]) -> None:
        """Promote every set-cover mandatory seed in *unique* to irreplaceable/."""
        for seed in unique:
            if id(seed) in mandatory:
                self._promote_seed(seed)

    def _bootstrap_pass(self, unique: list[bytes]) -> list[bytes]:
        """Bootstrap-percolation minimization of *unique* (transitive redundancy)."""
        f = self.f
        from fuzzer_tool.core.percolation import bootstrap_minimize_corpus

        unique, bootstrap_removed = bootstrap_minimize_corpus(
            unique, f._edge_tracker, k=getattr(f, "_bootstrap_k", 1)
        )
        if bootstrap_removed:
            log.info(
                "Bootstrap percolation removed %d seeds (transitive redundancy)",
                len(bootstrap_removed),
            )
        return unique

    def _set_aside(self, unique: list[bytes]) -> tuple[list[bytes], list[bytes]]:
        """Remove never-pruned seeds from *unique*: (irreplaceable, fresh)."""
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data

        # Irreplaceable seeds (loaded from corpus/seeds/irreplaceable/) are never pruned.
        # Separate them from the unique pool before minimization; re-add after.
        irreplaceable_seeds: list[bytes] = []
        if f.irreplaceable_hashes:
            for seed in unique[:]:  # iterate copy, mutate original
                if hash_data(seed) in f.irreplaceable_hashes:
                    irreplaceable_seeds.append(seed)
                    unique.remove(seed)

        # Fresh seeds (fuzz_count == 0) have never been picked by the seed picker.
        # Exclude them from minimization — we don't know their value yet.
        fresh_seeds: list[bytes] = []
        for seed in unique[:]:  # iterate copy, mutate original
            meta = f.seed_meta.get(seed)
            if meta and meta["fuzz_count"] == 0:
                fresh_seeds.append(seed)
                unique.remove(seed)
        return irreplaceable_seeds, fresh_seeds

    def _stale_ratio(self, unique: list[bytes]) -> float:
        """Fraction of seeds judged stale (heuristic, or Bayesian if higher)."""
        f = self.f
        stale_count = 0
        for seed in unique:
            meta = f.seed_meta.get(seed)
            if meta and meta["fuzz_count"] >= 50 and meta["coverage_edges"] == 0:
                stale_count += 1
        stale_ratio = stale_count / max(len(unique), 1)

        # Bayesian stale probability: P(stale | fuzz_count, 0_discoveries)
        # Uses Beta(1 + 0, 1 + fuzz_count) — the posterior probability that
        # a seed with `fuzz_count` attempts and 0 discoveries has discovery
        # probability below 0.01.
        #
        # Extreme-value asymptotics (docs/learnings/order-statistics-learnings.md):
        # n * min(U1..Un) → Exp(1) as n → ∞. If a seed's discovery probability
        # is the minimum of n independent tries, P(discovery < ε) ≈ 1 - exp(-n*ε).
        # For n = fuzz_count and ε = 0.01, this gives a simpler approximation:
        #   P(stale) ≈ 1 - exp(-fuzz_count * 0.01)
        # which matches the Beta CDF asymptotically and avoids the Beta integral.
        if not (f._use_bayesian and f._seed_quality):
            return stale_ratio

        bayesian_stale_count = 0
        for seed in unique:
            meta = f.seed_meta.get(seed)
            if not meta:
                continue
            fuzz = meta.get("fuzz_count", 0)
            if fuzz < 5:
                continue
            # P(discovery_prob < 0.01 | 0 discoveries in fuzz_count attempts)
            # = Beta.cdf(0.01, alpha=1, beta=1+fuzz_count)
            a, b = 1.0, 1.0 + fuzz
            # Mean of Beta = a/(a+b). If the posterior mean is below 0.01,
            # the seed is likely stale.
            if a / (a + b) < 0.01:
                bayesian_stale_count += 1
        bayesian_stale_ratio = bayesian_stale_count / max(len(unique), 1)
        # Use whichever stale ratio is higher (more conservative)
        return max(stale_ratio, bayesian_stale_ratio)

    def _base_target(self, unique: list[bytes], stale_ratio: float) -> int:
        """Corpus size budget: max_corpus or edge count, shrunk by staleness."""
        f = self.f
        if f.max_corpus > 0:
            target_size = f.max_corpus
        else:
            edges = 0
            if f.shm_cov:
                edges = f.shm_cov.cumulative_edges
            elif f.ptrace_cov:
                edges = f.ptrace_cov.cumulative_edges
            target_size = min(max(edges, 50), 5000)

        if stale_ratio > 0.3:
            if len(unique) > target_size:
                target_size = max(target_size, int(len(unique) * (1.0 - stale_ratio)))
            else:
                target_size = int(len(unique) * (1.0 - stale_ratio))
        return target_size

    def _cover_mandatory(self, unique: list[bytes], target_size: int) -> tuple[set[int], int]:
        """Greedy set-cover ids (by id(seed)) and the target size floored by them."""
        f = self.f
        # Greedy set-cover is O(n²) against the full seed list, so for large
        # corpora we first reduce the search space to one cheap candidate per
        # edge.  That bounds the inner loop by edge count rather than seed
        # count, while still preserving the actual minimal-cover result.
        et = f._edge_tracker
        all_edges = et.cumulative_edges if et and et.cumulative_edges else set()
        mandatory: set[int] = set()
        if all_edges and et.seed_edges:
            seed_edge_map: dict[int, set[int]] = {}
            for seed in unique:
                sk = self.seed_key(seed)
                s_edges = et.seed_edges.get(sk, set())
                if s_edges:
                    seed_edge_map[id(seed)] = s_edges
            if seed_edge_map:
                mandatory = self._greedy_cover(unique, seed_edge_map)
                target_size = max(target_size, len(mandatory))
        elif all_edges or f.corpus:
            productive = sum(
                1 for seed in unique if f.seed_meta.get(seed, {}).get("coverage_edges", 0) > 0
            )
            if productive > 0:
                target_size = max(target_size, productive)
        return mandatory, target_size

    def _greedy_cover(self, unique: list[bytes], seed_edge_map: dict[int, set[int]]) -> set[int]:
        """Set cover over *seed_edge_map* (core/set_cover.py); returns chosen seed ids."""
        # Terminate against what these seeds can actually cover, not
        # against cumulative_edges. EdgeTracker._prune_tracked_seeds drops
        # entries from seed_edges once past max_tracked_seeds (200,000
        # since fe8fd42, up from 200 -- so in practice it no longer
        # fires at all; see docs/TODO.md) but
        # never removes their edges from cumulative_edges, so on any run
        # past 200 seeds all_edges is a strict superset of anything the
        # loop can reach. `covered != all_edges` was therefore permanently
        # true: the loop never converged, always ran to best_gain == 0, and
        # selected every seed holding a unique edge — making `mandatory`,
        # and the target_size floor derived from it, meaningless.
        candidate_ids: set[int] = set(seed_edge_map.keys())
        if len(candidate_ids) > 5000:
            candidate_ids = self._cheap_candidates(unique, seed_edge_map)
        sizes = {id(seed): len(seed) for seed in unique}
        pool = {sid: seed_edge_map[sid] for sid in candidate_ids if sid in seed_edge_map}
        return set(min_cover(pool, sizes))

    def _cheap_candidates(
        self, unique: list[bytes], seed_edge_map: dict[int, set[int]]
    ) -> set[int]:
        """One lowest-cost (exec_us * size) seed id per edge, to bound set-cover."""
        f = self.f
        mean_us = f.mean_exec_time() * 1_000_000
        edge_to_seeds: dict[int, list[tuple[float, int]]] = {}
        for seed in unique:
            sid = id(seed)
            if sid not in seed_edge_map:
                continue
            meta = f.seed_meta.get(seed, {})
            exec_us = seed_exec_us(meta, mean_us)
            input_size = max(1, meta.get("input_size", 1))
            cost = exec_us * input_size
            for edge in seed_edge_map[sid]:
                edge_to_seeds.setdefault(edge, []).append((cost, sid))
        return {min(candidates)[1] for candidates in edge_to_seeds.values()}

    def _select_budget(
        self, unique: list[bytes], mandatory: set[int], target_size: int
    ) -> list[bytes]:
        """Keep mandatory seeds plus the best-scored optional ones within budget."""
        f = self.f
        # Split into mandatory (set-cover essential) and optional.
        mandatory_seeds = [s for s in unique if id(s) in mandatory]
        optional = [s for s in unique if id(s) not in mandatory]
        scored = [(self._minimize_score(seed), seed) for seed in optional]
        scored.sort(key=lambda x: x[0], reverse=True)
        if f.max_corpus_bytes > 0:
            # Knapsack: sort optional seeds by value/weight density
            # (value = coverage score, weight = seed byte size).
            # Greedy density-ordering is a well-known 2-approximation
            # for 0/1 knapsack.
            scored.sort(
                key=lambda x: x[0] / max(len(x[1]), 1),
                reverse=True,
            )
            selected = []
            total_bytes = sum(len(s) for s in mandatory_seeds)
            for _score, seed in scored:
                seed_bytes = len(seed)
                if total_bytes + seed_bytes <= f.max_corpus_bytes:
                    selected.append(seed)
                    total_bytes += seed_bytes
            return mandatory_seeds + selected
        if getattr(f, "_use_mds_select", False) and f._edge_tracker is not None:
            return mandatory_seeds + self._mds_select_optional(
                scored, target_size, len(mandatory_seeds)
            )
        if getattr(f, "_use_minimax_select", False):
            return mandatory_seeds + self._minimax_select_optional(
                scored, target_size - len(mandatory_seeds), mandatory_seeds
            )
        # Count-budget: keep top-K by score (original behavior)
        budget = target_size - len(mandatory_seeds)
        keep = min(budget, len(scored))
        return mandatory_seeds + [s for _, s in scored[:keep]]

    def _minimax_select_optional(
        self, scored: list[tuple[float, bytes]], budget: int, kept: list[bytes]
    ) -> list[bytes]:
        """Robust backups first, then top-K by score for the slots left."""
        ranked = [s for _, s in scored]
        picked = self.minimax_robust_admission(ranked, budget, kept)

        # Robustness stops once no seed lowers the worst single-seed loss;
        # the rest of the budget goes to the plain top-K order.
        taken = {id(s) for s in picked}
        rest = [s for s in ranked if id(s) not in taken]
        return picked + rest[: max(0, budget - len(picked))]

    def _minimize_score(self, seed: bytes) -> float:
        """Keep-score of an optional seed: edge score x Wasserstein x PPMD x QP."""
        f = self.f
        seed_key = self.seed_key(seed)
        meta = f.seed_meta.get(seed)
        fuzz = meta["fuzz_count"] if meta else 0
        discovered = meta["coverage_edges"] if meta else 0

        edge_score = self._edge_score(seed_key, fuzz, discovered)
        wasserstein_weight = f._edge_tracker.compute_wasserstein_weight(seed_key)

        # PPMD novelty: incompressible seeds are more diverse
        ppmd_bonus = 1.0
        if getattr(f, "_ppmd", None) and f._ppmd.enabled:
            ppmd_bonus = 1.0 + f._ppmd.compute_seed_novelty(seed) * 0.5

        # Quasiperiodicity novelty: seeds with no short internal
        # cover are structurally more diverse (see
        # core/quasiperiodicity.py) -- same shape as the PPMD bonus,
        # a different and independent redundancy signal.
        qp_bonus = 1.0
        if getattr(f, "_qp", None) and f._qp.enabled:
            qp_bonus = 1.0 + f._qp.compute_seed_novelty(seed) * 0.5

        return edge_score * wasserstein_weight * ppmd_bonus * qp_bonus

    def _edge_score(self, seed_key: str, fuzz: int, discovered: int) -> float:
        """Coverage score; Bayesian posterior mean when the seed has one."""
        f = self.f
        # Bayesian seed score: use the posterior mean from
        # BayesianSeedQuality when available.
        if f._use_bayesian and f._seed_quality and seed_key in f._seed_quality._alpha:
            mean = f._seed_quality.posterior_mean(seed_key)
            # Scale posterior mean to a useful range for scoring:
            # posterior mean ~ [0,1]. Multiply by discovered * 10
            # to get a score on a comparable scale to the heuristic.
            return mean * 10.0 + (discovered * 5.0 if discovered > 0 else 0.0)
        edge_score = discovered * 10
        if fuzz > 0 and discovered == 0:
            edge_score *= max(0.01, 1.0 / (1.0 + fuzz * 0.01))
        else:
            edge_score += 1.0 / max(fuzz, 1)
        return edge_score

    def _promote_seed(self, seed: bytes) -> None:
        """Move *seed* to irreplaceable/ and drop its seeds/ copy (no-op if already there)."""
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data

        h = hash_data(seed)
        if h in f.irreplaceable_hashes:
            return
        save_irreplaceable(
            seed,
            f.corpus_dir,
            f.seen_hashes,
            f.irreplaceable_hashes,
            f.bloom,
        )
        # Remove the original from seeds/ to avoid duplicate
        seeds_sub = f.corpus_dir / "seeds" / h[:2] / f"id_{h}"
        if seeds_sub.exists():
            seeds_sub.unlink()

    def _prune_lineage(self, unique: list[bytes], kept: list[bytes], mandatory: set[int]) -> None:
        """Drop unproductive lineage subtrees from *unique*; reset the credit clock.

        A dropped seed whose subtree contributed < 1.0 structural edge-weight
        and gained no coverage since the last minimize is an unproductive
        branch — drop the whole subtree instead of just the low-scoring seed.
        Mandatory/fresh/irreplaceable (*kept*) seeds are protected. No-op
        without --lineage.
        """
        f = self.f
        if not (f._use_lineage and getattr(f, "_lineage", None) is not None):
            return

        key_to_seed = {self.seed_key(s): s for s in f.corpus}
        kept_keys = {self.seed_key(s) for s in unique}
        protected = {id(s) for s in kept}
        if mandatory:
            protected |= {id(s) for s in unique if id(s) in mandatory}

        _coverage_fn = self._coverage_reader(key_to_seed)

        # subtree_weight is a volume: a wide branch of one-edge children
        # clears the < 1.0 gate that a narrow branch of high-yield
        # children fails, so pruning was biased toward keeping the
        # spray. pagerank_credit divides each child's contribution by
        # its sibling count, which ranks branches by yield per mutation;
        # requiring both keeps a branch alive if either measure rates it.
        credit = f._lineage.pagerank_credit()
        credit_floor = self._credit_floor(credit)

        subtree_drops: set[str] = set()
        for seed in f.corpus:
            sk = self.seed_key(seed)
            if sk in kept_keys or sk in subtree_drops:
                continue
            if (
                f._lineage.recent_credit(sk, _coverage_fn) == 0.0
                and f._lineage.subtree_weight(sk) < 1.0
                and credit.get(sk, 0.0) < credit_floor
            ):
                self._drop_subtree(sk, unique, key_to_seed, protected, subtree_drops)
        self._reset_credit_clock()

    @staticmethod
    def _credit_floor(credit: dict[str, float]) -> float:
        """1/n over nodes holding positive PageRank credit; 0.0 when none do.

        A share below 1/n of the distributed credit is below what an average
        productive node holds.
        """
        n_credited = sum(1 for v in credit.values() if v > 0.0)
        return (1.0 / n_credited) if n_credited else 0.0

    def _reset_credit_clock(self) -> None:
        """Record current coverage per seed so the next minimize measures the delta."""
        f = self.f
        for seed in f.corpus:
            meta = f.seed_meta.get(seed)
            if meta is not None:
                meta["coverage_edges_baseline"] = meta.get("coverage_edges", 0)

    def _coverage_reader(self, key_to_seed: dict):
        """Seed key -> (coverage_edges, coverage_edges_baseline); (0, 0) if unknown."""
        f = self.f

        def _coverage_fn(k: str) -> tuple[int, int]:
            seed = key_to_seed.get(k)
            if seed is None:
                return (0, 0)
            meta = f.seed_meta.get(seed, {})
            return (
                meta.get("coverage_edges", 0),
                meta.get("coverage_edges_baseline", 0),
            )

        return _coverage_fn

    def _drop_subtree(
        self,
        sk: str,
        unique: list[bytes],
        key_to_seed: dict,
        protected: set[int],
        subtree_drops: set[str],
    ) -> None:
        """Remove every unprotected seed of *sk*'s lineage subtree from *unique*."""
        for k in self.f._lineage.subtree_keys(sk):
            s = key_to_seed.get(k)
            if s is not None and id(s) not in protected and s in unique:
                unique.remove(s)
            subtree_drops.add(k)

    def _recover_uncovered(self, unique: list[bytes], mandatory: set[int]) -> None:
        """Re-add seeds whose unique edges were dropped by scoring or lineage pruning."""
        f = self.f
        et = f._edge_tracker
        if not (et and et.cumulative_edges and unique):
            return

        kept_coverage: set[int] = set()
        for seed in unique:
            sk = self.seed_key(seed)
            kept_coverage.update(et.seed_edges.get(sk, set()))
        uncovered = et.cumulative_edges - kept_coverage
        if not uncovered:
            return

        recovered_count = 0
        for seed in f.corpus:
            if seed in unique:
                continue
            sk = self.seed_key(seed)
            seed_edges = et.seed_edges.get(sk, set())
            if seed_edges & uncovered:
                unique.append(seed)
                mandatory.add(id(seed))
                if f.corpus_dir:
                    self._promote_seed(seed)
                # Mark seed as recovered so mutations are not skipped from it
                if getattr(f, "cuckoo_seed_filter", None) is not None:
                    h = self.seed_key(seed)
                    f._cuckoo_recovered.add(h)
                recovered_count += 1
        if recovered_count:
            log.warning(
                "Recovered %d seeds to cover %d uncovered edges after minimization",
                recovered_count,
                len(uncovered),
            )

    def _commit_minimize(self, unique: list[bytes], removed: int, stale_ratio: float) -> None:
        """Move pruned files to pruned/ and swap f.corpus/f.seed_meta to *unique*."""
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data as _hash

        kept_set = {_hash(s) for s in unique}
        self._prune_files(kept_set)
        del kept_set  # free kept hashes after file pruning

        # Add pruned seeds to the cuckoo seed filter if enabled
        if getattr(f, "cuckoo_seed_filter", None) is not None:
            for seed in f.corpus:
                if seed not in unique:
                    h = self.seed_key(seed)
                    f.cuckoo_seed_filter.add(h)
                    # Discard recovered exemption so re-pruned seeds are filtered again
                    f._cuckoo_recovered.discard(h)

        f.corpus = unique
        self.rebuild_entropy()
        new_meta = {}
        for seed in unique:
            if seed in f.seed_meta:
                new_meta[seed] = f.seed_meta[seed]
        f.seed_meta = new_meta
        f._agg_cache_valid = False
        f._weight_cache = None
        f._cached_weights = {}
        f._overlap_density_cache = {}
        f._last_minimize_exec = f.exec_count
        f._pruned_count += removed
        f._corpus_flux.record_eviction(removed)
        log.info(
            "Auto-minimized corpus: %d -> %d seeds -> pruned/ (stale_ratio=%.1f)",
            len(f.corpus) + removed,
            len(f.corpus),
            stale_ratio,
        )

    def _prune_files(self, kept_set: set[str]) -> None:
        """Move seed and delta files whose hash is not in *kept_set* under pruned/."""
        f = self.f
        seeds_dir = f.corpus_dir / "seeds"
        deltas_dir = f.corpus_dir / "deltas"
        pruned_dir = seeds_dir / "pruned"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        # Prune full seeds — seeds are stored in two-digit hash
        # subdirectories (seeds/ab/id_abc...), so walk recursively.
        # Skip the irreplaceable/ subdirectory — those seeds are never pruned.
        for fh in seeds_dir.rglob("id_*"):
            if not fh.is_file():
                continue
            # Skip files under seeds/irreplaceable/ (never pruned)
            if "irreplaceable" in fh.parts:
                continue
            h = fh.name[3:]
            if h not in kept_set:
                sub = pruned_dir / h[:2]
                sub.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fh), str(sub / fh.name))
        # Zip-resident seeds: a member cannot move, so tombstone it.
        store = seed_zip.lookup(f.corpus_dir)
        for h in store.main_hashes() - kept_set if store is not None else ():
            store.retire(h)
        # Prune delta files
        if not deltas_dir.exists():
            return
        deltas_pruned_dir = deltas_dir / "pruned"
        for fh in deltas_dir.iterdir():
            if not fh.is_file():
                continue
            if not (fh.suffix == ".json" and fh.name.startswith("delta_")):
                continue
            h = fh.name[6:-5]
            if h not in kept_set:
                sub = deltas_pruned_dir / h[:2]
                sub.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fh), str(sub / fh.name))

    def minimax_robust_admission(
        self,
        candidate_seeds: list[bytes],
        budget: int | None = None,
        kept: list[bytes] | None = None,
    ) -> list[bytes]:
        """Pick up to *budget* candidates that minimize the max single-seed loss.

        *kept* seeds (set-cover mandatory) count toward robustness but are
        never returned: a candidate is worth a slot when it backs up the
        kept seed whose removal would lose the most edges.

        Args:
            candidate_seeds: Optional seeds to consider.
            budget: Maximum number of candidates to admit (default: all).
            kept: Seeds already kept (default: none).

        Returns:
            Admitted candidates, in selection order.
        """
        budget = len(candidate_seeds) if budget is None else budget
        kept = kept or []
        et = self.f._edge_tracker
        if budget <= 0 or not candidate_seeds or et is None:
            return []

        seed_edges: dict[str, set[int]] = {}
        key_to_seed: dict[str, bytes] = {}
        for seed in candidate_seeds + kept:
            sk = self.seed_key(seed)
            edges = et.seed_edges.get(sk)
            if edges:
                seed_edges[sk] = edges
                key_to_seed[sk] = seed

        kept_keys = [self.seed_key(s) for s in kept]
        rd = RateDistortionCorpus()
        picked = rd.minimax_robust_corpus_admission(seed_edges, budget, kept_keys)
        return [key_to_seed[k] for k in picked]

    def deprioritize_near_duplicates(self):
        f = self.f
        # Short-circuit: if Poisson-disk admission is active and hasn't seen
        # enough redundant admissions, skip the expensive O(N) scan entirely.
        # This turns the periodic scan into an amortized O(1) check.
        if getattr(f, "_use_poisson_disk_admission", False):
            if f._redundant_admission_count < 50:
                return
            f._redundant_admission_count = 0  # reset after scan

        if len(f.corpus) < 10:
            return

        near_dupes = f._edge_tracker.find_near_duplicate_seeds(max_hamming=0.05)
        if not near_dupes:
            return

        to_remove: set[bytes] = set()
        for key_a, key_b, _hdist in near_dupes:
            seed_a, seed_b = self._find_pair(key_a, key_b)
            if not seed_a or not seed_b:
                continue
            if seed_a in to_remove or seed_b in to_remove:
                continue

            meta_a = f.seed_meta.get(seed_a, {})
            meta_b = f.seed_meta.get(seed_b, {})
            edges_a = meta_a.get("coverage_edges", 0)
            edges_b = meta_b.get("coverage_edges", 0)

            if edges_a <= edges_b:
                to_remove.add(seed_a)
            else:
                to_remove.add(seed_b)

        if to_remove:
            self._evict_dupes(to_remove)

    def _find_pair(self, key_a: str, key_b: str) -> tuple[bytes | None, bytes | None]:
        """Corpus seeds whose keys are *key_a* / *key_b* (None when absent)."""
        seed_a = None
        seed_b = None
        for s in self.f.corpus:
            if self.seed_key(s) == key_a:
                seed_a = s
            elif self.seed_key(s) == key_b:
                seed_b = s
            if seed_a and seed_b:
                break
        return seed_a, seed_b

    def _evict_dupes(self, to_remove: set[bytes]) -> None:
        """Drop *to_remove* from corpus/meta/entropy and invalidate weight caches."""
        f = self.f
        f.corpus = [s for s in f.corpus if s not in to_remove]
        for s in to_remove:
            f.seed_meta.pop(s, None)
            self._entropy_remove(s)
        f._agg_cache_valid = False
        f._weight_cache = None
        f._cached_weights = {}
        f._overlap_density_cache = {}
        f._corpus_flux.record_eviction(len(to_remove))
        log.info(
            "Deprioritized %d near-duplicate seeds (Hamming <= 0.05 on edge bitmaps)",
            len(to_remove),
        )
