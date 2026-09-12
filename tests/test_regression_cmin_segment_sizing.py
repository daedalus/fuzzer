"""cmin sized its coverage segment in bytes and told the child entries.

`_minimize_with_coverage` used a single `edge_map_size = 65536` for three
incompatible purposes:

  * AFL_MAP_SIZE handed to the child -- which afl_shim.c reads as a count of
    ENTRIES, not bytes
  * the byte size passed to shmget()
  * the byte count read back with string_at()

65,536 entries needs SHM_METADATA_SIZE + 65_536 * 8 = 524,320 bytes, so the
shim computed its table pointer inside a 64 KiB segment and wrote 448 KiB
past the end of it. Every replayed child died on SIGSEGV, every edge set came
back empty, and the blackout guard then refused to prune -- which is why this
failed safe instead of deleting corpora: cmin simply never worked against an
instrumented target.

The read was wrong independently of the size. It treated the segment as AFL's
byte-per-edge bitmap and took nonzero byte INDICES as edge ids, but this shim
writes {edge_id, count} entries behind a header, so those "edges" were byte
offsets into a hash table plus the header bytes.

Both are fixed by asking ShmCoverage, which owns the layout, for edge ids.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, SIZEOF_ENTRY, ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

# Calls __sanitizer_cov_trace_pc_guard directly rather than relying on
# -fsanitize-coverage=trace-pc-guard, which gcc does not have -- the same
# convention as tests/test_ctx_and_map_size.py. Distinct first bytes fire
# distinct guard sets, so set-cover has something real to do.
_TARGET = """
#include <stdio.h>
#include <stdlib.h>
static void fire(uint32_t g) { __sanitizer_cov_trace_pc_guard(&g); }
int main(void) {
    int ch = getchar();
    fire(1);                                  /* entry, common to all */
    if (ch == 'a')      { fire(10); fire(11); }
    else if (ch == 'b') { fire(20); fire(21); }
    else if (ch == 'c') { fire(10); fire(11); fire(30); }
    return 0;
}
"""

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


def _segment_bytes(shm_id: int) -> int | None:
    """Actual allocated size of a SysV segment, from shmctl(IPC_STAT)."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    class _IpcPerm(ctypes.Structure):
        _fields_ = [
            ("key", ctypes.c_int),
            ("uid", ctypes.c_uint),
            ("gid", ctypes.c_uint),
            ("cuid", ctypes.c_uint),
            ("cgid", ctypes.c_uint),
            ("mode", ctypes.c_ushort),
            ("_pad1", ctypes.c_ushort),
            ("seq", ctypes.c_ushort),
            ("_pad2", ctypes.c_ushort),
            ("_glibc1", ctypes.c_ulong),
            ("_glibc2", ctypes.c_ulong),
        ]

    class _ShmidDs(ctypes.Structure):
        _fields_ = [
            ("shm_perm", _IpcPerm),
            ("shm_segsz", ctypes.c_size_t),
            ("shm_atime", ctypes.c_long),
            ("shm_dtime", ctypes.c_long),
            ("shm_ctime", ctypes.c_long),
            ("shm_cpid", ctypes.c_int),
            ("shm_lpid", ctypes.c_int),
            ("shm_nattch", ctypes.c_ulong),
            ("_unused4", ctypes.c_ulong),
            ("_unused5", ctypes.c_ulong),
        ]

    buf = _ShmidDs()
    IPC_STAT = 2
    if libc.shmctl(shm_id, IPC_STAT, ctypes.byref(buf)) != 0:
        return None
    return int(buf.shm_segsz)


@pytest.fixture(scope="module")
def branching_target(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("cmin")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    r = subprocess.run(
        ["gcc", "-O0", "-g", "-D__AFL_CTX_SENSITIVE=0", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return str(exe)


class TestSegmentIsSizedInBytesForTheEntriesAdvertised:
    def test_a_byte_count_would_undersize_the_segment(self):
        """The arithmetic that made the old default wrong, stated once."""
        entries = 65536
        needed = SHM_METADATA_SIZE + entries * SIZEOF_ENTRY
        assert needed > entries, (
            "if entry count and byte size were interchangeable there would be "
            "no bug to regress against"
        )
        assert needed == 524_320

    @needs_cc
    def test_child_survives_and_records_edges(self, branching_target):
        """The end-to-end symptom: children died, so every edge set was empty."""
        shm = ShmCoverage()
        try:
            env = {
                **os.environ,
                "__AFL_SHM_ID": shm.env_id,
                "AFL_MAP_SIZE": str(shm.num_entries),
            }
            p = subprocess.run(
                [branching_target], input=b"a", env=env, capture_output=True, timeout=30
            )
            assert p.returncode == 0, (
                f"child exited {p.returncode} (-11 is SIGSEGV, the undersized "
                "segment); no coverage can be collected from a dead child"
            )
            assert shm.get_edge_ids(), "child ran but recorded no edges"
        finally:
            shm.cleanup()

    def test_the_env_cmin_hands_the_child_satisfies_the_size_contract(self, monkeypatch, tmp_path):
        """The invariant, captured from the real call path.

        Asserting "the child survived and recorded edges" is NOT enough, and
        this test exists because the first version of it was: a toy target
        firing four guards segfaults on whichever write first lands out of
        bounds, having already recorded three edges -- plenty for set-cover to
        prune, so the test passed with the bug reintroduced. The severity of
        an undersized segment scales with how many edges the target fires, so
        a small target hides it almost completely.

        What cannot be hidden is the arithmetic: whatever entry count cmin
        advertises in AFL_MAP_SIZE, the segment it allocates must be big
        enough for that many entries plus the front region. Captured here
        from the env _minimize_with_coverage actually builds.
        """
        from fuzzer_tool.services import minimize

        captured = {}

        def fake_run_stdin(target, data, timeout, env=None):
            # Stat the segment HERE: _minimize_with_coverage cleans it up
            # before returning, so it does not exist by the time the call
            # completes. The env the child would have been given and the
            # segment it would have attached to are both only live now.
            captured.update(env or {})
            captured["_seg_bytes"] = _segment_bytes(int((env or {})["__AFL_SHM_ID"]))
            return None

        monkeypatch.setattr(minimize, "run_target_stdin", fake_run_stdin, raising=False)
        monkeypatch.setattr(
            "fuzzer_tool.adapters.process.run_target_stdin", fake_run_stdin, raising=False
        )

        corpus = tmp_path / "c"
        corpus.mkdir()
        f = corpus / "id_0"
        f.write_bytes(b"a")

        minimize._minimize_with_coverage(
            corpus_files=[f],
            target="/bin/true",
            target_args=[],
            timeout=5,
            file_mode=False,
            output_dir=str(tmp_path / "o"),
            corpus_path=corpus,
        )

        assert "AFL_MAP_SIZE" in captured, "cmin did not advertise a map size"
        entries = int(captured["AFL_MAP_SIZE"])
        shm_id = int(captured["__AFL_SHM_ID"])
        needed = SHM_METADATA_SIZE + entries * SIZEOF_ENTRY

        seg_bytes = captured.get("_seg_bytes")
        assert seg_bytes is not None, "could not stat the segment cmin created"
        assert shm_id > 0
        assert needed <= seg_bytes, (
            f"cmin advertised {entries:,} entries (needing {needed:,} bytes) into a "
            f"{seg_bytes:,}-byte segment — the shim computes its table pointer from "
            f"AFL_MAP_SIZE alone and would write {needed - seg_bytes:,} bytes past "
            "the end"
        )


class TestDistinctInputsYieldDistinctEdgeSets:
    @needs_cc
    def test_edge_sets_differ_by_branch(self, branching_target):
        """Set-cover needs edge sets that actually distinguish inputs.

        The old reader returned nonzero byte offsets into the entry table, so
        two inputs hitting different edges could still land in overlapping
        byte ranges. Edge ids distinguish them by construction.
        """
        shm = ShmCoverage()
        seen = {}
        try:
            env = {
                **os.environ,
                "__AFL_SHM_ID": shm.env_id,
                "AFL_MAP_SIZE": str(shm.num_entries),
            }
            for ch in (b"a", b"b", b"c"):
                shm.reset_edge_map()
                subprocess.run(
                    [branching_target], input=ch, env=env, capture_output=True, timeout=30
                )
                seen[ch] = shm.get_edge_ids()
        finally:
            shm.cleanup()
        assert all(seen.values()), f"an input recorded nothing: { {k: len(v) for k, v in seen.items()} }"
        assert seen[b"a"] != seen[b"b"], "different branches produced identical edge sets"

    @needs_cc
    def test_cmin_keeps_a_covering_subset_and_prunes_the_rest(self, branching_target, tmp_path):
        """The whole point: a duplicate-heavy corpus actually gets pruned.

        Pre-fix this returned (n, 0) via the blackout guard on every corpus,
        because every child had segfaulted.
        """
        from fuzzer_tool.services.minimize import _minimize_with_coverage

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        files = []
        for i, ch in enumerate([b"a", b"b", b"c", b"a", b"a", b"b"]):
            f = corpus / f"id_{i}"
            f.write_bytes(ch)
            files.append(f)

        out = tmp_path / "out"
        # _commit_results returns (kept, removed).
        kept, removed = _minimize_with_coverage(
            corpus_files=files,
            target=branching_target,
            target_args=[],
            timeout=10,
            file_mode=False,
            output_dir=str(out),
            corpus_path=corpus,
        )
        assert kept > 0, "blackout guard fired — the children are not recording coverage"
        assert removed > 0, (
            f"nothing pruned from a duplicate-heavy corpus (kept {kept}, removed "
            f"{removed} of {len(files)})"
        )
        assert kept + removed == len(files)
        # Three distinct behaviours among six files, so a covering subset is
        # smaller than the corpus but must not collapse to one file.
        assert 1 < kept <= 3, f"set-cover kept {kept} of {len(files)}"
