"""__afl_map_shm() zeroed the generation tag the fuzzer had just written.

Publishing the context width masked off bits 0..7 AND bits 24..31 before
writing ctx, so a child that attaches after the parent's reset_edge_map()
stepped the generation back to 0. Every execution:

    diag before child = 0x01000000 (gen=1)
    diag after  child = 0x00000000 (gen=0)

The parent then read generation 0 too, and the entries the child stamped were
also tagged 0, so the two sides agreed -- consistently, on a value that never
advances. Nothing aged out. `get_edge_ids()` returned the cumulative union of
the whole run instead of the edges of the execution just performed:

    exec 1: [1]    exec 2: [1, 3]    exec 3: [1, 3]    exec 4: [1, 3, 5]

Scope is exactly the paths where the child attaches per execution, which is
the DEFAULT subprocess path (`run_target_fast`). The forkserver runs the
constructor once before forking and the in-process modes run it once in the
fuzzer itself, so both were unaffected -- which is why the measurement in
services/runner.py that established generation tagging (taken on a
shim-linked .so, in-process) could not see this.

Everything downstream of the live edge set is affected: per-execution
attribution, `has_new_coverage` and therefore corpus admission, the bandit
schedulers' edges-discovered reward, stability calibration, and the trim's
subset test.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

# Fires exactly one guard, chosen on the command line, so each execution has
# a known and distinct edge set.
_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    uint32_t g = (uint32_t)atoi(argv[1]);
    __sanitizer_cov_trace_pc_guard(&g);
    return 0;
}
"""

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def one_edge_target(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("gen")
    src = d / "one.c"
    src.write_text(_DRIVER)
    exe = d / "one"
    r = subprocess.run(
        ["gcc", "-O1", "-g", "-D__AFL_CTX_SENSITIVE=0", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return str(exe)


@pytest.fixture(scope="module")
def one_edge_so(tmp_path_factory):
    """The same shim as a loadable object, for the one test that must call
    __afl_map_reset() -- its only caller anywhere is in-process."""
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("gen_so")
    src = d / "lib.c"
    src.write_text("void touch(void) { }\n")
    so = d / "libone.so"
    r = subprocess.run(
        [
            "gcc", "-O1", "-g", "-shared", "-fPIC", "-D__AFL_CTX_SENSITIVE=0",
            "-include", SHIM, "-o", str(so), str(src),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build as .so: {r.stderr[:300]}")
    return str(so)


class TestAttachPreservesGeneration:
    @needs_cc
    def test_child_attach_leaves_the_fuzzers_tag_alone(self, one_edge_target):
        """The narrow assertion: the word survives the child unchanged."""
        cov = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": cov.env_id, "AFL_MAP_SIZE": "1024"}
            for _ in range(3):
                cov.reset_edge_map()
                before = cov.read_generation()
                subprocess.run([one_edge_target, "7"], env=env, capture_output=True)
                assert cov.read_generation() == before, (
                    "the child's attach rewrote the generation tag: "
                    f"{before} -> {cov.read_generation()}"
                )
        finally:
            cov.cleanup()

    @needs_cc
    def test_generation_actually_advances_across_executions(self, one_edge_target):
        cov = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": cov.env_id, "AFL_MAP_SIZE": "1024"}
            seen = []
            for _ in range(5):
                cov.reset_edge_map()
                subprocess.run([one_edge_target, "7"], env=env, capture_output=True)
                seen.append(cov.read_generation())
            assert seen == sorted(set(seen)) and len(set(seen)) == 5, (
                f"generation did not advance across executions: {seen}"
            )
        finally:
            cov.cleanup()

    @needs_cc
    def test_live_set_is_this_execution_not_the_union(self, one_edge_target):
        """The consequence that matters, and the one the old code inverted.

        Each execution fires a single guard, so the live set must have
        exactly one edge in it every time. Pre-fix this grew: 1, 2, 2, 3, 3.
        """
        cov = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": cov.env_id, "AFL_MAP_SIZE": "1024"}
            sizes = []
            for guard in range(1, 6):
                cov.reset_edge_map()
                subprocess.run([one_edge_target, str(guard)], env=env, capture_output=True)
                sizes.append(len(cov.get_edge_ids()))
            assert sizes == [1] * 5, (
                f"live edge set accumulated across executions instead of "
                f"reporting each one: {sizes}"
            )
        finally:
            cov.cleanup()


class TestShimResetUsesTheSharedTag:
    @needs_cc
    def test_map_reset_advances_the_word_not_a_private_static(self, one_edge_so):
        """__afl_map_reset() kept its own counter and wrote it to the word.

        That was only correct while attach zeroed the word to match a
        freshly-zeroed static. With attach preserving the fuzzer's tag, a
        static at 0 against a word at 1 writes 1 back -- no advance at all,
        so the previous execution's entries stay readable as live.

        Driven here through the in-process path, because that is the only
        caller __afl_map_reset has.
        """
        import ctypes

        cov = ShmCoverage(size=1024)
        try:
            cov.reset_edge_map()  # word now at generation 1, before any attach
            start = cov.read_generation()
            env_shm, env_size = os.environ.get("__AFL_SHM_ID"), os.environ.get("AFL_MAP_SIZE")
            os.environ["__AFL_SHM_ID"] = cov.env_id
            os.environ["AFL_MAP_SIZE"] = "1024"
            try:
                lib = ctypes.CDLL(one_edge_so)
            except OSError as e:
                pytest.skip(f"shim .so is not loadable: {e}")
            finally:
                if env_shm is None:
                    os.environ.pop("__AFL_SHM_ID", None)
                else:
                    os.environ["__AFL_SHM_ID"] = env_shm
                if env_size is None:
                    os.environ.pop("AFL_MAP_SIZE", None)
                else:
                    os.environ["AFL_MAP_SIZE"] = env_size
            reset = getattr(lib, "__afl_map_reset")
            reset.restype = None
            reset()
            assert cov.read_generation() == (start + 1) & 0xFF, (
                f"__afl_map_reset did not advance the shared tag: "
                f"{start} -> {cov.read_generation()}"
            )
        finally:
            cov.cleanup()
