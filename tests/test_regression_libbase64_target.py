"""libbase64 differential target: source bounds, build wiring, and oracle.

``targets/libbase64_read.c`` runs one input through every SIMD codec this CPU
supports and aborts on the first disagreement with the plain codec.

Two things are easy to get silently wrong:

* the build wiring. libbase64's ``lib/config.h`` is *generated* (upstream's
  Makefile does it), each SIMD codec needs its own ``-m`` flag, and
  ``codec_choose.c`` references only the codecs that config.h enables. So the
  objects are compiled in their own pass, without the shim, against one
  generated config.h per build suffix (lz4/secp256k1 discipline, Hard Rule 9).
* the oracle. A differential that cannot fire on a broken codec is not a
  differential. ``TestOracle`` builds the target against a clean tree
  (control), then against a tree with one byte of the AVX2 encoder's
  translate LUT changed, and requires SIGABRT. The control must pass first
  (Hard Rule 46).

The behaviour tests need clang and the vendored tree
(``tools/vendor_libbase64.sh``); they skip otherwise.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path

import pytest

from tests.support.scripted_rng import ScriptedRng

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "libbase64_read.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_libbase64.sh"
VENDOR_ROOT = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
VENDOR_TREE = VENDOR_ROOT / "libbase64"

SIGABRT_RC = -6

# Inputs the wrapper must survive without aborting: empty, mode byte only,
# stream modes missing their chunk seed, and a chunk seed that overflows the
# remaining payload (b64_next_chunk must clamp to `left`).
EDGE_INPUTS = {
    "empty": b"",
    "mode_decode": b"\x00",
    "mode_encode": b"\x01",
    "stream_decode_no_seed": b"\x02",
    "stream_encode_no_seed": b"\x03",
    "stream_huge_seed": b"\x02\xff" + b"QUJD" * 4,
}

# One zero byte per 6-bit group lands in the [0..25] range of the AVX2
# encode LUT, so shifting its first entry (65 -> 66) corrupts every output
# byte of a zero payload. Any input with a zero first byte triggers it.
FAULT_INPUT = b"\x01" + b"\x00" * 64
LUT_HEAD = "\t\t65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,"

# Generic dec_head.c (included by every codec) loads the saved stream state
# here. Corrupting the carry when a call resumes one char into a quantum
# breaks only chunked decodes split at offset 1 mod 4.
STATE_CARRY = "st.carry = state->carry;"
MID_QUANTUM_FAULT = "\nif (st.bytes == 1) { st.carry ^= 0x04; }"

# lib.c dispatches every stream decode here; the fault reads one byte past
# the input, the classic SIMD tail over-read.
STREAM_DEC_RETURN = "return codec.dec(state, src, srclen, out, outlen);"
OVERREAD_FAULT = "{ volatile char c = srclen ? src[srclen] : 0; (void)c; }\n\t"
OVERREAD_INPUT = b"\x00QUJD"


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def compile_body(build_src) -> str:
    m = re.search(r"compile_libbase64_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
    assert m, "compile_libbase64_objects not found in tools/build_targets.sh"
    return m.group(1)


class TestSource:
    def test_target_exists(self):
        assert TARGET.is_file(), "targets/libbase64_read.c is missing"

    def test_entry_points_exported(self):
        src = TARGET.read_text()
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in src
        assert "int fuzz_libbase64(const unsigned char *buf, size_t size)" in src

    def test_every_codec_is_pitted_against_plain(self):
        """A codec missing from the table is a codec the campaign never sees."""
        src = TARGET.read_text()
        for flag in (
            "BASE64_FORCE_PLAIN",
            "BASE64_FORCE_SSSE3",
            "BASE64_FORCE_SSE41",
            "BASE64_FORCE_SSE42",
            "BASE64_FORCE_AVX",
            "BASE64_FORCE_AVX2",
            "BASE64_FORCE_AVX512",
        ):
            assert flag in src, f"{flag} missing — that codec is never exercised"

    def test_cpu_gate_before_forcing_a_codec(self):
        """Forcing a codec the CPU lacks is SIGILL, not a mismatch."""
        src = TARGET.read_text()
        assert "__builtin_cpu_supports" in src
        assert "b64_cpu_has" in src

    def test_input_is_bounded(self):
        """direct_lite runs in-process: an adversarial length stalls the run."""
        src = TARGET.read_text()
        assert "#define B64_FUZZ_MAX_IN" in src
        assert "plen > B64_FUZZ_MAX_IN" in src


class TestBuildWiring:
    def test_vendor_script_exists_and_is_executable(self):
        assert VENDOR_SCRIPT.is_file(), "tools/vendor_libbase64.sh is missing"
        assert VENDOR_SCRIPT.stat().st_mode & 0o111, "vendor_libbase64.sh is not executable"
        assert 'LIBBASE64_DIR="$VENDOR_DIR/libbase64"' in VENDOR_SCRIPT.read_text()

    def test_build_script_defines_path(self, build_src):
        assert 'LIBBASE64="${LIBBASE64_DIR:-$VENDOR/libbase64}"' in build_src

    def test_objects_compiled_without_the_shim(self, compile_body):
        """Hard Rule 9: `-include $SHIM` applies to every .c on a command
        line, so the library TUs must be compiled in their own pass."""
        assert "-include" not in compile_body, "library objects must not get the shim"

    def test_one_object_per_codec(self, build_src, compile_body):
        """Each SIMD TU needs its own -m flag, so the codec table in
        build_targets.sh must name every one and the compile pass must use
        that table. A codec missing from the table is a codec the target never
        sees, with no symptom beyond a narrower differential."""
        m = re.search(r'^LIBBASE64_CODECS="([^"]+)"', build_src, re.M)
        assert m, "LIBBASE64_CODECS table not found"
        for arch in ("ssse3", "sse41", "sse42", "avx", "avx2", "avx512"):
            assert f"{arch}:" in m.group(1), f"{arch} codec missing from LIBBASE64_CODECS"
        assert "lib/arch/$name/codec.c" in compile_body, "codec objects not built from the table"
        assert "$mflags" in compile_body, "per-codec -m flags not applied"

    def test_config_h_is_generated(self, compile_body):
        """lib/config.h is generated, not shipped: without it every codec.c
        and codec_choose.c fail to compile, or worse, resolves to a stale
        upstream one."""
        assert "config.h" in compile_body
        assert "HAVE_" in compile_body

    def test_regression_disabled_codec_off_in_config(self, build_src, compile_body, tmp_path):
        """A codec the probe drops is compiled without its -m flag, so the
        final config.h must turn it off (stub body). Writing HAVE_<codec> 1
        for it failed the whole target instead of falling back."""
        table = re.search(r'^LIBBASE64_CODECS="[^"]+"', build_src, re.M).group(0)
        func = re.search(r"^write_b64_config\(\)\s*\{.*?\n\}", build_src, re.M | re.S).group(0)
        script = f'{table}\n{func}\nwrite_b64_config "{tmp_path}" "" " avx512 sse42"\n'
        subprocess.run(["bash", "-c", script], check=True)

        cfg = (tmp_path / "config.h").read_text()
        assert "#define HAVE_AVX512 0" in cfg
        assert "#define HAVE_SSE42 0" in cfg
        assert "#define HAVE_AVX2 1" in cfg, "enabled codec dropped (adversarial: no over-disable)"
        assert '"$disabled"' in compile_body, "final config.h not given the disabled list"

    def test_gcc_fallback_does_not_lose_coverage(self, compile_body):
        m = re.search(r"cov_flag_for_cc \"\$cc\"", compile_body)
        assert m, "coverage flag must be gated on the compiler"
        assert "-fsanitize-coverage=trace-pc-guard" not in compile_body, (
            "clang-only guard flag hardcoded; gcc must get trace-pc via cov_flag_for_cc"
        )

    def test_target_is_built_and_in_the_matrix(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/libbase64_read.c"' in build_src
        assert '"libbase64_read"' in build_src, "libbase64_read missing from the feature matrix"

    def test_vendor_script_documented_in_header(self, build_src):
        assert "tools/vendor_libbase64.sh ->" in build_src


class TestBase64Mutator:
    def test_registered_in_format_category(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        assert "base64_chunk_mutate" in REGISTRY.names()
        assert REGISTRY.category_of("base64_chunk_mutate") == "format"

    def test_sniffer_accepts_valid_base64(self):
        from fuzzer_tool.core.operator_registry import _sniff_base64

        assert _sniff_base64(b"\x00ABCD==")

    def test_sniffer_rejects_binary(self):
        from fuzzer_tool.core.operator_registry import _sniff_base64

        assert not _sniff_base64(b"\x00\xff\xfe")

    def test_mutator_flip_keeps_header(self):
        from fuzzer_tool.core.mutations.base64 import B64_ALPHA, Base64Mutator, parse_base64

        # op 0 (flip) at index 1 -> alphabet[25] ('Z').
        rng = ScriptedRng(randints=[0, 1], choice_idxs=[25])
        mutated = Base64Mutator().mutate(b"\x00ABCD==", rng=rng)
        assert mutated == b"\x00A" + B64_ALPHA[25:26] + b"CD=="
        assert parse_base64(mutated) is not None

    def test_regression_sniffer_skips_stream_seed(self):
        """Stream modes put an arbitrary seed byte at offset 1; the payload
        starts at 2. Checking from offset 1 rejected most stream inputs."""
        from fuzzer_tool.core.operator_registry import _sniff_base64

        assert _sniff_base64(b"\x02\xffQUJD")
        assert not _sniff_base64(b"\x02\xffQU\xffJ"), "adversarial: bad payload byte"

    def test_regression_sniffer_rejects_encode_mode(self):
        """Encode payloads (modes 1, 3) are raw bytes, not base64."""
        from fuzzer_tool.core.operator_registry import _sniff_base64

        assert not _sniff_base64(b"\x01QUJD")
        assert not _sniff_base64(b"\x03\x00QUJD")
        assert _sniff_base64(b"\x04QUJD"), "only the low 2 bits select the mode"

    def test_regression_mutate_keeps_stream_seed(self):
        """The seed byte is header, not payload: a flip at payload index 0
        must land on offset 2."""
        from fuzzer_tool.core.mutations.base64 import B64_ALPHA, Base64Mutator

        rng = ScriptedRng(randints=[0, 0], choice_idxs=[25])
        mutated = Base64Mutator().mutate(b"\x02\xffQUJD", rng=rng)
        assert mutated == b"\x02\xff" + B64_ALPHA[25:26] + b"UJD"

    @pytest.mark.parametrize("payload", [b"QUJD", b"QUJDR", b"QUJDRA", b"QUJDREU", b"QQ=", b"Q="])
    def test_regression_pad_yields_valid_quantum(self, payload):
        """The pad op appended "==" regardless of length, so its output was
        mostly undecodable. It must now produce a valid quantum."""
        from fuzzer_tool.core.mutations.base64 import Base64Mutator, parse_base64

        pad_op = 4
        mutated = Base64Mutator().mutate(b"\x00" + payload, rng=ScriptedRng(randints=[pad_op]))
        body = mutated[1:]
        assert len(body) % 4 == 0
        stripped = payload.rstrip(b"=")
        assert body.rstrip(b"=") == stripped[: len(stripped) - (len(stripped) % 4 == 1)]
        base64.b64decode(body, validate=True)
        if body:
            assert parse_base64(mutated) == body

    def test_regression_generate_encode_mode_raw(self):
        """Encode modes take arbitrary bytes; generating base64 for them
        wasted the payload's byte range."""
        from fuzzer_tool.core.mutations.base64 import Base64Mutator

        raw = b"\xff\x00\x80"
        rng = ScriptedRng(randints=[1, len(raw)], randbytes=[raw])
        assert Base64Mutator()._generate_random_base64(rng=rng) == b"\x01" + raw

    def test_regression_generate_stream_decode(self):
        """Stream decode: seed byte, then a valid, padded base64 payload."""
        from fuzzer_tool.core.mutations.base64 import B64_ALPHA, Base64Mutator, parse_base64

        raw = b"\x00\x41\xbf"
        rng = ScriptedRng(randints=[2, len(raw)], randbytes=[b"\x07", raw])
        out = Base64Mutator()._generate_random_base64(rng=rng)

        chars = bytes(B64_ALPHA[b % len(B64_ALPHA)] for b in raw)
        assert out == b"\x02\x07" + chars + b"="
        assert parse_base64(out) == chars + b"="
        base64.b64decode(out[2:], validate=True)

    def test_generate_respects_max_len(self):
        from fuzzer_tool.core.mutations.base64 import Base64Mutator

        rng = ScriptedRng(randints=[1, 8], randbytes=[bytes(8)])
        assert len(Base64Mutator()._generate_random_base64(max_len=4, rng=rng)) == 4


def _cpu_has(feature: str) -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return feature in flags


class Sanitizer(Enum):
    """(compile flags, link flags). clang has no ASAN runtime here; link
    gcc's libasan, as build_targets.sh does for the .so targets."""

    NONE = ((), ())
    ASAN = (("-fsanitize=address",), ("-lasan",))


# The harness binary is not linked with libasan first; ASAN must not refuse it.
ASAN_ENV = {"ASAN_OPTIONS": "verify_asan_link_order=0:detect_leaks=0"}


class Avx2Path(Enum):
    """AVX2 encoder build: portable C loop, or the x86-64 default inline asm."""

    C = ("-DBASE64_AVX2_USE_ASM=0",)
    ASM = ()


def _compile(
    tree: Path,
    out: Path,
    cfg: Path,
    san: Sanitizer = Sanitizer.NONE,
    avx2: Avx2Path = Avx2Path.C,
) -> Path:
    """Build the target plus a file harness against one SIMD codec (AVX2).

    config.h mirrors what build_targets.sh generates: every codec macro
    defined, only AVX2 enabled. All codec objects are still linked, because
    codec_choose_forced() names all seven unconditionally; the disabled ones
    compile to upstream's stubs.
    """
    cfg.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (cfg / "config.h").write_text(
        "#define HAVE_SSSE3 0\n"
        "#define HAVE_SSE41 0\n"
        "#define HAVE_SSE42 0\n"
        "#define HAVE_AVX 0\n"
        "#define HAVE_AVX2 1\n"
        "#define HAVE_AVX512 0\n"
        "#define HAVE_NEON32 0\n"
        "#define HAVE_NEON64 0\n"
    )

    harness = out / "harness.c"
    harness.write_text(
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <stddef.h>\n"
        "void __afl_map_edge(unsigned int e) { (void)e; }\n"
        "extern int fuzz_shm_run(const unsigned char *buf, size_t size);\n"
        "int main(int argc, char **argv) {\n"
        '    FILE *f = fopen(argv[1], "rb");\n'
        "    if (!f) { return 2; }\n"
        "    fseek(f, 0, SEEK_END);\n"
        "    long n = ftell(f);\n"
        "    fseek(f, 0, SEEK_SET);\n"
        "    unsigned char *buf = malloc((size_t)n + 1);\n"
        "    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { return 2; }\n"
        "    fclose(f);\n"
        "    int rc = fuzz_shm_run(buf, (size_t)n);\n"
        "    free(buf);\n"
        "    return rc;\n"
        "}\n"
    )

    sources = {
        "lib": (tree / "lib" / "lib.c", []),
        "codec_choose": (tree / "lib" / "codec_choose.c", []),
        "tables": (tree / "lib" / "tables" / "tables.c", []),
        "generic": (tree / "lib" / "arch" / "generic" / "codec.c", []),
        "avx2": (
            tree / "lib" / "arch" / "avx2" / "codec.c",
            # The x86-64 default is the inline-asm loop (enc_loop_asm.c);
            # Avx2Path.C forces the portable one (enc_translate.c).
            ["-mavx2", *avx2.value],
        ),
        "ssse3": (tree / "lib" / "arch" / "ssse3" / "codec.c", []),
        "sse41": (tree / "lib" / "arch" / "sse41" / "codec.c", []),
        "sse42": (tree / "lib" / "arch" / "sse42" / "codec.c", []),
        "avx": (tree / "lib" / "arch" / "avx" / "codec.c", []),
        "avx512": (tree / "lib" / "arch" / "avx512" / "codec.c", []),
        "neon32": (tree / "lib" / "arch" / "neon32" / "codec.c", []),
        "neon64": (tree / "lib" / "arch" / "neon64" / "codec.c", []),
    }
    objs = []
    for name, (src, tree_flags) in sources.items():
        obj = out / f"{name}.o"
        subprocess.run(
            [
                "clang",
                "-O1",
                "-g",
                "-fPIC",
                "-I",
                str(cfg),
                "-I",
                str(tree / "lib"),
                "-I",
                str(tree / "include"),
                *tree_flags,
                *san.value[0],
                "-c",
                str(src),
                "-o",
                str(obj),
            ],
            check=True,
        )
        objs.append(str(obj))

    target_obj = out / "target.o"
    subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-fPIC",
            "-I",
            str(tree / "include"),
            *san.value[0],
            "-c",
            str(TARGET),
            "-o",
            str(target_obj),
        ],
        check=True,
    )
    harness_obj = out / "harness.o"
    subprocess.run(["clang", "-O1", "-g", "-c", str(harness), "-o", str(harness_obj)], check=True)

    binary = out / "run"
    subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-o",
            str(binary),
            str(harness_obj),
            str(target_obj),
            *objs,
            *san.value[1],
        ],
        check=True,
    )
    return binary


def _run(binary: Path, data: bytes) -> int:
    payload = binary.parent / "input.bin"
    payload.write_bytes(data)
    return subprocess.run([str(binary), str(payload)], capture_output=True).returncode


def _run_asan(binary: Path, data: bytes) -> tuple[int, str]:
    payload = binary.parent / "input.bin"
    payload.write_bytes(data)
    env = {**os.environ, **ASAN_ENV}
    proc = subprocess.run([str(binary), str(payload)], capture_output=True, text=True, env=env)
    return proc.returncode, proc.stderr


@pytest.mark.skipif(
    shutil.which("clang") is None or not (VENDOR_TREE / "lib" / "lib.c").is_file(),
    reason="needs clang and tools/vendor_libbase64.sh (FUZZ_VENDOR_ROOT/libbase64)",
)
class TestOracle:
    @pytest.fixture(scope="class")
    def control(self, tmp_path_factory) -> Path:
        return _compile(
            VENDOR_TREE, tmp_path_factory.mktemp("b64control"), tmp_path_factory.mktemp("b64cfg")
        )

    def test_control_passes_on_clean_tree(self, control):
        """Hard Rule 46: a differential that cannot fail on identical inputs
        is not an oracle. The control has to be green before the fault means
        anything."""
        assert _run(control, FAULT_INPUT) == 0, "differential fired on a clean tree"

    @pytest.mark.parametrize("name", sorted(EDGE_INPUTS))
    def test_edge_inputs_do_not_abort(self, control, name):
        assert _run(control, EDGE_INPUTS[name]) == 0, f"{name} aborted the target"

    @pytest.mark.parametrize("payload", [b"QUJ", b"QQ=", b"Q"])
    def test_regression_stream_partial_quantum(self, control, payload):
        """One-shot decode fails a trailing partial quantum (state.bytes != 0);
        a chunked stream still waits for input and returns 1. Comparing raw
        return codes aborted on every non-quantum-aligned payload."""
        assert _run(control, b"\x02\x00" + payload) == 0, "stream decode false positive"

    def test_regression_stream_splits_mid_quantum(self, control, tmp_path):
        """Stream modes must split inside a 4-char quantum for most seeds.
        Chunks drawn from 1..4096 left a 200-char payload whole ~95% of the
        time. Fault: resuming at state.bytes == 1 flips a carry bit, which only
        a mid-quantum split reaches. Every seed is enumerated (no luck loop);
        the clean control must pass all of them first (Hard Rule 46)."""
        payload = base64.b64encode(bytes(range(150)))
        inputs = [bytes([2, seed]) + payload for seed in range(256)]
        assert all(_run(control, x) == 0 for x in inputs), "control fired on clean tree"

        tree = tmp_path / "tree"
        shutil.copytree(VENDOR_TREE, tree)
        head = tree / "lib" / "arch" / "generic" / "dec_head.c"
        text = head.read_text()
        assert STATE_CARRY in text, "dec_head.c changed — update the fault injection"
        head.write_text(text.replace(STATE_CARRY, STATE_CARRY + MID_QUANTUM_FAULT, 1))

        binary = _compile(tree, tmp_path / "faulty", tmp_path / "faultycfg")
        caught = sum(_run(binary, x) == SIGABRT_RC for x in inputs)
        assert caught >= len(inputs) // 2, f"mid-quantum split reached by {caught}/256 seeds"

    def test_regression_input_overread_caught(self, tmp_path):
        """The payload pointed into the runner's buffer (here malloc(n + 1)),
        so a codec reading one byte past the input stayed in bounds and ASAN
        was silent. The target must decode from an exact-size copy.
        Fault: base64_stream_decode reads src[srclen]. Control first."""
        clean = _compile(VENDOR_TREE, tmp_path / "clean", tmp_path / "cleancfg", Sanitizer.ASAN)
        for data in (FAULT_INPUT, OVERREAD_INPUT, *EDGE_INPUTS.values()):
            rc, err = _run_asan(clean, data)
            assert rc == 0 and "AddressSanitizer" not in err, f"control fired: {err[:300]}"

        tree = tmp_path / "tree"
        shutil.copytree(VENDOR_TREE, tree)
        lib = tree / "lib" / "lib.c"
        text = lib.read_text()
        assert STREAM_DEC_RETURN in text, "lib.c changed — update the fault injection"
        lib.write_text(text.replace(STREAM_DEC_RETURN, OVERREAD_FAULT + STREAM_DEC_RETURN, 1))

        binary = _compile(tree, tmp_path / "faulty", tmp_path / "faultycfg", Sanitizer.ASAN)
        rc, err = _run_asan(binary, OVERREAD_INPUT)
        assert "heap-buffer-overflow" in err, f"over-read not reported (rc={rc})"

    @pytest.mark.skipif(not _cpu_has("avx2"), reason="the injected fault lives in the AVX2 codec")
    @pytest.mark.parametrize(
        "path,lut_file", [(Avx2Path.C, "enc_translate.c"), (Avx2Path.ASM, "enc_loop_asm.c")]
    )
    def test_injected_avx2_fault_aborts(self, tmp_path, path, lut_file):
        """Falsification: corrupt one LUT entry of the AVX2 encoder and the
        differential must abort. Anything else means the target is not
        comparing codec output. ASM is what campaigns run; ASAN does not see
        its stores, so the differential is its only oracle."""
        tree = tmp_path / "tree"
        shutil.copytree(VENDOR_TREE, tree)
        translate = tree / "lib" / "arch" / "avx2" / lut_file
        text = translate.read_text()
        assert LUT_HEAD in text, "AVX2 translate LUT not found — update the fault injection"
        translate.write_text(text.replace(LUT_HEAD, LUT_HEAD.replace("65, 71", "66, 71", 1), 1))

        binary = _compile(tree, tmp_path / "faulty", tmp_path / "faultycfg", avx2=path)
        assert _run(binary, FAULT_INPUT) == SIGABRT_RC, "corrupted AVX2 encoder was not detected"
