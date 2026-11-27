"""Regression: small inputs starved push-mode demuxers of chain calls.

``gstreamer_read`` pushed 4 KiB buffers then EOS. avidemux's push-mode chain
advances one state per call (START -> HEADER -> MOVI), so an input under
4 KiB was one call: stream init ran, ``hdrl`` never parsed. Measured on a
258-byte grown AVI: 2,746 edges; the same file padded to 4,362 B: 3,594.
The harness now pushes empty drain buffers before EOS.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from .test_regression_gstreamer_caps import GST_H, HARNESS, HARNESS_DECLS, TARGET, TYPEFIND_H

SMALL, LARGE = 64, 9000
CHUNK = 4096
MIN_DRAIN = 3  # states after init: HEADER, MOVI, data

PUSH_STUB = """GstFlowReturn gst_pad_push(GstPad *p, GstBuffer *b)
{
    (void)p;
    last_push = strchr(events, '2') ? GST_FLOW_OK : GST_FLOW_NOT_NEGOTIATED;
    gst_buffer_unref(b);
    return last_push;
}"""

COUNTING_STUB = """static int pushes, pushes_before_eos;
GstFlowReturn gst_pad_push(GstPad *p, GstBuffer *b)
{
    (void)p;
    pushes++;
    if (!strchr(events, '4')) pushes_before_eos++;
    last_push = strchr(events, '2') ? GST_FLOW_OK : GST_FLOW_NOT_NEGOTIATED;
    gst_buffer_unref(b);
    return last_push;
}"""

MAIN = f"""int main(void)
{{
    static unsigned char buf[{LARGE}];
    for (int i = 0; i < 2; i++) {{
        size_t n = i ? {LARGE} : {SMALL};
        reset_probe(); pushes = pushes_before_eos = 0;
        fuzz_gstreamer(buf, n);
        printf("n%zu=%d,%d ", n, pushes, pushes_before_eos);
    }}
    printf("\\n");
    return 0;
}}"""


@pytest.fixture(scope="module")
def counts(tmp_path_factory):
    """Compile the real target against counting stubs; pushes per input size."""
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang not available")
    assert PUSH_STUB in HARNESS
    harness = HARNESS.replace(PUSH_STUB, COUNTING_STUB)
    harness = harness[: harness.index("int main(void)")] + HARNESS_DECLS + MAIN

    d = tmp_path_factory.mktemp("gst_drain")
    (d / "gst" / "base").mkdir(parents=True)
    (d / "gst" / "gst.h").write_text(GST_H)
    (d / "gst" / "base" / "gsttypefindhelper.h").write_text(TYPEFIND_H)
    (d / "harness.c").write_text(harness)
    exe = d / "probe"
    cmd = [
        clang,
        "-std=c11",
        "-O0",
        "-I",
        str(d),
        "-o",
        str(exe),
        str(d / "harness.c"),
        str(TARGET),
    ]
    built = subprocess.run(cmd, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    return {int(k[1:]): tuple(map(int, v.split(","))) for k, v in (p.split("=") for p in out)}


def test_regression_gst_drain(counts):
    """A sub-chunk input gets data push + drain pushes, all before EOS."""
    total, before_eos = counts[SMALL]
    assert total >= 1 + MIN_DRAIN
    assert before_eos == total


def test_large_input_keeps_chunks(counts):
    """Falsification: data still goes in 4 KiB chunks; drain is added, not substituted."""
    data_pushes = -(-LARGE // CHUNK)
    total, before_eos = counts[LARGE]
    assert total == data_pushes + (counts[SMALL][0] - 1)
    assert before_eos == total
