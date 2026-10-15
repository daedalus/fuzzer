"""FFmpeg harnesses must honour libavcodec's decode-API contract.

``avcodec_send_packet``/``avcodec_receive_frame`` serve audio and video
decoders only; subtitles use ``avcodec_decode_subtitle2``. A harness that
breaks this aborts inside libavcodec on any payload, and the abort looks
like a finding. FFmpeg#23945 was such an abort, reported and closed upstream
as harness misuse.

The contract, enforced on every ``targets/ffmpeg*_read.c``:

1.  The API is derived from the decoder's own type (``codec->type``),
    which is what libavcodec dispatches on, in one function:
    ``fuzz_decode_api``. ``codecpar->codec_type`` comes from the input and
    is copied into the context on decoder reuse without re-validation.
2.  The frame API is a whitelist (audio, video). Unknown types get none.
3.  Each decode call sits under a check for its own API.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESSES = sorted((ROOT / "targets").glob("ffmpeg*_read.c"))

# Decode call -> the only API value allowed to guard it.
CALL_API = {
    "avcodec_send_packet": "DECODE_API_FRAME",
    "avcodec_receive_frame": "DECODE_API_FRAME",
    "avcodec_decode_subtitle2": "DECODE_API_SUBTITLE",
}

# fuzz_decode_api must map exactly these media types.
TYPE_API = {
    "AVMEDIA_TYPE_VIDEO": "DECODE_API_FRAME",
    "AVMEDIA_TYPE_AUDIO": "DECODE_API_FRAME",
    "AVMEDIA_TYPE_SUBTITLE": "DECODE_API_SUBTITLE",
}

GUARD = re.compile(r"(?:==\s*|case\s+)(DECODE_API_\w+)")
CALL = re.compile(r"\b(" + "|".join(CALL_API) + r")\s*\(")
API_FN = re.compile(r"fuzz_decode_api\s*\([^)]*\)\s*\{(.*?)\n\}", re.S)
FN_HEADER = re.compile(
    r"^(?!\s*(?:if|else|while|for|switch|do)\b)\s*[\w\s\*]+?\b(\w+)\s*\([^;]*\)\s*$"
)
BARE_IF = re.compile(r"\bif\s*\(([^;{}]*)\)[^;{}]*$")
API_ASSIGN = re.compile(r"(?:->|\.)api\s*=(?!=)\s*([^;]*);")


def _strip_comments(src: str) -> str:
    """Drop C comments so prose mentioning an API does not count."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _fn_guard(code: str, name: str, seen: frozenset[str]) -> str | None:
    """Return the API guarding every call of helper ``name``, else None.

    One shared guard across all call sites: ``fuzz_decode_av`` is only
    reached under ``api == DECODE_API_FRAME``.
    """
    if name in seen:
        return None

    sites = [
        m.start()
        for m in re.finditer(rf"\b{name}\s*\(", code)
        if not re.search(r"\)\s*\{", code[m.end() : code.find("\n", m.end())])
    ]
    guards = {_guard(code, pos, seen | {name}) for pos in sites}
    return guards.pop() if len(guards) == 1 else None


def _guard(code: str, pos: int, seen: frozenset[str] = frozenset()) -> str | None:
    """Return the API checked by the innermost block enclosing ``pos``.

    Walks back over balanced braces; a block's guard is the ``== X`` in its
    header (``if (api == X) {``) or the last ``case X:`` before ``pos``.
    A braceless ``if (api == X) call;`` guards its own statement.
    Reaching a function body defers to that function's call sites.
    """
    stmt = code[max(code.rfind(c, 0, pos) for c in ";{}") + 1 : pos]
    bare = BARE_IF.search(stmt)
    hit = GUARD.findall(bare.group(1)) if bare else []
    if hit:
        return hit[-1]

    depth = 0
    end = pos
    for i in range(pos - 1, -1, -1):
        ch = code[i]
        if ch == "}":
            depth += 1
            continue
        if ch != "{":
            continue
        if depth:
            depth -= 1
            continue

        # Enclosing block opens at i: cases inside it, then its header.
        cases = re.findall(r"case\s+(DECODE_API_\w+)\s*:", code[i:end])
        if cases:
            return cases[-1]
        start = max(code.rfind(c, 0, i) for c in ";{}") + 1
        header = code[start:i]
        hit = GUARD.findall(header)
        if hit:
            return hit[-1]
        fn = FN_HEADER.search(header)
        if fn:
            return _fn_guard(code, fn.group(1), seen)
        end = start
    return None


def _violations(src: str) -> list[str]:
    """Return contract violations found in harness source ``src``."""
    code = _strip_comments(src)
    errors = []

    # 1+2: one mapping function, whitelist, default to none.
    fn = API_FN.search(code)
    if not fn:
        return ["no fuzz_decode_api"]

    body = fn.group(1)
    for media, api in TYPE_API.items():
        if not re.search(rf"case\s+{media}\s*:[^;]*?return\s+{api}\s*;", body, re.S):
            errors.append(f"{media} not mapped to {api}")
    for media in re.findall(r"case\s+(\w+)\s*:", body):
        if media not in TYPE_API:
            errors.append(f"unexpected case {media}")
    if not re.search(r"default\s*:\s*return\s+DECODE_API_NONE\s*;", body):
        errors.append("default is not DECODE_API_NONE")
    if "codec->type" not in body:
        errors.append("fuzz_decode_api does not read codec->type")

    # 1: the slot's API is only ever set from the mapping function, and is set.
    rhss = API_ASSIGN.findall(code)
    if not any(rhs.startswith("fuzz_decode_api(") for rhs in rhss):
        errors.append("api never bound from fuzz_decode_api")
    for rhs in rhss:
        if not rhs.startswith("fuzz_decode_api(") and rhs != "DECODE_API_NONE":
            errors.append(f"api assigned from {rhs!r}")

    # 3: the innermost enclosing API check is the call's own API.
    for m in CALL.finditer(code):
        got = _guard(code, m.start())
        want = CALL_API[m.group(1)]
        if got != want:
            errors.append(f"{m.group(1)} guarded by {got}, want {want}")

    return errors


@pytest.mark.parametrize("path", HARNESSES, ids=lambda p: p.name)
def test_regression_ffmpeg_api_contract(path: Path) -> None:
    """Falsification: every shipped harness passes the contract."""
    assert _violations(path.read_text()) == []


def test_harnesses_found() -> None:
    """The glob must not silently match nothing."""
    assert HARNESSES


GOOD = """
static DecodeApi fuzz_decode_api(const AVCodec *codec) {
    switch (codec->type) {
    case AVMEDIA_TYPE_VIDEO:
    case AVMEDIA_TYPE_AUDIO:    return DECODE_API_FRAME;
    case AVMEDIA_TYPE_SUBTITLE: return DECODE_API_SUBTITLE;
    default:                    return DECODE_API_NONE;
    }
}
void f(void) {
    slot->api = fuzz_decode_api(slot->ctx->codec);
    if (api == DECODE_API_SUBTITLE) {
        avcodec_decode_subtitle2(ctx, &sub, &got, pkt);
    } else if (api == DECODE_API_FRAME) {
        avcodec_send_packet(ctx, pkt);
        while (avcodec_receive_frame(ctx, frame) >= 0) {}
    }
}
"""


def test_control_good_passes() -> None:
    """Control: the reference shape must pass, or the lint is broken."""
    assert _violations(GOOD) == []


@pytest.mark.parametrize(
    "old, new, needle",
    [
        # FFmpeg#23945 shape: subtitle blacklist, frame API in the else.
        (
            "} else if (api == DECODE_API_FRAME) {",
            "} else {",
            "avcodec_send_packet guarded by None",
        ),
        # API chosen from input-controlled codecpar instead of the decoder.
        (
            "slot->api = fuzz_decode_api(slot->ctx->codec);",
            "slot->api = par->codec_type;",
            "api assigned from",
        ),
        # Frame API extended past audio/video.
        (
            "default:                    return DECODE_API_NONE;",
            "default:                    return DECODE_API_FRAME;",
            "default is not DECODE_API_NONE",
        ),
        # Subtitles mapped to the frame API.
        (
            "case AVMEDIA_TYPE_SUBTITLE: return DECODE_API_SUBTITLE;",
            "case AVMEDIA_TYPE_SUBTITLE: return DECODE_API_FRAME;",
            "AVMEDIA_TYPE_SUBTITLE not mapped",
        ),
        # Extra media type let into the frame API.
        (
            "    case AVMEDIA_TYPE_AUDIO:    return DECODE_API_FRAME;",
            "    case AVMEDIA_TYPE_AUDIO:    return DECODE_API_FRAME;\n"
            "    case AVMEDIA_TYPE_DATA:     return DECODE_API_FRAME;",
            "unexpected case AVMEDIA_TYPE_DATA",
        ),
        # Decoder open never binds the API: packets silently skip decoding.
        (
            "slot->api = fuzz_decode_api(slot->ctx->codec);",
            "",
            "api never bound from fuzz_decode_api",
        ),
        # Mapping keyed on the context type, which codecpar overwrites.
        ("switch (codec->type)", "switch (ctx->codec_type)", "does not read codec->type"),
        # Unguarded call.
        (
            "    }\n}\n",
            "    }\n    avcodec_send_packet(ctx, pkt);\n}\n",
            "avcodec_send_packet guarded by None",
        ),
    ],
)
def test_adversarial_breaches_caught(old: str, new: str, needle: str) -> None:
    """Adversarial: each contract breach is reported."""
    assert old in GOOD
    errors = _violations(GOOD.replace(old, new, 1))
    assert any(needle in e for e in errors), errors


HELPER = """
static unsigned fuzz_decode_av(int si) {
    return avcodec_send_packet(ctx, pkt);
}
void g(void) {
    if (api == DECODE_API_FRAME) fuzz_decode_av(si);
    if (api == DECODE_API_FRAME) { fuzz_decode_av(si); }
}
"""


def test_helper_guarded_at_callers() -> None:
    """Control: a helper whose callers all check its API passes."""
    code = _strip_comments(HELPER)
    assert _guard(code, code.index("avcodec_send_packet")) == "DECODE_API_FRAME"


def test_adversarial_helper_unguarded_caller() -> None:
    """One caller outside the check breaks the helper's guard."""
    code = _strip_comments(HELPER.rstrip()[:-1] + "    fuzz_decode_av(si);\n}\n")
    assert _guard(code, code.index("avcodec_send_packet")) is None


def test_adversarial_comment_is_not_guard() -> None:
    """A guard inside a comment must not satisfy the check."""
    src = GOOD.replace(
        "} else if (api == DECODE_API_FRAME) {",
        "} else { /* api == DECODE_API_FRAME */",
    )
    assert any("avcodec_send_packet" in e for e in _violations(src))
