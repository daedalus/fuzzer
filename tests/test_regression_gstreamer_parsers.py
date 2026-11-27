"""GStreamer target: demuxer pads feed parsers, and every per-exec object is released.

Without a parser the demuxer's output stops at a drop sink, so codec configs
(avcC/hvcC/esds, SSA headers) are never parsed: h264parse/aacparse/ssaparse
code is unreachable. The target compiles against stubs that model refcounts,
pad links, one parser factory (``video/x-h264``) and one subtitle decoder
(``application/x-ass``, as ssaparse registers as ``Codec/Decoder/Subtitle``).
Every stub element also has a static ``src`` pad, as wavparse/auparse/subparse do.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from enum import IntEnum
from pathlib import Path

import pytest

from tests.test_regression_gstreamer_caps import GST_H, TARGET, TYPEFIND_H

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "tools" / "build_targets.sh"

H264 = "video/x-h264"
ASS = "application/x-ass"
FLOOD_PADS = 200

HARNESS = r"""
#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

int fuzz_gstreamer(const unsigned char *, size_t);

enum { K_ELEM = 1, K_PAD = 2 };
#define MAX_OBJS 8192
#define MAX_DYN 512

static StubObj *objs[MAX_OBJS];
static int n_objs, exec_first;
static StubObj *dyn[MAX_DYN];
static int n_dyn;
static StubObj *demux;
static void (*pad_cb)(GstElement *, GstPad *, gpointer);
static int parsers_created, parser_links, uaf, caps_live, lists_live, pushed;
static int parser_not_null, static_links, make_fail, abandon;
static jmp_buf abandon_jmp;
/* Factory name = the caps it accepts. */
static GstElementFactory h264_factory = { "video/x-h264" };
static GstElementFactory ass_factory = { "application/x-ass" };
static GList ass_node = { &ass_factory, NULL };
static GList h264_node = { &h264_factory, NULL };
static GList both_node = { &h264_factory, &ass_node };

/* scenario: pad count, caps name (NULL = no caps), ANY flag */
static int sc_pads, sc_any;
static const char *sc_caps;

static StubObj *new_obj(int kind) {
    StubObj *o = calloc(1, sizeof *o);
    o->refs = 1; o->floating = 1; o->kind = kind;
    if (n_objs < MAX_OBJS) objs[n_objs++] = o;
    return o;
}
static StubObj *new_elem(int parser) {
    StubObj *e = new_obj(K_ELEM);
    e->parser = parser;
    e->sinkpad = new_obj(K_PAD); e->sinkpad->floating = 0; e->sinkpad->parent = e;
    e->srcpad = new_obj(K_PAD); e->srcpad->floating = 0; e->srcpad->parent = e;
    return e;
}

/* A dead pad's peer is the use-after-free a wrong teardown order causes. */
static void unlink_pad(StubObj *p) {
    if (!p || !p->peer) return;
    if (p->peer->refs <= 0) uaf = 1;
    p->peer->peer = NULL;
    p->peer = NULL;
}
static void dispose(StubObj *o) {
    if (o->kind == K_ELEM) {
        if (o->parser && o->state != GST_STATE_NULL) parser_not_null = 1;
        unlink_pad(o->sinkpad); unlink_pad(o->srcpad);
        o->sinkpad->refs = 0; o->srcpad->refs = 0;
        if (o == demux)
            for (int i = 0; i < n_dyn; i++) { unlink_pad(dyn[i]); dyn[i]->refs = 0; }
        return;
    }
    unlink_pad(o);
}

void __afl_map_edge(unsigned int id) { (void)id; }
void g_setenv(const gchar *a, const gchar *b, gboolean c) { (void)a; (void)b; (void)c; }
const gchar *g_getenv(const gchar *a) { (void)a; return NULL; }
void g_log_set_default_handler(GLogFunc f, gpointer d) { (void)f; (void)d; }
void g_log_set_handler(const gchar *a, GLogLevelFlags b, GLogFunc f, gpointer d)
{ (void)a; (void)b; (void)f; (void)d; }
void gst_init(int *a, char ***b) { (void)a; (void)b; }
static GstCaps *tracked_caps(const char *name)
{
    GstCaps *c = calloc(1, sizeof *c);
    c->name = name; c->tracked = 1;
    caps_live++;
    return c;
}
GstCaps *gst_type_find_helper_for_data(GstObject *o, const guint8 *d, gsize n,
                                       GstTypeFindProbability *p)
{ (void)o; (void)d; (void)n; *p = GST_TYPE_FIND_NONE; return tracked_caps("video/quicktime"); }
GstStructure *gst_caps_get_structure(const GstCaps *c, guint8 i)
{ (void)c; (void)i; static GstStructure s; return &s; }
const gchar *gst_structure_get_name(const GstStructure *s) { (void)s; return "video/quicktime"; }
GstCaps *gst_caps_from_string(const gchar *s) { return tracked_caps(s); }
void gst_caps_unref(GstCaps *c) { if (c->tracked) { caps_live--; free(c); } }
GstElement *gst_element_factory_make(const gchar *a, const gchar *b)
{ (void)a; (void)b; return make_fail ? NULL : new_elem(0); }
GstElement *gst_element_factory_create(GstElementFactory *f, const gchar *n)
{ (void)f; (void)n; parsers_created++; return new_elem(1); }
void g_signal_connect(gpointer a, const gchar *b, GCallback c, gpointer d)
{ (void)b; (void)d; demux = a; pad_cb = (void (*)(GstElement *, GstPad *, gpointer))c; }
GstPad *gst_pad_new(const gchar *a, GstPadDirection d) { (void)a; (void)d; return new_obj(K_PAD); }
GstPadDirection gst_pad_get_direction(const GstPad *p) { (void)p; return GST_PAD_SRC; }
void gst_pad_set_chain_function_full(GstPad *p, GstFlowReturn (*f)(GstPad *, GstObject *, GstBuffer *), gpointer d)
{ (void)p; (void)f; (void)d; }
void gst_pad_set_event_function_full(GstPad *p, gboolean (*f)(GstPad *, GstObject *, GstEvent *), gpointer d)
{ (void)p; (void)f; (void)d; }
void gst_pad_set_active(GstPad *p, gboolean a) { (void)p; (void)a; }
GstPad *gst_element_get_static_pad(GstElement *e, const gchar *n)
{
    StubObj *p = strcmp(n, "sink") == 0 ? e->sinkpad : e->srcpad;
    p->refs++;
    return p;
}
GstPadLinkReturn gst_pad_link(GstPad *a, GstPad *b)
{
    a->peer = b; b->peer = a;
    if (demux && a == demux->srcpad) static_links++;
    if (b->parent && b->parent->parser && a->parent == demux) parser_links++;
    return GST_PAD_LINK_OK;
}
GstStateChangeReturn gst_element_set_state(GstElement *e, GstState s) { e->state = s; return 0; }
void gst_segment_init(GstSegment *s, GstFormat f) { (void)s; (void)f; }
gboolean gst_pad_push_event(GstPad *p, GstEvent *e) { (void)p; free(e); return TRUE; }
static GstEvent *new_event(int type) { GstEvent *e = malloc(sizeof *e); e->type = type; return e; }
GstEvent *gst_event_new_stream_start(const gchar *id) { (void)id; return new_event(EVENT_STREAM_START); }
GstEvent *gst_event_new_segment(const GstSegment *s) { (void)s; return new_event(EVENT_SEGMENT); }
GstEvent *gst_event_new_caps(GstCaps *c) { (void)c; return new_event(EVENT_CAPS); }
GstEvent *gst_event_new_eos(void) { return new_event(EVENT_EOS); }
void gst_event_unref(GstEvent *e) { free(e); }
GstBuffer *gst_buffer_new_memdup(gconstpointer d, gsize n) { (void)d; (void)n; GstBuffer *b = malloc(sizeof *b); b->type = 0; b->offset = 0; return b; }
void gst_buffer_unref(GstBuffer *b) { free(b); }

void g_object_unref(gpointer p)
{
    StubObj *o = p;
    if (--o->refs == 0) dispose(o);
}
gpointer gst_object_ref_sink(gpointer p)
{
    StubObj *o = p;
    if (o->floating) o->floating = 0; else o->refs++;
    return p;
}

GList *gst_element_factory_list_get_elements(GstElementFactoryListType t, GstRank r)
{
    (void)r;
    int parsers = (t & GST_ELEMENT_FACTORY_TYPE_PARSER) != 0;
    int decoders = (t & GST_ELEMENT_FACTORY_TYPE_DECODER) != 0;
    return parsers && decoders ? &both_node : parsers ? &h264_node : decoders ? &ass_node : NULL;
}
GList *g_list_sort(GList *l, GCompareFunc f) { (void)f; return l; }
int gst_plugin_feature_rank_compare_func(const void *a, const void *b) { (void)a; (void)b; return 0; }
/* Real filtering intersects caps: ANY intersects every factory. */
GList *gst_element_factory_list_filter(GList *l, const GstCaps *c, GstPadDirection d, gboolean s)
{
    (void)d; (void)s;
    GstElementFactory *f = NULL;
    for (; l && !f; l = l->next) {
        GstElementFactory *cand = l->data;
        if (c->any || strcmp(c->name, cand->name) == 0) f = cand;
    }
    if (!f) return NULL;
    GList *n = calloc(1, sizeof *n);
    n->data = f;
    lists_live++;
    return n;
}
void gst_plugin_feature_list_free(GList *l)
{
    while (l) { GList *nx = l->next; free(l); lists_live--; l = nx; }
}
GstCaps *gst_pad_get_current_caps(GstPad *p)
{
    if (!p->caps) return NULL;
    GstCaps *src = p->caps, *c = tracked_caps(src->name);
    c->any = src->any;
    return c;
}
gboolean gst_caps_is_any(const GstCaps *c) { return c->any; }

/* The demuxer exposes its pads on the first buffer. */
GstFlowReturn gst_pad_push(GstPad *p, GstBuffer *b)
{
    (void)p;
    gst_buffer_unref(b);
    if (pushed++) return GST_FLOW_OK;
    static GstCaps pad_caps;
    pad_caps.any = sc_any; pad_caps.name = sc_caps ? sc_caps : "";
    for (int i = 0; i < sc_pads && n_dyn < MAX_DYN; i++) {
        StubObj *dp = new_obj(K_PAD);
        dp->floating = 0; dp->parent = demux;
        dp->caps = (sc_caps || sc_any) ? &pad_caps : NULL;
        dyn[n_dyn++] = dp;
        pad_cb(demux, dp, NULL);
    }
    if (abandon) longjmp(abandon_jmp, 1);    /* timeout guard abandons the exec */
    return GST_FLOW_OK;
}

/* mode: 0 normal, 1 demuxer creation fails, 2 previous exec abandoned mid-push */
static void run(const char *label, int pads, const char *caps, int any, int mode)
{
    unsigned char buf[64] = {0};
    sc_pads = pads; sc_caps = caps; sc_any = any;
    parsers_created = parser_links = uaf = caps_live = lists_live = pushed = 0;
    parser_not_null = 0; n_dyn = 0;
    fuzz_gstreamer(buf, sizeof buf);    /* first call runs init + pre-warm */
    if (mode == 2) {
        sc_pads = 64; sc_caps = "video/x-h264"; abandon = 1; pushed = 0; n_dyn = 0;
        if (!setjmp(abandon_jmp)) fuzz_gstreamer(buf, sizeof buf);
        abandon = 0; sc_pads = pads; sc_caps = caps;
    }
    exec_first = n_objs;
    parsers_created = parser_links = pushed = static_links = caps_live = 0; n_dyn = 0;
    make_fail = mode == 1;
    fuzz_gstreamer(buf, sizeof buf);
    make_fail = 0;
    int leaked = 0, linked = 0;
    for (int i = exec_first; i < n_objs; i++) leaked += objs[i]->refs > 0;
    for (int i = 0; i < n_dyn; i++) linked += dyn[i]->peer != NULL || dyn[i]->refs == 0;
    printf("%s parsers=%d links=%d leaked=%d uaf=%d caps_live=%d lists_live=%d "
           "not_null=%d static_links=%d\n", label, parsers_created, parser_links, leaked, uaf,
           caps_live, lists_live, parser_not_null, static_links);
}

int main(int argc, char **argv)
{
    int pads = atoi(argv[2]);
    const char *caps = strcmp(argv[3], "-") == 0 ? NULL : argv[3];
    run(argv[1], pads, caps, atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 0);
    return 0;
}
"""


def _parse(line: str) -> dict[str, int]:
    label, *pairs = line.split()
    return {k: int(v) for k, v in (p.split("=") for p in pairs)} | {"label": label}


@pytest.fixture(scope="module")
def probe_exe(tmp_path_factory) -> Path:
    """Compile the real target against the object-model stubs."""
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("gst_parsers")
    (d / "gst" / "base").mkdir(parents=True)
    (d / "gst" / "gst.h").write_text(GST_H)
    (d / "gst" / "base" / "gsttypefindhelper.h").write_text(TYPEFIND_H)
    (d / "harness.c").write_text(HARNESS)
    exe = d / "probe"
    cmd = [clang, "-std=c11", "-O0", "-g", "-fsanitize=address", "-I", str(d)]
    cmd += ["-o", str(exe), str(d / "harness.c"), str(TARGET)]
    compiled = subprocess.run(cmd, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    return exe


class Mode(IntEnum):
    """Exec scenario, mirrors ``run()``'s mode argument in the C harness."""

    NORMAL = 0
    MAKE_FAILS = 1
    AFTER_ABANDON = 2


def run_probe(
    exe: Path, pads: int, caps: str | None, any_caps: bool = False, mode: Mode = Mode.NORMAL
) -> dict[str, int]:
    """One exec with *pads* demuxer pads carrying *caps*; returns the probe counters."""
    args = [str(exe), "case", str(pads), caps or "-", str(int(any_caps)), str(int(mode))]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return _parse(out.strip())


def test_regression_parser_follows_demux_pad(probe_exe):
    """A pad whose caps a parser accepts is linked into that parser."""
    r = run_probe(probe_exe, 1, H264)
    assert r["parsers"] == 1, r
    assert r["links"] == 1, r


def test_regression_exec_releases_everything(probe_exe):
    """Parsers, drop sinks and caps die with the exec: memory stays bounded."""
    for caps in (H264, "audio/x-raw", None):
        r = run_probe(probe_exe, 3, caps)
        assert (r["leaked"], r["caps_live"], r["lists_live"]) == (0, 0, 0), (caps, r)


def test_regression_parser_torn_down_after_demux(probe_exe):
    """Parsers stop (state NULL) and die after the demuxer: no peer use-after-free."""
    r = run_probe(probe_exe, 2, H264)
    assert r["uaf"] == 0, r
    assert r["not_null"] == 0, r


def test_falsify_unmatched_caps_get_no_parser(probe_exe):
    """Falsification: caps no parser accepts must not create one."""
    for caps in ("audio/x-raw", None):
        r = run_probe(probe_exe, 2, caps)
        assert r["parsers"] == 0, (caps, r)
        assert r["links"] == 0, (caps, r)


def test_adversarial_any_caps_skip_parsers(probe_exe):
    """ANY intersects every factory; picking one would be arbitrary."""
    r = run_probe(probe_exe, 1, None, any_caps=True)
    assert r["parsers"] == 0, r


def test_adversarial_pad_flood_is_bounded(probe_exe):
    """A file with hundreds of tracks cannot grow per-exec objects without bound."""
    r = run_probe(probe_exe, FLOOD_PADS, H264)
    assert 0 < r["parsers"] < FLOOD_PADS, r
    assert (r["leaked"], r["uaf"], r["caps_live"]) == (0, 0, 0), r


def _meson_opts() -> str:
    text = BUILD.read_text()
    start = text.index("local meson_opts=(")
    return text[start : text.index("\n    )\n", start)]


def test_reachability_plugins_built_and_registered():
    """Every plugin the target registers is enabled in the build."""
    opts = _meson_opts()
    src = TARGET.read_text()
    plugins = {
        "gst-plugins-base:subparse": "subparse",
        "gst-plugins-good:id3demux": "id3demux",
        "gst-plugins-good:apetag": "apetag",
        "gst-plugins-good:auparse": "auparse",
        "gst-plugins-good:audioparsers": "audioparsers",
        "gst-plugins-bad:videoparsers": "videoparsersbad",
    }
    for opt, name in plugins.items():
        assert f"-D{opt}=enabled" in opts, opt
        assert f"GST_PLUGIN_STATIC_DECLARE({name});" in src, name
        assert f"GST_PLUGIN_STATIC_REGISTER({name});" in src, name


def test_typefind_routes_new_demuxers():
    """Typefound caps map to the new demuxers; the byte-0 fallback is unchanged."""
    src = TARGET.read_text()
    for caps, elem in (
        ("application/x-subtitle", "subparse"),
        ("application/x-subtitle-vtt", "subparse"),
        ("application/x-id3", "id3demux"),
        ("application/x-apetag", "apedemux"),
        ("audio/x-au", "auparse"),
    ):
        assert f'{{ "{caps}", "{elem}" }}' in src, caps
    fallback = re.search(r"kFallback\[\] = \{([^}]*)\}", src)
    assert fallback, "kFallback missing"
    names = re.findall(r'"(\w+)"', fallback.group(1))
    assert names == ["qtdemux", "matroskademux", "wavparse", "avidemux", "flvdemux"]


def test_regression_static_src_pad_is_drained(probe_exe):
    """wavparse/auparse/subparse push on a static "src" pad that pad-added never
    reports: unlinked, the first push returns NOT_LINKED and parsing stops."""
    r = run_probe(probe_exe, 0, None)
    assert r["static_links"] == 1, r
    assert r["leaked"] == 0, r


def test_regression_subtitle_decoder_follows_pad(probe_exe):
    """ssaparse is a Codec/Decoder: matroska's SSA pad must reach it."""
    r = run_probe(probe_exe, 1, ASS)
    assert (r["parsers"], r["links"]) == (1, 1), r


def test_regression_failed_demux_frees_caps(probe_exe):
    """Early return after typefind must release the typefind caps."""
    r = run_probe(probe_exe, 1, H264, mode=Mode.MAKE_FAILS)
    assert r["caps_live"] == 0, r


def test_adversarial_abandoned_exec_keeps_capacity(probe_exe):
    """An exec the timeout guard abandons with every parser slot taken must not
    starve the next exec of parsers."""
    r = run_probe(probe_exe, 1, H264, mode=Mode.AFTER_ABANDON)
    assert (r["parsers"], r["links"]) == (1, 1), r
    assert r["uaf"] == 0, r
