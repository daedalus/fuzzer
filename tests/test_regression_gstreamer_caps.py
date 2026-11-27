"""The GStreamer target must negotiate CAPS before pushing buffers.

Without a CAPS event the demuxer's chain function returns
``GST_FLOW_NOT_NEGOTIATED`` for every buffer. The campaign still reports a
handful of wrapper edges, but none of the demuxer's parsing code runs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "gstreamer_read.c"

GST_H = r"""
#ifndef STUB_GST_H
#define STUB_GST_H
#include <stddef.h>
typedef int gboolean;
typedef char gchar;
typedef void *gpointer;
typedef const void *gconstpointer;
typedef unsigned char guint8;
typedef size_t gsize;
#define TRUE 1
#define FALSE 0
#define G_CALLBACK(f) ((GCallback)(f))
typedef void (*GCallback)(void);
typedef void (*GLogFunc)(const gchar *, int, const gchar *, gpointer);
#define G_LOG_LEVEL_CRITICAL 1
#define G_LOG_LEVEL_WARNING 2
#define G_LOG_LEVEL_MESSAGE 4
#define G_LOG_LEVEL_INFO 8
#define G_LOG_LEVEL_DEBUG 16
typedef int GLogLevelFlags;
/* One object type for elements and pads, so stubs can model refcounts and links. */
typedef struct StubObj {
    int refs, floating, kind, state, parser;
    void *caps;
    struct StubObj *peer, *parent, *sinkpad, *srcpad;
} StubObj;
typedef StubObj GstObject;
typedef StubObj GstElement;
typedef StubObj GstPad;
typedef struct { int dummy; } GstSegment;
typedef struct { int type; size_t offset; } GstBuffer;
typedef struct { int type; } GstEvent;
typedef struct { int refs; int any; const char *name; int tracked; } GstCaps;
typedef struct { int dummy; } GstStructure;
typedef struct GList { void *data; struct GList *next; } GList;
typedef struct { const char *name; } GstElementFactory;
typedef int (*GCompareFunc)(const void *, const void *);
typedef enum { GST_RANK_NONE = 0 } GstRank;
typedef unsigned long long GstElementFactoryListType;
#define GST_ELEMENT_FACTORY_TYPE_PARSER (1ULL << 6)
typedef enum { GST_TYPE_FIND_NONE = 0 } GstTypeFindProbability;
typedef enum { GST_FLOW_OK = 0, GST_FLOW_NOT_NEGOTIATED = -4 } GstFlowReturn;
typedef enum { GST_STATE_NULL = 1, GST_STATE_PLAYING = 4 } GstState;
typedef enum { GST_STATE_CHANGE_FAILURE = -1 } GstStateChangeReturn;
typedef enum { GST_PAD_LINK_OK = 0 } GstPadLinkReturn;
typedef enum { GST_PAD_SINK = 1, GST_PAD_SRC = 2 } GstPadDirection;
typedef enum { GST_FORMAT_BYTES = 2 } GstFormat;
typedef enum { EVENT_UNKNOWN = 0, EVENT_STREAM_START = 1, EVENT_CAPS = 2,
               EVENT_SEGMENT = 3, EVENT_EOS = 4 } EventType;
#define GST_EVENT_TYPE(ev) ((int)(ev)->type)
#define GST_BUFFER_OFFSET(buf) ((buf)->offset)
#define GST_PLUGIN_STATIC_DECLARE(name)
#define GST_PLUGIN_STATIC_REGISTER(name)
void g_setenv(const gchar *, const gchar *, gboolean);
const gchar *g_getenv(const gchar *);
void g_log_set_default_handler(GLogFunc, gpointer);
void g_log_set_handler(const gchar *, GLogLevelFlags, GLogFunc, gpointer);
void gst_init(int *, char ***);
GstCaps *gst_type_find_helper_for_data(GstObject *, const guint8 *, gsize,
                                       GstTypeFindProbability *);
GstStructure *gst_caps_get_structure(const GstCaps *, guint8);
const gchar *gst_structure_get_name(const GstStructure *);
GstCaps *gst_caps_from_string(const gchar *);
void gst_caps_unref(GstCaps *);
GstElement *gst_element_factory_make(const gchar *, const gchar *);
void g_signal_connect(gpointer, const gchar *, GCallback, gpointer);
GstPad *gst_pad_new(const gchar *, GstPadDirection);
GstPadDirection gst_pad_get_direction(const GstPad *);
void gst_pad_set_chain_function_full(GstPad *, GstFlowReturn (*)(GstPad *, GstObject *, GstBuffer *), gpointer);
void gst_pad_set_event_function_full(GstPad *, gboolean (*)(GstPad *, GstObject *, GstEvent *), gpointer);
#define gst_pad_set_chain_function(pad, fn) gst_pad_set_chain_function_full(pad, fn, NULL)
#define gst_pad_set_event_function(pad, fn) gst_pad_set_event_function_full(pad, fn, NULL)
void gst_pad_set_active(GstPad *, gboolean);
GstPad *gst_element_get_static_pad(GstElement *, const gchar *);
GstPadLinkReturn gst_pad_link(GstPad *, GstPad *);
GstStateChangeReturn gst_element_set_state(GstElement *, GstState);
void gst_segment_init(GstSegment *, GstFormat);
gboolean gst_pad_push_event(GstPad *, GstEvent *);
GstFlowReturn gst_pad_push(GstPad *, GstBuffer *);
GstEvent *gst_event_new_stream_start(const gchar *);
GstEvent *gst_event_new_segment(const GstSegment *);
GstEvent *gst_event_new_caps(GstCaps *);
GstEvent *gst_event_new_eos(void);
void gst_event_unref(GstEvent *);
GstBuffer *gst_buffer_new_memdup(gconstpointer, gsize);
void gst_buffer_unref(GstBuffer *);
void g_object_unref(gpointer);
#define gst_object_unref(obj) g_object_unref(obj)
gpointer gst_object_ref_sink(gpointer);
GList *gst_element_factory_list_get_elements(GstElementFactoryListType, GstRank);
GList *gst_element_factory_list_filter(GList *, const GstCaps *, GstPadDirection, gboolean);
void gst_plugin_feature_list_free(GList *);
GList *g_list_sort(GList *, GCompareFunc);
int gst_plugin_feature_rank_compare_func(const void *, const void *);
GstElement *gst_element_factory_create(GstElementFactory *, const gchar *);
GstCaps *gst_pad_get_current_caps(GstPad *);
gboolean gst_caps_is_any(const GstCaps *);
#endif
"""

TYPEFIND_H = r"""
#ifndef STUB_TYPEFIND_H
#define STUB_TYPEFIND_H
#include <gst/gst.h>
#endif
"""

HARNESS = r"""
#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fuzz_gstreamer(const unsigned char *, size_t);

static char events[128];
static int caps_refs = 1;
static int caps_underflow;
static int last_push = 99;
static GstCaps the_caps;

void __afl_map_edge(unsigned int id) { (void)id; }
void g_setenv(const gchar *a, const gchar *b, gboolean c) { (void)a; (void)b; (void)c; }
const gchar *g_getenv(const gchar *a) { (void)a; return NULL; }
void g_log_set_default_handler(GLogFunc f, gpointer d) { (void)f; (void)d; }
void g_log_set_handler(const gchar *a, GLogLevelFlags b, GLogFunc f, gpointer d)
{ (void)a; (void)b; (void)f; (void)d; }
void gst_init(int *a, char ***b) { (void)a; (void)b; }
GstCaps *gst_type_find_helper_for_data(GstObject *o, const guint8 *d, gsize n,
                                       GstTypeFindProbability *p)
{ (void)o; (void)d; (void)n; *p = GST_TYPE_FIND_NONE; return &the_caps; }
GstStructure *gst_caps_get_structure(const GstCaps *c, guint8 i)
{ (void)c; (void)i; static GstStructure s; return &s; }
const gchar *gst_structure_get_name(const GstStructure *s) { (void)s; return "video/quicktime"; }
GstCaps *gst_caps_from_string(const gchar *s) { (void)s; return &the_caps; }
void gst_caps_unref(GstCaps *c) { (void)c; if (--caps_refs < 0) caps_underflow = 1; }
GstElement *gst_element_factory_make(const gchar *a, const gchar *b) { (void)a; (void)b; static GstElement e; return &e; }
void g_signal_connect(gpointer a, const gchar *b, GCallback c, gpointer d) { (void)a; (void)b; (void)c; (void)d; }
GstPad *gst_pad_new(const gchar *a, GstPadDirection d) { (void)a; static GstPad p; (void)d; return &p; }
GstPadDirection gst_pad_get_direction(const GstPad *p) { (void)p; return GST_PAD_SRC; }
void gst_pad_set_chain_function_full(GstPad *p, GstFlowReturn (*f)(GstPad *, GstObject *, GstBuffer *), gpointer d)
{ (void)p; (void)f; (void)d; }
void gst_pad_set_event_function_full(GstPad *p, gboolean (*f)(GstPad *, GstObject *, GstEvent *), gpointer d)
{ (void)p; (void)f; (void)d; }
void gst_pad_set_active(GstPad *p, gboolean a) { (void)p; (void)a; }
GstPad *gst_element_get_static_pad(GstElement *e, const gchar *n) { (void)e; (void)n; static GstPad p; return &p; }
GstPadLinkReturn gst_pad_link(GstPad *a, GstPad *b) { (void)a; (void)b; return GST_PAD_LINK_OK; }
GstStateChangeReturn gst_element_set_state(GstElement *e, GstState s) { (void)e; (void)s; return 0; }
void gst_segment_init(GstSegment *s, GstFormat f) { (void)s; (void)f; }
gboolean gst_pad_push_event(GstPad *p, GstEvent *e)
{
    (void)p;
    size_t n = strlen(events);
    if (n < sizeof(events) - 2) { events[n] = (char)('0' + GST_EVENT_TYPE(e)); events[n + 1] = '\0'; }
    gst_event_unref(e);
    return TRUE;
}
static GstEvent *new_event(int type) { GstEvent *e = malloc(sizeof *e); e->type = type; return e; }
GstEvent *gst_event_new_stream_start(const gchar *id) { (void)id; return new_event(EVENT_STREAM_START); }
GstEvent *gst_event_new_segment(const GstSegment *s) { (void)s; return new_event(EVENT_SEGMENT); }
GstEvent *gst_event_new_caps(GstCaps *c) { (void)c; return new_event(EVENT_CAPS); }
GstEvent *gst_event_new_eos(void) { return new_event(EVENT_EOS); }
void gst_event_unref(GstEvent *e) { free(e); }
GstBuffer *gst_buffer_new_memdup(gconstpointer d, gsize n) { (void)d; (void)n; GstBuffer *b = malloc(sizeof *b); b->type = 0; b->offset = 0; return b; }
void gst_buffer_unref(GstBuffer *b) { free(b); }
void g_object_unref(gpointer p) { (void)p; }
gpointer gst_object_ref_sink(gpointer p) { return p; }
GList *gst_element_factory_list_get_elements(GstElementFactoryListType t, GstRank r)
{ (void)t; (void)r; return NULL; }
GList *gst_element_factory_list_filter(GList *l, const GstCaps *c, GstPadDirection d, gboolean s)
{ (void)l; (void)c; (void)d; (void)s; return NULL; }
void gst_plugin_feature_list_free(GList *l) { (void)l; }
GList *g_list_sort(GList *l, GCompareFunc f) { (void)f; return l; }
int gst_plugin_feature_rank_compare_func(const void *a, const void *b) { (void)a; (void)b; return 0; }
GstElement *gst_element_factory_create(GstElementFactory *f, const gchar *n)
{ (void)f; (void)n; return NULL; }
GstCaps *gst_pad_get_current_caps(GstPad *p) { (void)p; return NULL; }
gboolean gst_caps_is_any(const GstCaps *c) { (void)c; return FALSE; }

/* No CAPS event means no negotiation: model the demuxer's chain function. */
GstFlowReturn gst_pad_push(GstPad *p, GstBuffer *b)
{
    (void)p;
    last_push = strchr(events, '2') ? GST_FLOW_OK : GST_FLOW_NOT_NEGOTIATED;
    gst_buffer_unref(b);
    return last_push;
}

void reset_probe(void) { events[0] = '\0'; caps_refs = 1; caps_underflow = 0; last_push = 99; }
const char *event_log(void) { return events; }
int caps_refcount(void) { return caps_refs; }
int caps_underflowed(void) { return caps_underflow; }
int push_result(void) { return last_push; }

int main(void)
{
    unsigned char buf[64] = {0};
    reset_probe();
    fuzz_gstreamer(buf, sizeof buf);
    printf("events=%s refs=%d underflow=%d push=%d\n", event_log(), caps_refcount(),
           caps_underflowed(), push_result());
    return 0;
}
"""

HARNESS_DECLS = (
    "void reset_probe(void);\n"
    "const char *event_log(void);\n"
    "int caps_refcount(void);\n"
    "int caps_underflowed(void);\n"
    "int push_result(void);\n"
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """Compile the real target against GStreamer stubs; return its probe output."""
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("gst_stub")
    (d / "gst").mkdir()
    (d / "gst" / "base").mkdir()
    (d / "gst" / "gst.h").write_text(GST_H)
    (d / "gst" / "base" / "gsttypefindhelper.h").write_text(TYPEFIND_H)
    (d / "harness.c").write_text(HARNESS + HARNESS_DECLS)
    exe = d / "probe"
    cmd = [
        clang,
        "-std=c11",
        "-O0",
        "-g",
        "-I",
        str(d),
        "-o",
        str(exe),
        str(d / "harness.c"),
        str(TARGET),
    ]
    compiled = subprocess.run(cmd, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.strip()
    return dict(part.split("=", 1) for part in out.split())


def test_caps_event_precedes_buffers(probe):
    """Regression: no CAPS event => NOT_NEGOTIATED on every push."""
    assert probe["events"] == "1234", probe
    assert probe["push"] == "0", probe


def test_caps_released_exactly_once(probe):
    assert probe["refs"] == "0", probe
    assert probe["underflow"] == "0", probe
