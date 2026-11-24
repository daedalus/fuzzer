/* Fuzz target for GStreamer (vendored, static libs) -- demuxer layer.
 *
 * Typefinds the input, picks the matching demuxer (qtdemux, matroskademux,
 * wavparse, avidemux, flvdemux; byte 0 % 5 when typefind finds nothing) and
 * pushes the data into it from the calling thread (no pipeline threads):
 *   src pad -> demuxer sink; each dynamic demuxer src pad -> a drop sink.
 * Build: tools/build_targets.sh (vendored libs are instrumented there)
 */
#include <gst/gst.h>
#include <gst/base/gsttypefindhelper.h>
#include <string.h>

extern void __afl_map_edge(unsigned int cur_loc);

GST_PLUGIN_STATIC_DECLARE(typefindfunctions);
GST_PLUGIN_STATIC_DECLARE(isomp4);
GST_PLUGIN_STATIC_DECLARE(matroska);
GST_PLUGIN_STATIC_DECLARE(wavparse);
GST_PLUGIN_STATIC_DECLARE(avi);
GST_PLUGIN_STATIC_DECLARE(flv);

static const struct { const char *caps; const char *elem; } kMap[] = {
    { "video/quicktime", "qtdemux" },
    { "application/x-3gp", "qtdemux" },
    { "video/x-matroska", "matroskademux" },
    { "video/webm", "matroskademux" },
    { "audio/x-wav", "wavparse" },
    { "video/x-msvideo", "avidemux" },
    { "video/x-flv", "flvdemux" },
};
static const char *kFallback[] = { "qtdemux", "matroskademux", "wavparse",
                                   "avidemux", "flvdemux" };
#define FUZZ_GST_CHUNK 4096u

static int g_inited;

/* GLib log handler that swallows GStreamer/plugin g_warning()/g_critical()/
 * g_message() noise (e.g. "(python3:PID): GStreamer-WARNING **: ..." and
 * "** (python3:PID): CRITICAL **: file ../gst-plugins-good/...: assertion
 * ... failed"). Plugins built without G_LOG_DOMAIN log under the NULL
 * domain, so a per-domain handler alone misses them: also replace the
 * default handler. G_LOG_LEVEL_ERROR is never dropped (it aborts anyway).
 * Set FUZZ_GST_VERBOSE=1 to keep all output. */
static void drop_gst_log(const gchar *domain, GLogLevelFlags level,
                         const gchar *msg, gpointer data) {
    (void)domain; (void)level; (void)msg; (void)data;
}

static GstFlowReturn drop_chain(GstPad *pad, GstObject *parent, GstBuffer *buf) {
    (void)pad; (void)parent;
    __afl_map_edge(0x3300);
    gst_buffer_unref(buf);
    return GST_FLOW_OK;
}

static gboolean drop_event(GstPad *pad, GstObject *parent, GstEvent *ev) {
    (void)pad; (void)parent;
    __afl_map_edge(0x3310 + (GST_EVENT_TYPE(ev) & 0x3F));
    gst_event_unref(ev);
    return TRUE;
}

static void on_pad_added(GstElement *el, GstPad *pad, gpointer data) {
    (void)el; (void)data;
    __afl_map_edge(0x3320);
    GstPad *sink = gst_pad_new("sink", GST_PAD_SINK);
    gst_pad_set_chain_function(sink, drop_chain);
    gst_pad_set_event_function(sink, drop_event);
    gst_pad_set_active(sink, TRUE);
    if (gst_pad_link(pad, sink) != GST_PAD_LINK_OK) __afl_map_edge(0x3321);
    /* sink is owned by the link/pad tree and is unreffed when the demuxer dies */
}

static void fuzz_gst_init(void) {
    if (g_inited) return;
    g_inited = 1;
    g_setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "", TRUE);
    g_setenv("GST_PLUGIN_PATH_1_0", "", TRUE);
    g_setenv("GST_REGISTRY_FORK", "no", TRUE);
    g_setenv("GST_DEBUG", "0", TRUE);
    if (!g_getenv("FUZZ_GST_VERBOSE")) {
        const GLogLevelFlags quiet = (GLogLevelFlags)(
            G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING | G_LOG_LEVEL_MESSAGE |
            G_LOG_LEVEL_INFO | G_LOG_LEVEL_DEBUG);
        g_log_set_default_handler(drop_gst_log, NULL);       /* any domain */
        g_log_set_handler(NULL, quiet, drop_gst_log, NULL);  /* NULL domain */
        g_log_set_handler("GStreamer", quiet, drop_gst_log, NULL);
    }
    gst_init(NULL, NULL);
    GST_PLUGIN_STATIC_REGISTER(typefindfunctions);
    GST_PLUGIN_STATIC_REGISTER(isomp4);
    GST_PLUGIN_STATIC_REGISTER(matroska);
    GST_PLUGIN_STATIC_REGISTER(wavparse);
    GST_PLUGIN_STATIC_REGISTER(avi);
    GST_PLUGIN_STATIC_REGISTER(flv);
}

/* Init at dlopen so registry/plugin setup never counts against an exec timeout. */

__attribute__((visibility("default")))
int fuzz_gstreamer(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3000);
    if (size < 8) { __afl_map_edge(0x3001); return 0; }
    fuzz_gst_init();

    const char *elem = NULL;
    GstTypeFindProbability prob = GST_TYPE_FIND_NONE;
    GstCaps *caps = gst_type_find_helper_for_data(NULL, buf, size, &prob);
    if (caps) {
        const char *name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
        for (size_t i = 0; i < sizeof kMap / sizeof kMap[0]; i++)
            if (strcmp(name, kMap[i].caps) == 0) { elem = kMap[i].elem; __afl_map_edge(0x3010 + i); break; }
    }
    if (!elem) {
        __afl_map_edge(0x3002);
        elem = kFallback[buf[0] % 5];
        /* No typefind match (or a mismatch between kMap and kFallback's chosen
         * element): fall back to the caps string this element is known for, so
         * a CAPS event can still be sent below. */
        if (caps) { gst_caps_unref(caps); caps = NULL; }
        for (size_t i = 0; i < sizeof kMap / sizeof kMap[0]; i++)
            if (strcmp(elem, kMap[i].elem) == 0) { caps = gst_caps_from_string(kMap[i].caps); break; }
    }

    GstElement *demux = gst_element_factory_make(elem, NULL);
    if (!demux) { __afl_map_edge(0x3003); return 0; }
    g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), NULL);

    GstPad *src = gst_pad_new("src", GST_PAD_SRC);
    gst_pad_set_active(src, TRUE);
    GstPad *dsink = gst_element_get_static_pad(demux, "sink");
    if (!dsink || gst_pad_link(src, dsink) != GST_PAD_LINK_OK) {
        __afl_map_edge(0x3004);
        if (dsink) gst_object_unref(dsink);
        gst_object_unref(src); gst_object_unref(demux);
        return 0;
    }
    if (gst_element_set_state(demux, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
        __afl_map_edge(0x3005);

    GstSegment seg;
    gst_segment_init(&seg, GST_FORMAT_BYTES);
    gst_pad_push_event(src, gst_event_new_stream_start("fuzz"));
    /* Sticky-event order matters: STREAM_START, CAPS, SEGMENT, then buffers.
     * Without a CAPS event the demuxer's chain function returns
     * GST_FLOW_NOT_NEGOTIATED for every push regardless of payload, so the
     * demuxer's own parsing code (the real coverage target) is never
     * reached and every exec converges on the same handful of edges. */
    if (caps) {
        gst_pad_push_event(src, gst_event_new_caps(caps));
        gst_caps_unref(caps);
        caps = NULL;
    } else {
        __afl_map_edge(0x3006);
    }
    gst_pad_push_event(src, gst_event_new_segment(&seg));

    GstFlowReturn fr = GST_FLOW_OK;
    for (size_t off = 0; off < size && fr == GST_FLOW_OK; off += FUZZ_GST_CHUNK) {
        size_t n = size - off < FUZZ_GST_CHUNK ? size - off : FUZZ_GST_CHUNK;
        GstBuffer *b = gst_buffer_new_memdup(buf + off, n);
        GST_BUFFER_OFFSET(b) = off;
        fr = gst_pad_push(src, b);
        __afl_map_edge(0x3100 + (fr & 0x0F));
    }
    gst_pad_push_event(src, gst_event_new_eos());

    gst_element_set_state(demux, GST_STATE_NULL);
    gst_object_unref(dsink);
    gst_object_unref(src);
    gst_object_unref(demux);
    return 0;
}

/* Pre-warm: GType registration is lazy and non-reentrant; an exec interrupted by
 * the fuzzer's timeout guard mid-registration leaves g_once_init stuck. Force
 * every demuxer type (and the typefind path) to register before any guarded exec. */
__attribute__((constructor)) static void fuzz_gst_ctor(void) {
    fuzz_gst_init();
    unsigned char b[64];
    for (unsigned i = 0; i < 5; i++) {
        memset(b, 0, sizeof b);
        b[0] = (unsigned char)i;
        fuzz_gstreamer(b, sizeof b);
    }
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_gstreamer(buf, size);
}
