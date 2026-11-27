/* Fuzz target for GStreamer (vendored, static libs) -- demuxer layer.
 *
 * Typefinds the input, picks the matching demuxer (qtdemux, matroskademux,
 * wavparse, avidemux, flvdemux, subparse, id3demux, apedemux, auparse; byte
 * 0 % 5 over the first five when typefind finds nothing) and pushes the data
 * into it from the calling thread (no pipeline threads):
 *   src pad -> demuxer sink -> demuxer src pad -> [parser] -> drop sink
 * A parser or subtitle decoder (h264parse, aacparse, ssaparse, ...) is
 * inserted when one accepts the pad's caps, so codec configs (avcC/hvcC/esds,
 * SSA headers) are parsed, not dropped. Static "src" pads (wavparse, auparse,
 * subparse) are drained too: pad-added never reports them.
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
GST_PLUGIN_STATIC_DECLARE(subparse);
GST_PLUGIN_STATIC_DECLARE(id3demux);
GST_PLUGIN_STATIC_DECLARE(apetag);
GST_PLUGIN_STATIC_DECLARE(auparse);
GST_PLUGIN_STATIC_DECLARE(audioparsers);
GST_PLUGIN_STATIC_DECLARE(videoparsersbad);

static const struct { const char *caps; const char *elem; } kMap[] = {
    { "video/quicktime", "qtdemux" },
    { "application/x-3gp", "qtdemux" },
    { "video/x-matroska", "matroskademux" },
    { "video/webm", "matroskademux" },
    { "audio/x-wav", "wavparse" },
    { "video/x-msvideo", "avidemux" },
    { "video/x-flv", "flvdemux" },
    { "application/x-subtitle", "subparse" },
    { "application/x-subtitle-sami", "subparse" },
    { "application/x-subtitle-tmplayer", "subparse" },
    { "application/x-subtitle-mpl2", "subparse" },
    { "application/x-subtitle-dks", "subparse" },
    { "application/x-subtitle-qttext", "subparse" },
    { "application/x-subtitle-lrc", "subparse" },
    { "application/x-subtitle-vtt", "subparse" },
    { "application/x-id3", "id3demux" },
    { "application/x-apetag", "apedemux" },
    { "audio/x-au", "auparse" },
};
static const char *kFallback[] = { "qtdemux", "matroskademux", "wavparse",
                                   "avidemux", "flvdemux" };
#define FUZZ_GST_CHUNK 4096u
#define FUZZ_GST_DRAIN 3u /* empty pushes before EOS: see the push loop */
/* Per-exec cap on harness-owned objects: a 500-track file must not grow memory. */
#define FUZZ_GST_MAX_PARSERS 16u
#define FUZZ_GST_MAX_SINKS 64u

static int g_inited;
static GList *g_parsers;    /* parser + decoder factories, best rank first; built once */

/* Objects one exec creates; released after the demuxer, see release_owned(). */
static GstElement *g_parser_els[FUZZ_GST_MAX_PARSERS];
static GstPad *g_sinks[FUZZ_GST_MAX_SINKS];
static unsigned g_n_parsers, g_n_sinks;

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

/* Parentless sink that drops everything; NULL once the per-exec cap is hit. */
static GstPad *drop_sink(void) {
    if (g_n_sinks >= FUZZ_GST_MAX_SINKS) { __afl_map_edge(0x3322); return NULL; }
    GstPad *sink = gst_pad_new("sink", GST_PAD_SINK);
    gst_object_ref_sink(sink);
    g_sinks[g_n_sinks++] = sink;
    gst_pad_set_chain_function(sink, drop_chain);
    gst_pad_set_event_function(sink, drop_event);
    gst_pad_set_active(sink, TRUE);
    return sink;
}

/* Best-ranked parser accepting the pad's caps. ANY caps match every factory,
 * so they get none: e.g. video/x-h264 -> h264parse, audio/mpeg -> aacparse. */
static GstElement *parser_for(GstPad *pad) {
    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (!caps) return NULL;

    GstElement *parser = NULL;
    if (!gst_caps_is_any(caps)) {
        GList *hit = gst_element_factory_list_filter(g_parsers, caps, GST_PAD_SINK, FALSE);
        if (hit) parser = gst_element_factory_create(hit->data, NULL);
        gst_plugin_feature_list_free(hit);
    }
    gst_caps_unref(caps);
    return parser;
}

/* Running parser whose src feeds a drop sink; returns its sink pad (caller
 * unrefs) or NULL when no parser fits or a cap is hit. */
static GstPad *parser_sink(GstPad *pad) {
    if (g_n_parsers >= FUZZ_GST_MAX_PARSERS || g_n_sinks >= FUZZ_GST_MAX_SINKS) return NULL;
    GstElement *parser = parser_for(pad);
    if (!parser) return NULL;

    gst_object_ref_sink(parser);
    g_parser_els[g_n_parsers++] = parser;
    GstPad *psrc = gst_element_get_static_pad(parser, "src");
    GstPad *out = drop_sink();
    if (!psrc || !out || gst_pad_link(psrc, out) != GST_PAD_LINK_OK) __afl_map_edge(0x3331);
    if (psrc) gst_object_unref(psrc);

    gst_element_set_state(parser, GST_STATE_PLAYING);
    __afl_map_edge(0x3330);
    return gst_element_get_static_pad(parser, "sink");
}

static void on_pad_added(GstElement *el, GstPad *pad, gpointer data) {
    (void)el; (void)data;
    __afl_map_edge(0x3320);

    GstPad *psink = parser_sink(pad);
    if (psink) {
        if (gst_pad_link(pad, psink) != GST_PAD_LINK_OK) __afl_map_edge(0x3332);
        gst_object_unref(psink);
        return;
    }

    GstPad *sink = drop_sink();
    if (!sink) return;    /* over the cap: pad stays unlinked (NOT_LINKED) */
    if (gst_pad_link(pad, sink) != GST_PAD_LINK_OK) __afl_map_edge(0x3321);
}

/* After the demuxer is gone (its dispose unlinks its pads from ours): stop
 * and drop parsers, then the sinks they fed. Reverse order would leave the
 * demuxer pads peered to freed pads. */
static void release_owned(void) {
    for (unsigned i = 0; i < g_n_parsers; i++) {
        gst_element_set_state(g_parser_els[i], GST_STATE_NULL);
        gst_object_unref(g_parser_els[i]);
    }
    for (unsigned i = 0; i < g_n_sinks; i++) gst_object_unref(g_sinks[i]);
    g_n_parsers = 0;
    g_n_sinks = 0;
}

/* Instantiate once so lazy GType registration (non-reentrant g_once) never
 * runs inside a guarded exec, e.g. h264parse first met mid-campaign. */
static void prewarm_factories(void) {
    for (size_t i = 0; i < sizeof kMap / sizeof kMap[0]; i++) {
        GstElement *el = gst_element_factory_make(kMap[i].elem, NULL);
        if (el) gst_object_unref(el);
    }
    for (GList *l = g_parsers; l; l = l->next) {
        GstElement *el = gst_element_factory_create(l->data, NULL);
        if (el) gst_object_unref(el);
    }
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
    GST_PLUGIN_STATIC_REGISTER(subparse);
    GST_PLUGIN_STATIC_REGISTER(id3demux);
    GST_PLUGIN_STATIC_REGISTER(apetag);
    GST_PLUGIN_STATIC_REGISTER(auparse);
    GST_PLUGIN_STATIC_REGISTER(audioparsers);
    GST_PLUGIN_STATIC_REGISTER(videoparsersbad);

    /* Decoders too: ssaparse/subparse register as Codec/Decoder/Subtitle. The
     * build has no other decoders (they need external libs). */
    g_parsers = gst_element_factory_list_get_elements(
        GST_ELEMENT_FACTORY_TYPE_PARSER | GST_ELEMENT_FACTORY_TYPE_DECODER, GST_RANK_NONE);
    g_parsers = g_list_sort(g_parsers, gst_plugin_feature_rank_compare_func);
    prewarm_factories();
}

/* Init at dlopen so registry/plugin setup never counts against an exec timeout. */

__attribute__((visibility("default")))
int fuzz_gstreamer(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3000);
    if (size < 8) { __afl_map_edge(0x3001); return 0; }
    fuzz_gst_init();

    /* Leftovers mean the timeout guard abandoned the last exec mid-stream.
     * Its objects may sit under locks the interrupted stack held, so they
     * leak with its demuxer instead of being touched; only the slots return. */
    if (g_n_parsers || g_n_sinks) {
        __afl_map_edge(0x3007);
        g_n_parsers = 0;
        g_n_sinks = 0;
    }

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
    if (!demux) {
        __afl_map_edge(0x3003);
        if (caps) gst_caps_unref(caps);
        return 0;
    }
    g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), NULL);

    GstPad *src = gst_pad_new("src", GST_PAD_SRC);
    gst_pad_set_active(src, TRUE);
    GstPad *dsink = gst_element_get_static_pad(demux, "sink");
    if (!dsink || gst_pad_link(src, dsink) != GST_PAD_LINK_OK) {
        __afl_map_edge(0x3004);
        if (dsink) gst_object_unref(dsink);
        gst_object_unref(src); gst_object_unref(demux);
        if (caps) gst_caps_unref(caps);
        return 0;
    }

    /* Static src pad: link it now, or the first push returns NOT_LINKED. */
    GstPad *ssrc = gst_element_get_static_pad(demux, "src");
    if (ssrc) {
        __afl_map_edge(0x3008);
        on_pad_added(demux, ssrc, NULL);
        gst_object_unref(ssrc);
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

    /* Drain: avidemux's push-mode chain advances one state per call
     * (START -> HEADER -> MOVI -> data), so an input under one chunk got a
     * single call and never parsed hdrl. Empty buffers give the buffered data
     * those calls; adapter-based demuxers take them as no-ops. */
    for (unsigned i = 0; i < FUZZ_GST_DRAIN && fr == GST_FLOW_OK; i++)
        fr = gst_pad_push(src, gst_buffer_new());
    gst_pad_push_event(src, gst_event_new_eos());

    gst_element_set_state(demux, GST_STATE_NULL);
    gst_object_unref(dsink);
    gst_object_unref(src);
    gst_object_unref(demux);
    release_owned();
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
