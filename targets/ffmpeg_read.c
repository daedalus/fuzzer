/* Fuzz target for FFmpeg — exercises the full demux→decode pipeline.
 *
 * Feeds arbitrary data through:
 *   avformat_open_input → avformat_find_stream_info → av_read_frame
 *   → avcodec_send_packet → avcodec_receive_frame → close
 *
 * Covers all registered demuxers, decoders, parsers, and BSFs.
 * Core library paths (buffer mgmt, packet lifecycle, error recovery)
 * account for ~80% of executable code; per-format leaf functions ~20%.
 *
 * Compile standalone:
 *   gcc -O2 -g -fsanitize=address -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/ffmpeg_read targets/ffmpeg_read.c -lavformat -lavcodec -lavutil -lswresample -lm
 *
 * Compile shared library (for inprocess modes):
 *   gcc -O2 -g -shared -fPIC -fsanitize-coverage=trace-cmp \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       src/fuzzer_tool/adapters/cmplog_shim.c -ldl \
 *       -o targets/ffmpeg_read.so targets/ffmpeg_read.c \
 *       -lavformat -lavcodec -lavutil -lswresample -lm -Wl,--export-dynamic
 */
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/mem.h>
#include <malloc.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* I/O buffer size for AVIOContext (256KB) */
#ifndef FFMPEG_IO_BUF_SIZE
#define FFMPEG_IO_BUF_SIZE (256 * 1024)
#endif

/* ── In-memory I/O for avformat ──────────────────────────────────── */

typedef struct {
    const unsigned char *data;
    size_t size;
    size_t offset;
} FuzzIOState;

static int fuzz_read_packet(void *opaque, unsigned char *buf, int buf_size) {
    FuzzIOState *st = (FuzzIOState *)opaque;
    int avail = (int)(st->size - st->offset);
    if (avail <= 0) return AVERROR_EOF;
    if (buf_size > avail) buf_size = avail;
    memcpy(buf, st->data + st->offset, buf_size);
    st->offset += buf_size;
    return buf_size;
}

/* ── Persistent decoder pool ──────────────────────────────────────── */

#define FUZZ_MAX_DECODERS 128

typedef struct {
    AVCodecContext *ctx;
    int              codec_id;
    int              codec_type;   /* AVMediaType: picks the decode API */
    int              inited;
} DecoderSlot;

static AVPacket    *g_pkt          = NULL;
static AVFrame     *g_frame        = NULL;
/* Lives for the whole run, not just fuzz_open_input(): avio_alloc_context
 * stores this pointer as the AVIOContext's `opaque` and fuzz_read_packet
 * dereferences it on every read issued by avformat_find_stream_info() and
 * av_read_frame(), both of which run *after* fuzz_open_input() returns.
 * Freed by fuzz_release_io_state() once the format context is closed. */
static FuzzIOState *g_io_state     = NULL;
/* Our AVIOContext, when ffmpeg kept it as fmt_ctx->pb. Setting pb before
 * avformat_open_input makes ffmpeg set AVFMT_FLAG_CUSTOM_IO, which means
 * avformat_close_input will NOT free it -- we own it and must. Released by
 * fuzz_release_io_state() after the format context is closed. */
static AVIOContext *g_avio_ctx     = NULL;
static int          fuzz_network_inited = 0;
static int          fuzz_reset_counter = 0;
static DecoderSlot  g_decs[FUZZ_MAX_DECODERS];

/* Release the run-scoped AVIO opaque. Safe to call more than once, and safe
 * to call before the first open. Must not run until the AVFormatContext that
 * reads through it has been closed. */
static void fuzz_release_io_state(void) {
    if (g_avio_ctx) {
        /* avio_context_free frees the context but NOT its buffer, so the
         * buffer goes first. Read it from the context rather than caching the
         * original pointer: ffmpeg may have reallocated it during probing. */
        av_freep(&g_avio_ctx->buffer);
        avio_context_free(&g_avio_ctx);
        g_avio_ctx = NULL;
    }
    if (g_io_state) {
        av_free(g_io_state);
        g_io_state = NULL;
    }
}

static void fuzz_cleanup_decoders(void) {
    for (int i = 0; i < FUZZ_MAX_DECODERS; i++) {
        if (g_decs[i].ctx) {
            avcodec_free_context(&g_decs[i].ctx);
            g_decs[i].ctx      = NULL;
            g_decs[i].codec_id = 0;
            g_decs[i].inited   = 0;
        }
    }
}

static int fuzz_ensure_initialized(void) {
    if (!g_pkt) {
        g_pkt = av_packet_alloc();
        if (!g_pkt) return 0;
    }
    if (!g_frame) {
        g_frame = av_frame_alloc();
        if (!g_frame) return 0;
    }
    if (!fuzz_network_inited) {
        avformat_network_init();
        fuzz_network_inited = 1;
    }
    return 1;
}

/* Open an input from an in-memory buffer. Each call gets its own
 * AVIOContext buffer because ffmpeg's probe path may free the buffer
 * during avformat_open_input. */
static AVFormatContext *fuzz_open_input(const unsigned char *buf, size_t size) {
    /* Reclaim the previous run's state before overwriting the slot. */
    fuzz_release_io_state();

    FuzzIOState *io_state = av_malloc(sizeof(*io_state));
    if (!io_state) return NULL;
    io_state->data = buf;
    io_state->size = size;
    io_state->offset = 0;
    g_io_state = io_state;

    unsigned char *io_buffer = av_malloc(FFMPEG_IO_BUF_SIZE);
    if (!io_buffer) { fuzz_release_io_state(); return NULL; }

    AVIOContext *avio_ctx = avio_alloc_context(
        io_buffer, FFMPEG_IO_BUF_SIZE, 0,
        io_state, fuzz_read_packet, NULL, NULL);
    if (!avio_ctx) {
        av_free(io_buffer);
        fuzz_release_io_state();
        return NULL;
    }
    /* Take ownership here, not on the success path. Every exit below runs
     * through fuzz_release_io_state(), so claiming it now is what makes the
     * *failure* paths release it too -- and under a fuzzer avformat_open_input
     * failing is the common case, not the rare one. */
    g_avio_ctx = avio_ctx;

    AVFormatContext *fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) {
        fuzz_release_io_state();
        return NULL;
    }
    fmt_ctx->max_analyze_duration = 300000; /* 300 ms probe cap */
    fmt_ctx->probesize = 1 * 1024 * 1024;  /* 1 MB probe window */
    fmt_ctx->pb = avio_ctx;

    if (avformat_open_input(&fmt_ctx, NULL, NULL, NULL) < 0) {
        avformat_close_input(&fmt_ctx);
        /* Nothing reads through the opaque any more — safe to reclaim here. */
        fuzz_release_io_state();
        return NULL;
    }

    if (fmt_ctx->pb != avio_ctx) {
        /* ffmpeg swapped in its own pb (ffio_rewind_with_probe_data) and owns
         * whatever it built; ours is unreferenced. Disown before freeing so
         * fuzz_release_io_state() cannot double-free it, and free only the
         * struct -- the probe path may already have taken or released
         * io_buffer, so a double free is worse than leaking on this branch. */
        g_avio_ctx = NULL;
        av_free(avio_ctx);
    }
    /* Otherwise g_avio_ctx still holds it: AVFMT_FLAG_CUSTOM_IO means
     * avformat_close_input leaves pb alone, so the context and its 256 KB
     * buffer are ours to release -- but only after the format context is
     * closed, since it reads through them. */
    /* g_io_state stays live: fmt_ctx still reads through it in
     * avformat_find_stream_info() and av_read_frame(). The caller releases
     * it after avformat_close_input(). */
    return fmt_ctx;
}

/* ── Per-iteration phase timers ──────────────────────────────────── */

typedef struct {
    struct timespec t_open;
    struct timespec t_stream_info;
    struct timespec t_decode;
    struct timespec t_cleanup;
} FuzzPhaseTimers;

static inline void fuzz_ts_now(struct timespec *ts) {
    clock_gettime(CLOCK_MONOTONIC, ts);
}

static inline long long fuzz_ts_ns(const struct timespec *start,
                                    const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000000LL +
           (end->tv_nsec - start->tv_nsec);
}

static FuzzPhaseTimers fuzz_phase_timers;
static int fuzz_phase_stats_initialized = 0;
static int fuzz_phase_profile_enabled = 0;

/* All diagnostic output is behind this. It used to be unconditional: six
 * stderr lines and 226 bytes per execution, each fflush'd, in the same loop
 * the Python side was optimizing at microsecond scale. */
#define FUZZ_LOG(...)                                   \
    do {                                                \
        if (fuzz_phase_profile_enabled) {               \
            fprintf(stderr, __VA_ARGS__);               \
            fflush(stderr);                             \
        }                                               \
    } while (0)

static void fuzz_report_phase(const char *phase, long long ns) {
    FUZZ_LOG("[ffmpeg phase] %s %.2f ms\n", phase, ns / 1e6);
}

/* ── Watchdog ─────────────────────────────────────────────────────────
 * One process-wide POSIX timer, armed and disarmed per phase, replacing a
 * pthread_create/cancel/join per phase (two phases per execution, so six
 * thread-lifecycle operations plus two stack mmaps on every input).
 *
 * SIGRTMIN rather than SIGALRM, and timer_create rather than setitimer,
 * because InProcessRunner enforces *its own* timeout with SIGALRM +
 * ITIMER_REAL (inprocess.py:465-472, 545-553). There is one ITIMER_REAL and
 * one SIGALRM disposition per process: arming either here would silently
 * cancel the fuzzer's timeout and replace its handler. POSIX per-process
 * timers are independent of both.
 *
 * Disabled entirely when something else in the process is already managing
 * timeouts -- see fuzz_watchdog_init. That is the in-process case, where
 * _exit(124) would kill the fuzzer itself rather than the test case.        */

#define FUZZ_WATCHDOG_SIG (SIGRTMIN + 3)

static timer_t g_watchdog_timer;
static _Atomic int g_watchdog_enabled = 0;  /* read from the signal handler */
static int g_watchdog_ready = 0;
static long long g_outer_deadline_ns = 0;
static long g_watchdog_budget_ms = 900;  /* per-execution outer budget */

static long long fuzz_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Async-signal-safe: write(2) and _exit(2) only. The old handler called
 * fprintf/fflush from signal context, which is not. */
static void fuzz_watchdog_fire(int sig) {
    (void)sig;
    static const char msg[] = "[ffmpeg watchdog] timeout, exiting\n";
    ssize_t r = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)r;
    _exit(124);
}

static void fuzz_watchdog_init(void) {
    /* FFMPEG_WATCHDOG_MS sets the per-execution budget; 0 disables entirely.
     * Also the hook the tests use to make the watchdog observable. */
    const char *env = getenv("FFMPEG_WATCHDOG_MS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v <= 0) return;
        g_watchdog_budget_ms = v;
    }

    /* Stand down if the host process already installed a SIGALRM handler.
     * That is InProcessRunner, which times the target out itself and can
     * recover; killing the process from here would take the whole campaign
     * down over a single slow input. A forkserver/subprocess child has no
     * such handler and does need the watchdog, since nothing inside it can
     * escape a hung demuxer. */
    struct sigaction cur;
    if (sigaction(SIGALRM, NULL, &cur) == 0 &&
        cur.sa_handler != SIG_DFL && cur.sa_handler != SIG_IGN) {
        return;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = fuzz_watchdog_fire;
    sigemptyset(&sa.sa_mask);
    if (sigaction(FUZZ_WATCHDOG_SIG, &sa, NULL) != 0) return;

    struct sigevent sev;
    memset(&sev, 0, sizeof sev);
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = FUZZ_WATCHDOG_SIG;
    if (timer_create(CLOCK_MONOTONIC, &sev, &g_watchdog_timer) != 0) return;

    g_watchdog_ready = 1;
    g_watchdog_enabled = 1;
}

/* Two syscalls per execution (arm + disarm) in place of the thread churn. */
static void fuzz_watchdog_arm_ns(long long ns) {
    if (!g_watchdog_ready || !g_watchdog_enabled) return;
    if (ns < 1000LL) ns = 1000LL;  /* 1 us floor: an all-zero it_value disarms */
    struct itimerspec its;
    memset(&its, 0, sizeof its);
    its.it_value.tv_sec = ns / 1000000000LL;
    its.it_value.tv_nsec = ns % 1000000000LL;
    timer_settime(g_watchdog_timer, 0, &its, NULL);
}

static void fuzz_watchdog_disarm(void) {
    if (!g_watchdog_ready) return;
    struct itimerspec its;
    memset(&its, 0, sizeof its);
    timer_settime(g_watchdog_timer, 0, &its, NULL);
}

/* Open the outer budget for one execution. */
static void fuzz_start_watchdog(long timeout_ms) {
    g_outer_deadline_ns = fuzz_now_ns() + timeout_ms * 1000000LL;
    fuzz_watchdog_arm_ns(timeout_ms * 1000000LL);
}

static void fuzz_cancel_watchdog(void) {
    g_outer_deadline_ns = 0;
    fuzz_watchdog_disarm();
}

/* Run *fn* under a tighter inner budget, then restore whatever remains of
 * the outer one. The previous version ran a second concurrent thread for
 * this; a single timer cannot hold two deadlines, so the outer deadline is
 * tracked as a timestamp and re-armed against the clock afterwards. Net
 * semantics are unchanged: the inner phase is bounded by *timeout_ms* and
 * the execution as a whole by the outer budget measured from its start. */
static int fuzz_with_watchdog(long timeout_ms, int (*fn)(void *), void *arg) {
    long long inner_ns = timeout_ms * 1000000LL;
    long long remaining = g_outer_deadline_ns ? g_outer_deadline_ns - fuzz_now_ns() : 0;
    if (remaining > 0 && remaining < inner_ns) inner_ns = remaining;

    fuzz_watchdog_arm_ns(inner_ns);
    int rc = fn(arg);

    if (g_outer_deadline_ns) {
        long long left = g_outer_deadline_ns - fuzz_now_ns();
        if (left <= 0) left = 1000LL;
        fuzz_watchdog_arm_ns(left);
    } else {
        fuzz_watchdog_disarm();
    }
    return rc;
}

static int fuzz_run_find_stream_info(void *arg) {
    AVFormatContext **p = (AVFormatContext **)arg;
    return avformat_find_stream_info(p[0], NULL);
}

static void fuzz_init_phase_stats(void) {
    if (fuzz_phase_stats_initialized) return;
    fuzz_phase_stats_initialized = 1;
    fuzz_phase_profile_enabled = getenv("FFMPEG_PROFILE") != NULL;
    fuzz_watchdog_init();
}

#define FUZZ_PHASE_START(ts) fuzz_ts_now(&(ts))
#define FUZZ_PHASE_END(ts, accum) \
    do { \
        struct timespec _end; \
        fuzz_ts_now(&_end); \
        (accum) += fuzz_ts_ns(&(ts), &_end); \
    } while (0)

/* Forward declarations for functions defined after their callers. */
void fuzz_write_profile(void);
void fuzz_print_phase_stats(void);
static int fuzz_run_find_stream_info(void *arg);

/* Global stats accumulated across all calls. */
static struct {
    int calls;
    long long t_open_ns;
    long long t_stream_info_ns;
    long long t_decode_ns;
    long long t_cleanup_ns;
    int avio_alloc;
    int decoder_opens;
    int decoder_reuses;
} fuzz_phase_stats;

/* ── Core fuzz function ──────────────────────────────────────────── */

__attribute__((visibility("default")))
int fuzz_ffmpeg(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 16) { __afl_map_edge(0x1001); return 0; }

    /* Silence FFmpeg's internal logging */
    av_log_set_level(AV_LOG_QUIET);

    fuzz_init_phase_stats();

    fuzz_start_watchdog(g_watchdog_budget_ms);
    if (!fuzz_ensure_initialized()) {
        __afl_map_edge(0x1003);
        fuzz_cancel_watchdog();
        return 0;
    }

    /* Tear down any previous input; decoders are freed below after
     * we know the new stream layout so cleanup stays O(nb_streams). */
    av_packet_unref(g_pkt);
    fuzz_cleanup_decoders();

    /* Open input using the reusable probe buffer. */
    struct timespec t_open_start, t_info_start, t_decode_start, t_cleanup_start;
    FUZZ_PHASE_START(t_open_start);
    __afl_map_edge(0x1002);

    AVFormatContext *fmt_ctx = fuzz_open_input(buf, size);
    if (!fmt_ctx) {
        __afl_map_edge(0x1006);
        FUZZ_PHASE_END(t_open_start, fuzz_phase_stats.t_open_ns);
        fuzz_report_phase("open", fuzz_phase_stats.t_open_ns);
        fuzz_phase_stats.calls++;
        fuzz_cancel_watchdog();
        return 0;
    }
    FUZZ_PHASE_END(t_open_start, fuzz_phase_stats.t_open_ns);
    fuzz_report_phase("open", fuzz_phase_stats.t_open_ns);
    fuzz_phase_stats.avio_alloc++;
    __afl_map_edge(0x1010);

    /* Find stream info — runs codec probing for each stream */
    FUZZ_PHASE_START(t_info_start);
    FUZZ_LOG("[ffmpeg debug] before avformat_find_stream_info\n");
    int ret = fuzz_with_watchdog(150, fuzz_run_find_stream_info, &fmt_ctx);
    FUZZ_LOG("[ffmpeg debug] after avformat_find_stream_info ret=%d\n", ret);
    FUZZ_PHASE_END(t_info_start, fuzz_phase_stats.t_stream_info_ns);
    fuzz_report_phase("stream_info", fuzz_phase_stats.t_stream_info_ns);
    if (ret < 0) {
        __afl_map_edge(0x1011);
        avformat_close_input(&fmt_ctx);
        fuzz_release_io_state();
        fuzz_phase_stats.calls++;
        fuzz_cancel_watchdog();
        return 0;
    }
    __afl_map_edge(0x1020);

    /* Iterate packets and decode each stream */
    unsigned nb_streams = fmt_ctx->nb_streams;
    if (nb_streams > FUZZ_MAX_DECODERS) nb_streams = FUZZ_MAX_DECODERS;

    unsigned total_packets = 0;
    unsigned total_frames  = 0;

    /* Per-stream decoder contexts — reuse existing open decoders when
     * the codec ID matches the new stream, otherwise reset. */
    for (unsigned i = 0; i < nb_streams; i++) {
        const AVCodecParameters *par = fmt_ctx->streams[i]->codecpar;
        /* Audio, video and subtitles are all decoded -- but not through the
         * same API, which is the whole point of tracking codec_type below.
         *
         * avcodec_send_packet/avcodec_receive_frame is the frame-based API
         * for audio and video only. A subtitle decoder driven through it
         * reaches decode_simple_internal, which returns 0 without filling
         * frame->buf[0], and libavcodec's `if (!ret) av_assert0(frame->buf[0])`
         * aborts -- a SIGABRT manufactured by the harness rather than a
         * finding in the library. Subtitles go through
         * avcodec_decode_subtitle2 in the packet loop instead.
         *
         * Data and attachment streams have no decoder contract at all and
         * stay excluded. */
        if (par->codec_type != AVMEDIA_TYPE_VIDEO &&
            par->codec_type != AVMEDIA_TYPE_AUDIO &&
            par->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            continue;
        }
        const AVCodec *codec = avcodec_find_decoder(par->codec_id);
        if (!codec) { continue; }

        DecoderSlot *slot = &g_decs[i];
        if (!slot->inited || slot->codec_id != par->codec_id) {
            if (slot->ctx) avcodec_free_context(&slot->ctx);
            slot->ctx = avcodec_alloc_context3(codec);
            if (!slot->ctx) continue;
            avcodec_parameters_to_context(slot->ctx, par);
            ret = avcodec_open2(slot->ctx, codec, NULL);
            if (ret < 0) {
                avcodec_free_context(&slot->ctx);
                slot->ctx      = NULL;
                slot->codec_id = 0;
                slot->inited   = 0;
                continue;
            }
            slot->codec_id   = par->codec_id;
            slot->codec_type = par->codec_type;
            slot->inited     = 1;
            fuzz_phase_stats.decoder_opens++;
        } else {
            avcodec_parameters_to_context(slot->ctx, par);
            slot->codec_type = par->codec_type;
            fuzz_phase_stats.decoder_reuses++;
        }
        __afl_map_edge(0x1100 + (i & 0xFF));
    }

    /* Read and decode frames */
    FUZZ_PHASE_START(t_decode_start);
    while (av_read_frame(fmt_ctx, g_pkt) >= 0) {
        total_packets++;
        __afl_map_edge(0x1200 + (g_pkt->stream_index & 0x1F));

        int si = g_pkt->stream_index;
        if (si >= 0 && (int)si < (int)nb_streams && g_decs[si].inited && g_decs[si].ctx) {
            if (g_decs[si].codec_type == AVMEDIA_TYPE_SUBTITLE) {
                /* Subtitles use the packet-in/AVSubtitle-out API. Driving
                 * them through send_packet/receive_frame trips
                 * av_assert0(frame->buf[0]) inside decode_simple_internal. */
                AVSubtitle sub;
                memset(&sub, 0, sizeof(sub));
                int got_sub = 0;
                ret = avcodec_decode_subtitle2(g_decs[si].ctx, &sub, &got_sub, g_pkt);
                if (ret >= 0 && got_sub) {
                    total_frames++;
                    __afl_map_edge(0x1600 + (si & 0xFF));
                    /* Touch the decoded rects so a corrupt region descriptor
                     * is actually observed rather than just allocated. */
                    for (unsigned r = 0; r < sub.num_rects; r++) {
                        const AVSubtitleRect *rect = sub.rects[r];
                        if (!rect) continue;
                        __afl_map_edge(0x1700 + (rect->type & 0x0F));
                        if (rect->w > 0 && rect->h > 0) __afl_map_edge(0x1710);
                        if (rect->nb_colors > 0)        __afl_map_edge(0x1711);
                    }
                }
                /* avcodec_decode_subtitle2 allocates rects on success; the
                 * struct is zeroed above so this is safe on failure too.
                 * Omitting it leaks every decoded subtitle, per execution. */
                avsubtitle_free(&sub);
            } else {
                ret = avcodec_send_packet(g_decs[si].ctx, g_pkt);
                if (ret >= 0) {
                    while (avcodec_receive_frame(g_decs[si].ctx, g_frame) >= 0) {
                        total_frames++;
                        __afl_map_edge(0x1400 + (si & 0xFF));
                    }
                }
            }
        }

        av_packet_unref(g_pkt);

        if ((total_packets & 1023) == 0) {
            FUZZ_LOG("[ffmpeg decode] packets=%u frames=%u\n", total_packets, total_frames);
        }

        /* Guard against pathological inputs */
        if (total_packets > 500 || total_frames > 500) break;
    }
    FUZZ_PHASE_END(t_decode_start, fuzz_phase_stats.t_decode_ns);
    fuzz_report_phase("decode", fuzz_phase_stats.t_decode_ns);

    FUZZ_PHASE_START(t_cleanup_start);
    avformat_close_input(&fmt_ctx);
    fuzz_release_io_state();

    /* Ask glibc to release free top arena pages back to the kernel. */
    malloc_trim(0);
    FUZZ_PHASE_END(t_cleanup_start, fuzz_phase_stats.t_cleanup_ns);
    fuzz_report_phase("cleanup", fuzz_phase_stats.t_cleanup_ns);

    fuzz_phase_stats.calls++;
    __afl_map_edge(0x1500);
    fuzz_cancel_watchdog();

    if (++fuzz_reset_counter >= 1000) {
        fuzz_reset_counter = 0;
        if (fuzz_network_inited) {
            avformat_network_deinit();
            fuzz_network_inited = 0;
        }
        FUZZ_LOG("[ffmpeg reset] global state cleared after %d calls\n", fuzz_phase_stats.calls);
    }

    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_ffmpeg(buf, len);
    }
    fuzz_print_phase_stats();
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(size);
        if (buf) {
            fread(buf, 1, size, f);
            int rc = fuzz_ffmpeg(buf, size);
            free(buf);
            fclose(f);
            fuzz_print_phase_stats();
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_ffmpeg(buf, n);
    }
    return 0;
}
#endif

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_ffmpeg(buf, size);
}

/* Write accumulated phase stats to a CSV for post-run analysis. */
__attribute__((visibility("default")))
void fuzz_write_profile(void) {
    const char *path = getenv("FFMPEG_PROFILE_OUT");
    if (!path || !path[0]) return;

    static int atexit_registered = 0;
    if (!atexit_registered) {
        atexit_registered = 1;
        atexit(fuzz_write_profile);
    }
    /* The atexit handler will flush on normal exit. Also flush now so the
     * wrapper can read intermediate data if it checks before exit. */
    {
        FILE *f = fopen(path, "w");
        if (!f) return;
        fprintf(f, "phase,calls_ns,total_ns,avg_ns\n");
        if (fuzz_phase_stats.calls > 0) {
            fprintf(f, "open_input,%d,%lld,%lld\n",
                    fuzz_phase_stats.avio_alloc,
                    fuzz_phase_stats.t_open_ns,
                    fuzz_phase_stats.t_open_ns / fuzz_phase_stats.calls);
            fprintf(f, "stream_info,%d,%lld,%lld\n",
                    fuzz_phase_stats.calls,
                    fuzz_phase_stats.t_stream_info_ns,
                    fuzz_phase_stats.t_stream_info_ns / fuzz_phase_stats.calls);
            fprintf(f, "decode,%d,%lld,%lld\n",
                    fuzz_phase_stats.calls,
                    fuzz_phase_stats.t_decode_ns,
                    fuzz_phase_stats.t_decode_ns / fuzz_phase_stats.calls);
            fprintf(f, "cleanup,%d,%lld,%lld\n",
                    fuzz_phase_stats.calls,
                    fuzz_phase_stats.t_cleanup_ns,
                    fuzz_phase_stats.t_cleanup_ns / fuzz_phase_stats.calls);
        }
        fprintf(f, "allocations,%d,%d,%d\n",
                fuzz_phase_stats.avio_alloc,
                fuzz_phase_stats.decoder_opens,
                fuzz_phase_stats.decoder_reuses);
        fclose(f);
    }
}

__attribute__((visibility("default")))
void fuzz_print_phase_stats(void) {
    printf("[fuzz_ffmpeg phases] calls=%d\n", fuzz_phase_stats.calls);
    if (fuzz_phase_stats.calls > 0) {
        printf("  open_input_av: %.1f ms total, %.2f ms/call\n",
               fuzz_phase_stats.t_open_ns / 1e6,
               fuzz_phase_stats.t_open_ns / (double)fuzz_phase_stats.calls / 1e6);
        printf("  stream_info:   %.1f ms total, %.2f ms/call\n",
               fuzz_phase_stats.t_stream_info_ns / 1e6,
               fuzz_phase_stats.t_stream_info_ns / (double)fuzz_phase_stats.calls / 1e6);
        printf("  decode:        %.1f ms total, %.2f ms/call\n",
               fuzz_phase_stats.t_decode_ns / 1e6,
               fuzz_phase_stats.t_decode_ns / (double)fuzz_phase_stats.calls / 1e6);
        printf("  cleanup:       %.1f ms total, %.2f ms/call\n",
               fuzz_phase_stats.t_cleanup_ns / 1e6,
               fuzz_phase_stats.t_cleanup_ns / (double)fuzz_phase_stats.calls / 1e6);
    }
    printf("  avio_alloc: %d, decoder_opens: %d, decoder_reuses: %d\n",
           fuzz_phase_stats.avio_alloc,
           fuzz_phase_stats.decoder_opens,
           fuzz_phase_stats.decoder_reuses);
}
