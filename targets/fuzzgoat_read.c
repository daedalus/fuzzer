/* Fuzz target for Fuzzgoat JSON parser — in-memory wrapper.
 *
 * Feeds the raw fuzz buffer directly to json_parse() and walks the AST
 * to exercise allocation, parsing, and free paths.
 *
 * Compile standalone:
 *   gcc -O2 -g -Ivendor/fuzzgoat -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/fuzzgoat_read targets/fuzzgoat_read.c \
 *       vendor/fuzzgoat/fuzzgoat.c -lm
 *
 * Compile shared library:
 *   gcc -O2 -g -Ivendor/fuzzgoat -shared -fPIC \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/fuzzgoat_read.so targets/fuzzgoat_read.c \
 *       vendor/fuzzgoat/fuzzgoat.c -lm -Wl,--export-dynamic
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuzzgoat.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

static void process_value(json_value *value, int depth);

static void process_object(json_value *value, int depth)
{
    int length = value->u.object.length;
    for (int x = 0; x < length; x++) {
        __afl_map_edge(0x1100 + (depth & 0xFF));
        process_value(value->u.object.values[x].value, depth + 1);
    }
}

static void process_array(json_value *value, int depth)
{
    int length = value->u.array.length;
    for (int x = 0; x < length; x++) {
        __afl_map_edge(0x1200 + (depth & 0xFF));
        process_value(value->u.array.values[x], depth);
    }
}

static void process_value(json_value *value, int depth)
{
    if (!value) {
        __afl_map_edge(0x1300);
        return;
    }
    switch (value->type) {
    case json_none:
        __afl_map_edge(0x1400);
        break;
    case json_object:
        __afl_map_edge(0x1500);
        process_object(value, depth + 1);
        break;
    case json_array:
        __afl_map_edge(0x1600);
        process_array(value, depth + 1);
        break;
    case json_integer:
        __afl_map_edge(0x1700);
        break;
    case json_double:
        __afl_map_edge(0x1800);
        break;
    case json_string:
        __afl_map_edge(0x1900);
        break;
    case json_boolean:
        __afl_map_edge(0x1A00);
        break;
    case json_null:
        __afl_map_edge(0x1B00);
        break;
    default:
        __afl_map_edge(0x1C00);
        break;
    }
}

/* Optional settings trailer: input ending in "\0 F" parses with
 * json_parse_ex() so settings-gated branches become reachable. F bits:
 *   0x01 comments   0x02 max_memory = 64 << (F >> 4)
 *   0x04 error buf  0x08 allocation #(F >> 4) + 1 returns NULL
 * e.g. "[1," + "\0\x04" fills the error buffer. */
#define TRAILER_LEN 2
#define F_COMMENTS 0x01
#define F_LIMIT 0x02
#define F_ERRBUF 0x04
#define F_FAILALLOC 0x08
#define LIMIT_BASE 64

static unsigned int alloc_count, alloc_fail_at;

static void *failing_alloc(size_t n, int zero, void *user)
{
    (void)user;
    if (++alloc_count == alloc_fail_at) {
        __afl_map_edge(0x100A);
        return NULL;
    }
    return zero ? calloc(1, n) : malloc(n);
}

static void plain_free(void *p, void *user)
{
    (void)user;
    free(p);
}

static int has_trailer(const unsigned char *buf, size_t size)
{
    return size >= TRAILER_LEN && buf[size - TRAILER_LEN] == '\0';
}

static json_value *parse_trailer(const unsigned char *buf, size_t len,
                                 unsigned char flags, json_settings *settings)
{
    char err[json_error_max] = {0};
    unsigned int nibble = flags >> 4;

    __afl_map_edge(0x1006);
    memset(settings, 0, sizeof *settings);
    if (flags & F_COMMENTS) {
        __afl_map_edge(0x1007);
        settings->settings |= json_enable_comments;
    }
    if (flags & F_LIMIT) {
        __afl_map_edge(0x1008);
        settings->max_memory = (unsigned long)LIMIT_BASE << nibble;
    }

    /* Always custom: json_value_free_ex() needs mem_free set. */
    alloc_count = 0;
    alloc_fail_at = (flags & F_FAILALLOC) ? nibble + 1 : 0;
    settings->mem_alloc = failing_alloc;
    settings->mem_free = plain_free;

    json_value *value = json_parse_ex(settings, (const json_char *)buf, len,
                                      (flags & F_ERRBUF) ? err : NULL);
    if (!value && err[0])
        __afl_map_edge(0x1009);
    return value;
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size)
{
    __afl_map_edge(0x1000);
    if (size == 0) {
        __afl_map_edge(0x1001);
        return 0;
    }

    json_settings settings = {0};
    int trailer = has_trailer(buf, size);
    json_value *value = trailer
        ? parse_trailer(buf, size - TRAILER_LEN, buf[size - 1], &settings)
        : json_parse((const json_char *)buf, size);
    if (value) {
        __afl_map_edge(0x1002);
        process_value(value, 0);
        if (trailer)
            json_value_free_ex(&settings, value);
        else
            json_value_free(value);
        __afl_map_edge(0x1003);
    } else {
        __afl_map_edge(0x1004);
    }

    __afl_map_edge(0x1005);
    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void)
{
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_shm_run(buf, len);
    }
    return 0;
}
#else
int main(int argc, char **argv)
{
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(size);
        if (buf) {
            fread(buf, 1, size, f);
            int rc = fuzz_shm_run(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_shm_run(buf, n);
    }
    return 0;
}
#endif
