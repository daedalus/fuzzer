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

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size)
{
    __afl_map_edge(0x1000);
    if (size == 0) {
        __afl_map_edge(0x1001);
        return 0;
    }

    json_value *value = json_parse((const json_char *)buf, size);
    if (value) {
        __afl_map_edge(0x1002);
        process_value(value, 0);
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
