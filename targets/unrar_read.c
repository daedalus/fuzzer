/* Fuzz target for RAR archive parsing via libarchive (dlopen).
 *
 * Loads libarchive.so at runtime and calls archive_read_* APIs to
 * exercise RAR header parsing, decompression, and CRC verification.
 * No compile-time headers needed — all function pointers resolved via
 * dlsym.
 *
 * Compile shared library (for inprocess modes):
 *   gcc -O2 -g -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o unrar_read.so unrar_read.c -ldl -Wl,--export-dynamic
 *
 * Compile standalone:
 *   gcc -O2 -g -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/unrar_read targets/unrar_read.c -ldl
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* libarchive function pointers (resolved at runtime) */
typedef void *archive_t;
typedef void *archive_entry_t;

static archive_t (*la_archive_read_new)(void);
static int (*la_archive_read_support_filter_all)(archive_t);
static int (*la_archive_read_support_format_all)(archive_t);
static int (*la_archive_read_open_memory)(archive_t, const void *, size_t);
static int (*la_archive_read_next_header)(archive_t, archive_entry_t *);
static const char *(*la_archive_entry_pathname)(archive_entry_t);
static int64_t (*la_archive_entry_size)(archive_entry_t);
static ssize_t (*la_archive_read_data)(archive_t, void *, size_t);
static int (*la_archive_read_free)(archive_t);

static void *la_handle = NULL;

static int load_libarchive(void) {
    if (la_handle) return 0;

    const char *names[] = {
        "libarchive.so.13",
        "libarchive.so",
        "libarchive.so.13.7.4",
        NULL
    };
    for (int i = 0; names[i]; i++) {
        la_handle = dlopen(names[i], RTLD_LAZY);
        if (la_handle) break;
    }
    if (!la_handle) return -1;

    #define LOAD(name) \
        la_##name = dlsym(la_handle, #name); \
        if (!la_##name) { dlclose(la_handle); la_handle = NULL; return -1; }

    LOAD(archive_read_new);
    LOAD(archive_read_support_filter_all);
    LOAD(archive_read_support_format_all);
    LOAD(archive_read_open_memory);
    LOAD(archive_read_next_header);
    LOAD(archive_entry_pathname);
    LOAD(archive_entry_size);
    LOAD(archive_read_data);
    LOAD(archive_read_free);
    #undef LOAD

    return 0;
}

__attribute__((visibility("default")))
int fuzz_unrar(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 4) { __afl_map_edge(0x1001); return 0; }

    if (load_libarchive() < 0) {
        __afl_map_edge(0x1002);
        return 0;
    }
    __afl_map_edge(0x1003);

    archive_t a = la_archive_read_new();
    if (!a) { __afl_map_edge(0x1004); return 0; }
    __afl_map_edge(0x1005);

    la_archive_read_support_filter_all(a);
    la_archive_read_support_format_all(a);

    int rc = la_archive_read_open_memory(a, (void *)buf, size);
    if (rc != 0) {
        __afl_map_edge(0x1100);
        la_archive_read_free(a);
        return 0;
    }
    __afl_map_edge(0x1200);

    archive_entry_t entry;
    int entry_count = 0;
    unsigned char discard_buf[4096];

    while (la_archive_read_next_header(a, &entry) == 0) {
        entry_count++;
        __afl_map_edge(0x1300 + (entry_count & 0xFF));

        int64_t entry_size = la_archive_entry_size(entry);
        if (entry_size > 0 && entry_size < 100 * 1024 * 1024) {
            la_archive_read_data(a, discard_buf, sizeof(discard_buf));
            __afl_map_edge(0x1400 + (entry_count & 0xFF));
        }

        if (entry_count > 1000) {
            __afl_map_edge(0x1500);
            break;
        }
    }

    __afl_map_edge(0x1600);
    la_archive_read_free(a);
    __afl_map_edge(0x1700);
    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_unrar(buf, len);
    }
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
            int rc = fuzz_unrar(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_unrar(buf, n);
    }
    return 0;
}
#endif
