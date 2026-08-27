/* Fuzz target for SQLite (vendored amalgamation) — exercises the database
 * file reader (page/b-tree/record decoding, via sqlite3_deserialize) and,
 * for inputs that are not database images, the SQL parser and bytecode
 * engine.
 *
 * Modeled on lz4_read.c / secp256k1_read.c: same __afl_map_edge landmark
 * scheme, same fuzz_shm_run entry point for direct_lite mode, same "bound
 * the work so bad input can't OOM or hang the in-process fuzzer" discipline.
 *
 * Input layout: NONE. The input IS the artifact.
 *
 *   bytes 0..15 == "SQLite format 3\0" and size >= 100  -> database image
 *   anything else                                       -> SQL text
 *
 * This is deliberately different from lz4_read.c, which spends byte 0 on a
 * mode selector. The structure-aware SQLite mutator is gated on the file
 * itself:
 *
 *   core/operator_registry.py:
 *     "sqlite_chunk_mutate": lambda d: len(d) >= 100
 *                            and d[:16] == b"SQLite format 3\x00"
 *
 * A one-byte prefix would shift the magic to offset 1, the sniffer would
 * never fire, and every database in the corpus would be mutated as if it
 * were a flat byte string — the exact opposite of what a format-aware
 * fuzzer is for. The dispatch below uses the same two conditions as that
 * predicate so the target and the sniffer agree by construction, and
 * dictionaries/sqlite.dict tokens land at their real file offsets.
 *
 * What is being tested: SQLite's own documented contract for hostile
 * database files (https://www.sqlite.org/security.html). A corrupt database
 * may legitimately produce SQLITE_CORRUPT/SQLITE_NOTADB/SQLITE_ERROR — those
 * are expected returns, not crashes. What must NOT happen is a segfault, a
 * buffer overflow, or an assertion failure, and those are what this target
 * is here to find. Same for the SQL path: a syntax error is a pass.
 *
 * Sandboxing (both paths, see open_sandboxed()):
 *   - :memory: only; nothing ever touches the filesystem
 *   - SQLITE_DBCONFIG_DEFENSIVE      — no writes to shadow/schema tables
 *   - SQLITE_DBCONFIG_TRUSTED_SCHEMA — schema cannot invoke non-trusted
 *                                      functions; the documented mitigation
 *                                      for opening untrusted DB files
 *   - extension loading off, ATTACH/DETACH denied on the SQL path
 *   - progress handler with a fixed opcode budget so an input cannot hang
 *   - hard heap limit so an input cannot OOM the fuzzer process
 * direct_lite runs the target in-process: a hang or an OOM here kills the
 * whole campaign, not one exec, so these bounds are load-bearing rather
 * than defensive decoration.
 *
 * Vendor the SQLite amalgamation first (extracts to vendor/sqlite/):
 *   tools/vendor_sqlite.sh
 *
 * Then build via the normal path, which handles ASAN/cmplog variants:
 *   tools/build_targets.sh
 *
 * Manual build (what build_targets.sh does under the hood) — note the
 * amalgamation is compiled SEPARATELY, without the shim:
 *   clang -O2 -g -fPIC -I vendor/sqlite -DSQLITE_THREADSAFE=0 \
 *       -c vendor/sqlite/sqlite3.c -o /tmp/sqlite3.o
 *   clang -O2 -g -shared -fPIC -I vendor/sqlite \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/sqlite_read.so targets/sqlite_read.c \
 *       /tmp/sqlite3.o -lm -lpthread -Wl,--export-dynamic
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * sqlite3.c must not be passed alongside the wrapper — doing so emits
 * __afl_map_shm / __afl_area / __afl_guarded_call into both objects and the
 * link fails with multiple-definition errors (Hard Rule 8).
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sqlite3.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Same two conditions as the sqlite_chunk_mutate sniffer predicate. */
#define SQLITE_FUZZ_MAGIC "SQLite format 3"  /* + implicit NUL == 16 bytes */
#define SQLITE_FUZZ_HDR_SIZE 100u

/* Bounds. Every one of these is a "the fuzzer process must survive this"
 * limit, not a correctness limit. */
#define SQLITE_FUZZ_MAX_DB (64u * 1024u * 1024u)      /* reject giant images  */
#define SQLITE_FUZZ_MAX_SQL (1u * 1024u * 1024u)      /* SQL text cap         */
#define SQLITE_FUZZ_HEAP_LIMIT (128 * 1024 * 1024)    /* sqlite allocator cap */
#define SQLITE_FUZZ_MAX_OPCODES 250000u               /* progress budget      */
#define SQLITE_FUZZ_MAX_ROWS 2000u                    /* rows per statement   */
#define SQLITE_FUZZ_MAX_TABLES 24u                    /* tables scanned       */
#define SQLITE_FUZZ_MAX_STMTS 64u                     /* statements per input */
#define SQLITE_FUZZ_PROGRESS_TICK 64                  /* vdbe ops per call    */

/* ── Landmark map ───────────────────────────────────────────────────────
 * The blocks below are disjoint on purpose. fuzz_drain() emits
 * `base + 7 + (rc & 0xFF)`, so every drain site needs 0x107 of headroom;
 * giving each one its own 0x200-aligned block keeps a statement's return
 * code from aliasing onto another site's landmarks, which would merge two
 * unrelated behaviours into one edge and hide coverage from the scheduler.
 *
 *   0x4000-0x400F  entry dispatch          0x4400-0x4417  table scan index
 *   0x4010         authorizer deny         0x4418         scan prepare failed
 *   0x4100-0x410F  deserialize outcome     0x4500         db path done
 *   0x4110-0x412F  header fields           0x4600-0x460F  sql path entry
 *   0x4200-0x4201  integrity_check prep    0x4700-0x47FF  sql prepare rc
 *   0x4300-0x4328  schema read             0x4800-0x483F  sql statement index
 *                                          0x4900         sql path done
 *   0x5000 / 0x5200 / 0x5400  fuzz_drain blocks (integrity / scan / sql)
 */
#define SQLITE_FUZZ_DRAIN_CHECK 0x5000u
#define SQLITE_FUZZ_DRAIN_SCAN 0x5200u
#define SQLITE_FUZZ_DRAIN_SQL 0x5400u

/* Header version bytes are one byte wide but only a handful of values are
 * meaningful; clamp so the landmark block stays bounded. */
static unsigned fuzz_clamp(unsigned char v) { return v <= 3u ? (unsigned)v : 4u; }

/* ── One-time library init ──────────────────────────────────────────── */
/* sqlite3_config() must run before sqlite3_initialize(), and in direct_lite
 * the process is long-lived across millions of execs, so this is done once
 * and never torn down. The heap limit is process-wide, which is what we
 * want: it bounds the sum of everything live, not one connection. */
static int fuzz_sqlite_init(void) {
    static int initialized = 0;
    if (initialized) return initialized > 0;
    if (sqlite3_initialize() != SQLITE_OK) {
        initialized = -1;
        return 0;
    }
    sqlite3_hard_heap_limit64((sqlite3_int64)SQLITE_FUZZ_HEAP_LIMIT);
    initialized = 1;
    return 1;
}

/* ── Work budget ────────────────────────────────────────────────────── */
/* Returning non-zero from the progress handler interrupts the current
 * statement with SQLITE_INTERRUPT. A recursive-CTE bomb or a corrupt page
 * that sends the b-tree walker in a circle terminates here instead of
 * wedging the campaign. */
static int fuzz_progress(void *ctx) {
    unsigned *ticks = (unsigned *)ctx;
    *ticks += (unsigned)SQLITE_FUZZ_PROGRESS_TICK;
    return *ticks > SQLITE_FUZZ_MAX_OPCODES;
}

/* Deny the statements that could reach outside this process. The DB path
 * does not install this (it needs PRAGMA integrity_check); it is the SQL
 * path, where the input chooses the statements, that needs it. */
static int fuzz_authorizer(void *unused, int op, const char *a, const char *b,
                           const char *c, const char *d) {
    (void)unused;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    switch (op) {
        case SQLITE_ATTACH:
        case SQLITE_DETACH:
        case SQLITE_PRAGMA:
            __afl_map_edge(0x4010);
            return SQLITE_DENY;
        default:
            return SQLITE_OK;
    }
}

/* ── Connection setup ───────────────────────────────────────────────── */
static sqlite3 *open_sandboxed(unsigned *ticks) {
    sqlite3 *db = NULL;
    if (sqlite3_open_v2(":memory:", &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        NULL) != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return NULL;
    }
    sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 0, NULL);
    sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, 1, NULL);
    sqlite3_db_config(db, SQLITE_DBCONFIG_TRUSTED_SCHEMA, 0, NULL);
    sqlite3_limit(db, SQLITE_LIMIT_LENGTH, 32 * 1024 * 1024);
    sqlite3_limit(db, SQLITE_LIMIT_SQL_LENGTH, (int)SQLITE_FUZZ_MAX_SQL);
    sqlite3_limit(db, SQLITE_LIMIT_EXPR_DEPTH, 128);
    sqlite3_limit(db, SQLITE_LIMIT_COMPOUND_SELECT, 32);
    sqlite3_limit(db, SQLITE_LIMIT_VDBE_OP, 100000);
    sqlite3_limit(db, SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 4096);
    sqlite3_progress_handler(db, SQLITE_FUZZ_PROGRESS_TICK, fuzz_progress, ticks);
    return db;
}

/* Step a prepared statement, touching every column value so the record
 * decoder actually runs. Reading a column is what turns a serial-type
 * header into a decode: without it the b-tree walk stops at the cell
 * boundary and the interesting parsing never happens. */
static void fuzz_drain(sqlite3_stmt *stmt, unsigned landmark) {
    unsigned rows = 0;
    int rc;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int ncol = sqlite3_column_count(stmt);
        for (int i = 0; i < ncol; i++) {
            switch (sqlite3_column_type(stmt, i)) {
                case SQLITE_INTEGER:
                    (void)sqlite3_column_int64(stmt, i);
                    __afl_map_edge(landmark + 1);
                    break;
                case SQLITE_FLOAT:
                    (void)sqlite3_column_double(stmt, i);
                    __afl_map_edge(landmark + 2);
                    break;
                case SQLITE_TEXT:
                    (void)sqlite3_column_text(stmt, i);
                    (void)sqlite3_column_bytes(stmt, i);
                    __afl_map_edge(landmark + 3);
                    break;
                case SQLITE_BLOB:
                    (void)sqlite3_column_blob(stmt, i);
                    (void)sqlite3_column_bytes(stmt, i);
                    __afl_map_edge(landmark + 4);
                    break;
                default:
                    __afl_map_edge(landmark + 5);
                    break;
            }
        }
        if (++rows >= SQLITE_FUZZ_MAX_ROWS) {
            __afl_map_edge(landmark + 6);
            break;
        }
    }
    __afl_map_edge(landmark + 7 + (unsigned)(rc & 0xFF));
}

/* ── Database-image path: sqlite3_deserialize ───────────────────────── */
static int fuzz_sqlite_db(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x4100);
    if (size > SQLITE_FUZZ_MAX_DB) {
        __afl_map_edge(0x4101);
        return 0;
    }

    unsigned ticks = 0;
    sqlite3 *db = open_sandboxed(&ticks);
    if (!db) return 0;

    /* The buffer handed to sqlite3_deserialize must come from
     * sqlite3_malloc64, and with RESIZEABLE|FREEONCLOSE sqlite owns it from
     * this point on: it may realloc the buffer as pages are added and frees
     * it at sqlite3_close(). Never free() or touch `image` after this call
     * — the pointer can be stale. This mirrors SQLite's own test/dbfuzz2.c,
     * which also leaves the buffer to the connection on the error path. */
    unsigned char *image = sqlite3_malloc64(size ? size : 1);
    if (!image) {
        sqlite3_close(db);
        return 0;
    }
    memcpy(image, buf, size);

    int rc = sqlite3_deserialize(db, "main", image, (sqlite3_int64)size,
                                 (sqlite3_int64)size,
                                 SQLITE_DESERIALIZE_RESIZEABLE |
                                     SQLITE_DESERIALIZE_FREEONCLOSE);
    if (rc != SQLITE_OK) {
        __afl_map_edge(0x4102);
        sqlite3_close(db);
        return 0;
    }
    __afl_map_edge(0x4103);

    /* Landmark a few header fields the mutator flips, so structurally
     * different-but-valid databases get distinct coverage rather than all
     * collapsing onto the same edge. The version bytes are clamped rather
     * than used raw: a fuzzed byte spans 0..255, and an unclamped
     * `base + byte` would run one field's landmarks straight through the
     * next field's block and alias them. */
    __afl_map_edge(0x4110 + fuzz_clamp(buf[18]));        /* write version */
    __afl_map_edge(0x4118 + fuzz_clamp(buf[19]));        /* read version  */
    __afl_map_edge(0x4120 + (unsigned)(buf[56] & 0x7));  /* text encoding */
    __afl_map_edge(0x4128 + (unsigned)(buf[44] & 0x7));  /* schema format */

    sqlite3_stmt *stmt = NULL;

    /* integrity_check is the whole point of the DB path: it walks every
     * page, every b-tree, every overflow chain and every index, which is
     * where corrupt-image bugs live. The argument caps the number of
     * reported problems so a thoroughly broken file doesn't spend the
     * budget formatting error strings. */
    if (sqlite3_prepare_v2(db, "PRAGMA integrity_check(4)", -1, &stmt, NULL) ==
        SQLITE_OK) {
        __afl_map_edge(0x4200);
        fuzz_drain(stmt, SQLITE_FUZZ_DRAIN_CHECK);
        sqlite3_finalize(stmt);
    } else {
        __afl_map_edge(0x4201);
    }

    /* Schema parse: reading sqlite_master runs the tokenizer and parser
     * over whatever CREATE statements the file claims to contain. */
    char *names[SQLITE_FUZZ_MAX_TABLES];
    unsigned ntables = 0;
    if (sqlite3_prepare_v2(db,
                           "SELECT name, type FROM sqlite_master "
                           "WHERE type IN ('table','view') AND name IS NOT NULL",
                           -1, &stmt, NULL) == SQLITE_OK) {
        __afl_map_edge(0x4300);
        while (sqlite3_step(stmt) == SQLITE_ROW && ntables < SQLITE_FUZZ_MAX_TABLES) {
            const unsigned char *name = sqlite3_column_text(stmt, 0);
            if (!name) continue;
            char *copy = sqlite3_mprintf("%s", name);
            if (!copy) break;
            names[ntables++] = copy;
            __afl_map_edge(0x4310 + ntables);
        }
        sqlite3_finalize(stmt);
    } else {
        __afl_map_edge(0x4301);
    }

    /* Table scans: this is what pulls cells off pages and decodes records.
     * %w quotes the identifier the way SQLite quotes its own schema
     * identifiers (doubling embedded '"'), so a table literally named
     * `a"; DROP` is scanned rather than reinterpreted as more SQL. */
    for (unsigned i = 0; i < ntables; i++) {
        char *sql = sqlite3_mprintf("SELECT * FROM \"%w\"", names[i]);
        if (sql) {
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
                __afl_map_edge(0x4400 + i);
                fuzz_drain(stmt, SQLITE_FUZZ_DRAIN_SCAN);
                sqlite3_finalize(stmt);
            } else {
                __afl_map_edge(0x4418);
            }
            sqlite3_free(sql);
        }
        sqlite3_free(names[i]);
    }

    __afl_map_edge(0x4500);
    sqlite3_close(db);
    return 0;
}

/* ── SQL-text path ──────────────────────────────────────────────────── */
static int fuzz_sqlite_sql(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x4600);
    if (size > SQLITE_FUZZ_MAX_SQL) size = SQLITE_FUZZ_MAX_SQL;

    char *sql = malloc(size + 1);
    if (!sql) return 0;
    memcpy(sql, buf, size);
    sql[size] = '\0';

    unsigned ticks = 0;
    sqlite3 *db = open_sandboxed(&ticks);
    if (!db) {
        free(sql);
        return 0;
    }
    sqlite3_set_authorizer(db, fuzz_authorizer, NULL);

    /* A scratch table gives DML and SELECT something real to bind against,
     * so inputs that are not pure DDL still reach the bytecode engine
     * rather than dying in name resolution. */
    if (sqlite3_exec(db,
                     "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c BLOB, d REAL);"
                     "INSERT INTO t VALUES(1,'one',x'01',1.5),(2,'two',x'0202',2.5);"
                     "CREATE INDEX t_b ON t(b);",
                     NULL, NULL, NULL) == SQLITE_OK) {
        __afl_map_edge(0x4601);
    }

    /* Walk the input statement by statement: sqlite3_prepare_v2's tail
     * pointer is the parser's own statement splitter, so a semicolon inside
     * a string literal or a trigger body is handled correctly — which a
     * hand-rolled split on ';' would get wrong. */
    const char *tail = sql;
    unsigned nstmt = 0;
    while (*tail && nstmt < SQLITE_FUZZ_MAX_STMTS && ticks <= SQLITE_FUZZ_MAX_OPCODES) {
        sqlite3_stmt *stmt = NULL;
        const char *next = NULL;
        int rc = sqlite3_prepare_v2(db, tail, -1, &stmt, &next);
        if (rc != SQLITE_OK) {
            /* Don't abandon the input here. A denied ATTACH or a single
             * syntax error fails one statement, and pzTail still points
             * past it, so the statements after it are reachable — bailing
             * out would silently cap every multi-statement input at its
             * first bad statement. The nstmt++ keeps a wall of errors from
             * spinning: failures spend the statement budget too. */
            __afl_map_edge(0x4700 + (unsigned)(rc & 0xFF));
            if (stmt) sqlite3_finalize(stmt);
            nstmt++;
            if (!next || next <= tail) break;
            tail = next;
            continue;
        }
        if (!stmt) {  /* whitespace or a comment — no statement to run */
            __afl_map_edge(0x4602);
            if (!next || next == tail) break;
            tail = next;
            continue;
        }
        __afl_map_edge(0x4800 + nstmt);
        fuzz_drain(stmt, SQLITE_FUZZ_DRAIN_SQL);
        sqlite3_finalize(stmt);
        nstmt++;
        if (!next || next == tail) break;
        tail = next;
    }

    __afl_map_edge(0x4900);
    sqlite3_close(db);
    free(sql);
    return 0;
}

__attribute__((visibility("default")))
int fuzz_sqlite(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x4000);
    if (size == 0) {
        __afl_map_edge(0x4001);
        return 0;
    }
    if (!fuzz_sqlite_init()) {
        __afl_map_edge(0x4002);
        return 0;
    }
    if (size >= SQLITE_FUZZ_HDR_SIZE &&
        memcmp(buf, SQLITE_FUZZ_MAGIC, sizeof(SQLITE_FUZZ_MAGIC)) == 0) {
        __afl_map_edge(0x4003);
        return fuzz_sqlite_db(buf, size);
    }
    __afl_map_edge(0x4004);
    return fuzz_sqlite_sql(buf, size);
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_sqlite(buf, size);
}
