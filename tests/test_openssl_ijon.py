"""openssl_read target IJON slots (--ijon), driven through a fake libcrypto.

Element counts are not coverage: a certificate with 3 or 30 extensions
walks the same decoder edges. The target exposes each count as an
IJON_MAX slot, one direct slot per signal (no hashing, no aliasing):

    0 cert extensions   1 cert subject RDNs   2 cert issuer RDNs
    3 CRL revoked       4 CRL extensions      5 CSR attributes
    6 CSR subject RDNs  7 PKCS#7 signers
    value = min(n, CAP) + 1                    0 means "never set"

The fake library keeps the test independent of the vendored tree: byte 0
picks the one decoder that succeeds ('X' cert, 'C' CRL, 'R' CSR, 'P'
PKCS#7), bytes 1..3 are its counts; 0xff reads as -1 (absent stack),
0xfe as 2^20 (past the cap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import IjonMaxShm, ShmCoverage
from fuzzer_tool.core.ijon import IJON_MAP_SLOTS
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

TARGET = "targets/openssl_read.c"
MAP_ENTRIES = 8192
IJON_ENV = "__AFL_IJON_SHM_ID"

# Slot layout spec (independent of the C constants).
CERT_EXT, CERT_SUBJ, CERT_ISS, CRL_REV, CRL_EXT, REQ_ATTR, REQ_SUBJ, P7_SIGNERS = range(8)
CAP = 256
HUGE = 0xFE

FAKE_HDR = r"""
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#define OPENSSL_INIT_NO_LOAD_CONFIG 0x80L
#define OPENSSL_free(p) free(p)
typedef struct fake_count { int n; } X509_NAME;
typedef struct fake_count STACK_OF_X509_REVOKED;
typedef struct fake_count STACK_OF_PKCS7_SIGNER_INFO;
typedef struct fake_obj { struct fake_count c[3]; } X509, X509_CRL, X509_REQ, PKCS7;
typedef struct fake_bio { int unused; } BIO;
typedef struct fake_bio BIO_METHOD;
typedef struct fake_pkey { int unused; } EVP_PKEY;
#define STACK_OF(t) STACK_OF_##t
int OPENSSL_init_crypto(uint64_t, const void *);
BIO *BIO_new(const BIO_METHOD *);
const BIO_METHOD *BIO_s_null(void);
void ERR_clear_error(void);
void EVP_PKEY_free(EVP_PKEY *);
EVP_PKEY *d2i_AutoPrivateKey(EVP_PKEY **, const unsigned char **, long);
EVP_PKEY *d2i_PUBKEY(EVP_PKEY **, const unsigned char **, long);
#define FAKE_DECL(T, pfx) \
    T *d2i_##pfx(T **, const unsigned char **, long); \
    int i2d_##pfx(T *, unsigned char **); \
    void pfx##_free(T *);
FAKE_DECL(X509, X509)
FAKE_DECL(X509_CRL, X509_CRL)
FAKE_DECL(X509_REQ, X509_REQ)
FAKE_DECL(PKCS7, PKCS7)
int X509_print(BIO *, X509 *);
int X509_CRL_print(BIO *, X509_CRL *);
int X509_REQ_print(BIO *, X509_REQ *);
int PKCS7_print_ctx(BIO *, const PKCS7 *, int, const void *);
EVP_PKEY *X509_get_pubkey(X509 *);
int X509_check_ca(X509 *);
int X509_get_ext_count(const X509 *);
X509_NAME *X509_get_subject_name(const X509 *);
X509_NAME *X509_get_issuer_name(const X509 *);
int X509_NAME_entry_count(const X509_NAME *);
STACK_OF(X509_REVOKED) *X509_CRL_get_REVOKED(X509_CRL *);
int sk_X509_REVOKED_num(const STACK_OF(X509_REVOKED) *);
int X509_CRL_get_ext_count(const X509_CRL *);
int X509_REQ_get_attr_count(const X509_REQ *);
X509_NAME *X509_REQ_get_subject_name(const X509_REQ *);
STACK_OF(PKCS7_SIGNER_INFO) *PKCS7_get_signer_info(PKCS7 *);
int sk_PKCS7_SIGNER_INFO_num(const STACK_OF(PKCS7_SIGNER_INFO) *);
"""

FAKE_LIB = r"""
#include <openssl/x509.h>
static struct fake_bio g_bio;
int OPENSSL_init_crypto(uint64_t o, const void *s) { (void)o; (void)s; return 1; }
BIO *BIO_new(const BIO_METHOD *m) { (void)m; return &g_bio; }
const BIO_METHOD *BIO_s_null(void) { return &g_bio; }
void ERR_clear_error(void) {}
void EVP_PKEY_free(EVP_PKEY *k) { (void)k; }
EVP_PKEY *d2i_AutoPrivateKey(EVP_PKEY **a, const unsigned char **p, long n) {
    (void)a; (void)p; (void)n; return NULL;
}
EVP_PKEY *d2i_PUBKEY(EVP_PKEY **a, const unsigned char **p, long n) {
    (void)a; (void)p; (void)n; return NULL;
}
#define HUGE_COUNT (1 << 20)
static struct fake_obj *fake_d2i(char kind, const unsigned char **p, long n) {
    if (n < 4 || (*p)[0] != kind) return NULL;
    struct fake_obj *o = calloc(1, sizeof *o);
    for (int i = 0; i < 3; i++) {
        unsigned char b = (*p)[1 + i];
        o->c[i].n = b == 0xff ? -1 : b == 0xfe ? HUGE_COUNT : b;
    }
    return o;
}
#define FAKE_DEF(T, pfx, kind) \
    T *d2i_##pfx(T **a, const unsigned char **p, long n) { (void)a; return fake_d2i(kind, p, n); } \
    int i2d_##pfx(T *o, unsigned char **out) { (void)o; *out = NULL; return 0; } \
    void pfx##_free(T *o) { free(o); }
FAKE_DEF(X509, X509, 'X')
FAKE_DEF(X509_CRL, X509_CRL, 'C')
FAKE_DEF(X509_REQ, X509_REQ, 'R')
FAKE_DEF(PKCS7, PKCS7, 'P')
int X509_print(BIO *b, X509 *x) { (void)b; (void)x; return 1; }
int X509_CRL_print(BIO *b, X509_CRL *x) { (void)b; (void)x; return 1; }
int X509_REQ_print(BIO *b, X509_REQ *x) { (void)b; (void)x; return 1; }
int PKCS7_print_ctx(BIO *b, const PKCS7 *x, int i, const void *c) {
    (void)b; (void)x; (void)i; (void)c; return 1;
}
EVP_PKEY *X509_get_pubkey(X509 *x) { (void)x; return NULL; }
int X509_check_ca(X509 *x) { (void)x; return 0; }
int X509_get_ext_count(const X509 *x) { return x->c[0].n; }
X509_NAME *X509_get_subject_name(const X509 *x) { return (X509_NAME *)&x->c[1]; }
X509_NAME *X509_get_issuer_name(const X509 *x) { return (X509_NAME *)&x->c[2]; }
int X509_NAME_entry_count(const X509_NAME *n) { return n->n; }
STACK_OF(X509_REVOKED) *X509_CRL_get_REVOKED(X509_CRL *c) {
    return c->c[0].n < 0 ? NULL : &c->c[0];
}
int sk_X509_REVOKED_num(const STACK_OF(X509_REVOKED) *s) { return s ? s->n : -1; }
int X509_CRL_get_ext_count(const X509_CRL *c) { return c->c[1].n; }
int X509_REQ_get_attr_count(const X509_REQ *r) { return r->c[0].n; }
X509_NAME *X509_REQ_get_subject_name(const X509_REQ *r) { return (X509_NAME *)&r->c[1]; }
STACK_OF(PKCS7_SIGNER_INFO) *PKCS7_get_signer_info(PKCS7 *p) {
    return p->c[0].n < 0 ? NULL : &p->c[0];
}
int sk_PKCS7_SIGNER_INFO_num(const STACK_OF(PKCS7_SIGNER_INFO) *s) { return s ? s->n : -1; }
"""

HARNESS = r"""
#include <stdio.h>
int main(int argc, char **argv) {
    static unsigned char buf[1 << 16];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    return fuzz_shm_run(buf, n);
}
"""

_HEADERS = ("bio", "crypto", "err", "evp", "pkcs7", "x509", "x509v3")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("osslijon")
    inc = tmp / "openssl"
    inc.mkdir()
    (inc / "x509.h").write_text(FAKE_HDR)
    for name in _HEADERS:
        if name != "x509":
            (inc / f"{name}.h").write_text("#include <openssl/x509.h>\n")

    target = Path(__file__).resolve().parents[1] / TARGET
    src = f'{FAKE_LIB}\n#include "{target}"\n{HARNESS}'
    return _build(tmp, src, "-fsanitize-coverage=trace-pc-guard", f"-I{tmp}")


def _slots(exe, tmp_path, data: bytes, ijon: bool = True) -> np.ndarray | None:
    import subprocess

    inp = tmp_path / "in.bin"
    inp.write_bytes(data)
    shm = ShmCoverage(size=MAP_ENTRIES)
    seg = IjonMaxShm(slots=IJON_MAP_SLOTS) if ijon else None
    try:
        env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        if seg is not None:
            env[IJON_ENV] = seg.env_id
        rc = subprocess.run([str(exe), str(inp)], env=env, timeout=30).returncode
        assert rc == 0
        if seg is None:
            return None
        out = np.zeros(seg.slots, dtype=np.uint64)
        seg.read_into(out)
        return out
    finally:
        shm.cleanup()
        if seg is not None:
            seg.cleanup()


def _val(n: int) -> int:
    return min(n, CAP) + 1


def _nonzero(v) -> dict[int, int]:
    return {int(i): int(v[i]) for i in np.flatnonzero(v)}


@requires_clang
@pytest.mark.parametrize(
    ("blob", "expected"),
    [
        (b"X\x03\x05\x02", {CERT_EXT: _val(3), CERT_SUBJ: _val(5), CERT_ISS: _val(2)}),
        (b"C\x07\x01\x00", {CRL_REV: _val(7), CRL_EXT: _val(1)}),
        (b"R\x04\x06\x00", {REQ_ATTR: _val(4), REQ_SUBJ: _val(6)}),
        (b"P\x02\x00\x00", {P7_SIGNERS: _val(2)}),
    ],
)
def test_counts_land_in_own_slots(exe, tmp_path, blob, expected):
    assert _nonzero(_slots(exe, tmp_path, blob)) == expected


@requires_clang
def test_falsify_failed_decode_sets_nothing(exe, tmp_path):
    assert _nonzero(_slots(exe, tmp_path, b"Z\x03\x05\x02")) == {}


@requires_clang
def test_falsify_no_segment_is_inert(exe, tmp_path):
    assert _slots(exe, tmp_path, b"X\x03\x05\x02", ijon=False) is None


@requires_clang
def test_adversarial_zero_counts_register(exe, tmp_path):
    """n = 0 must still register: value 0 means "never set"."""
    v = _slots(exe, tmp_path, b"X\x00\x00\x00")
    assert _nonzero(v) == {CERT_EXT: 1, CERT_SUBJ: 1, CERT_ISS: 1}


@requires_clang
def test_adversarial_absent_stack_sets_no_slot(exe, tmp_path):
    """Unsigned PKCS#7 / CRL without revoked list: -1 is not a count."""
    assert _nonzero(_slots(exe, tmp_path, b"P\xff\x00\x00")) == {}
    assert _nonzero(_slots(exe, tmp_path, b"C\xff\x02\x00")) == {CRL_EXT: _val(2)}


@requires_clang
def test_adversarial_count_capped(exe, tmp_path):
    """Growth past CAP earns nothing: 2^20 revoked entries reads as CAP + 1."""
    v = _slots(exe, tmp_path, bytes((ord("C"), HUGE, 0, 0)))
    assert int(v[CRL_REV]) == CAP + 1
