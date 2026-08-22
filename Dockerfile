# Reproducible build and test environment for fuzzer-tool.
#
# The image exists because the suite is toolchain-sensitive in ways a bare
# `pip install` does not capture. conftest.py SKIPS rather than fails when
# clang or z3 is missing, so an incomplete environment reports a green run
# that silently never executed ~76 tests -- the same failure mode
# .github/workflows/ci.yml added an explicit verification step to catch.
# Pinning the toolchain here makes "works in CI" and "works on my machine"
# the same claim.
#
#   docker build -t fuzzer-tool .
#   docker run --rm fuzzer-tool                       # run the suite
#   docker run --rm -it fuzzer-tool bash              # interactive
#   docker run --rm -v "$PWD/out:/out" fuzzer-tool \
#       fuzzer-tool fuzz /work/targets/test_target -d /out/corpus -o /out/crashes
#
# Ubuntu 24.04 to match the ubuntu-latest runner CI uses, so a failure
# reproduces here rather than turning out to be a distro difference.
FROM ubuntu:24.04

# Never prompt during apt; tzdata otherwise blocks the build.
ENV DEBIAN_FRONTEND=noninteractive

# clang builds the cmplog/tracecmp shims and the instrumented targets;
# gcc/g++ build the plain and C++ targets (tailslayer_read.cpp);
# binutils supplies nm, which cli/ldpreload_wrapper shells out to for
# sanitizer detection -- without it every target reads as uninstrumented.
# The -dev libraries back the vendored targets under tools/vendor_*.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        binutils \
        git \
        make \
        python3 \
        python3-pip \
        python3-venv \
        zlib1g-dev \
        libpng-dev \
        libjpeg-dev \
        liblz4-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependency metadata first: this layer is cached unless pyproject.toml
# changes, so an edit to src/ does not re-resolve the whole dependency set.
COPY pyproject.toml README.md ./

# Ubuntu 24.04 marks the system Python externally-managed (PEP 668). A venv
# is the sanctioned way through, and is cleaner than --break-system-packages
# because it also pins what `fuzzer-tool` on PATH resolves to.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY . .

# [dev,smt] rather than plain [dev]: z3 backs the SMT path-constraint solver
# and is not pulled by dev, so omitting it is exactly the silent-skip case
# this image is meant to rule out.
RUN pip install --no-cache-dir -e ".[dev,smt]"

# fuzz_loader is gitignored, so a fresh checkout has none. Without it the
# suite HANGS rather than fails: a ForkserverRunner blocks waiting for a
# reply from a loader that does not exist, and each such test only unblocks
# after its grace period. Build it at image-build time so the failure, if
# any, is a compiler error here rather than a slow mystery at run time.
RUN python -c "from fuzzer_tool.adapters.forkserver import _ensure_compiled; \
               import sys; sys.exit(0 if _ensure_compiled() else 'fuzz_loader failed to build')"

# Fail the build if the toolchain is incomplete, rather than shipping an
# image whose green test run is green only because tests were skipped.
# Mirrors the "Verify the toolchain-gated tests are not skipped" CI step.
RUN python - <<'PY'
import importlib.util, shutil, sys
missing = [n for n, ok in (("clang", shutil.which("clang")),
                           ("nm", shutil.which("nm")),
                           ("z3", importlib.util.find_spec("z3")))
           if not ok]
if missing:
    sys.exit(f"toolchain missing: {', '.join(missing)} — "
             "conftest.py would skip the tests that need them")
PY

CMD ["pytest", "-q"]
