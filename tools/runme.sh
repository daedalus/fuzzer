#!/bin/bash
set -x
TARGET=$1
fuzzer-tool fuzz targets/"$TARGET"_read.elf -d /tmp/fuzz_"$TARGET" \
            -c --markov --markov-gen --markov-order 0,1,2,3 \
            -g dictionaries/$TARGET.gram \
            --calibrate 2000 --stall 1000 --max-collision-risk 25 \
            -n 105000  --elo \
            --report /tmp/"$TARGET"_fuzz_report.txt

