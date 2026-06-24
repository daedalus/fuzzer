#!/bin/bash
set -x
python3 -m fuzzer_tool fuzz targets/png_read \
  -d ~/fuzzing/png_read/corpus \
  -o ~/fuzzing/png_read/crashes \
  -F -A "{file}" \
  -c \
  -D dictionaries/png.dict \
  -g dictionaries/png.gram \
  --markov --markov-gen \
  --mc-bandit --mc-cem \
  --auto-timeout \
  --coverage-report ~/fuzzing/png_read/coverage.json \
  --stats-file ~/fuzzing/png_read/stats.json \
  -s 42 \
  -n 3000 \
  -t 2 \
  -M 16
