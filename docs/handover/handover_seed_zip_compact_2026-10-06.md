# seed_zip compaction (2026-10-06)

Source: Dostoevsky-style LSM simulator (user-supplied). Verdict: only `adapters/seed_zip.py` maps (append-only log + tombstones, never compacted).

## Not ported
- LSM merge/overwrite: seeds are content-addressed and immutable.
- Dropping long-pruned data (the original plan): violates the corpus rule (never delete). Replaced by moving it to `seeds/pruned/`.
- Levels/size ratios: one hot level suffices; the block buffer already plays the memtable.

## Built
`compact()`: spill cold seeds -> rewrite -> replay check -> `os.replace`. CLI `compact-seeds`.

## Limits
- One small file per pruned seed (what file mode already does); 9000 files in the benchmark.
- Offline only; the fuzzer must not be running on the corpus (only detected in-process).
- Foreign members never compacted. Recompression at level 9 (2.8 s / 10k).
- Not run against a real long campaign corpus.
