# GStreamer Fuzzing Findings

## Target

[GStreamer](https://gstreamer.freedesktop.org/) 1.28.9 — the GStreamer multimedia framework, covering audio/video decoding, demuxing, and processing.

## Fuzz Targets

| Target | Attack Surface | Input Format |
|--------|---------------|--------------|
| `gstreamer_read` | Matroska/WebM demux + Opus audio decode | EBML/WebM container |

## Bugs

- **[gst_buffer_add_audio_clipping_meta.md](gstreamer/gst_buffer_add_audio_clipping_meta.md)** — NULL buffer dereference in `gst_buffer_add_audio_clipping_meta()` called from Matroska demuxer's block-group parser when processing crafted EBML lacing fields. SIGSEGV at `gstaudiometa.c:274`. Severity: Medium.
