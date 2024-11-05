# Architecture

resona is organised as a shallow stack: each layer depends only on the ones
below it, and every layer speaks the same *time-major* array convention —
feature matrices are shaped `(n_frames, n_features)`, never transposed.

```
            ┌─────────────────────────────────────────────┐
 pipeline   │  extract_embedding · detect_events · evaluate │   cli.py
            └─────────────────────────────────────────────┘
                   │              │                │
        embeddings/        detection/           metrics.py
     ┌───────────────┐  ┌──────────────────┐  ┌────────────┐
     │ base · logmel │  │ events · postproc │  │ segment /  │
     │ mfcc · spectral│  │ energy · threshold│  │ event      │
     └───────────────┘  └──────────────────┘  └────────────┘
                   │              │                │
            ┌─────────────────────────────────────────────┐
   core     │   features.py  ·  windowing.py  ·  io.py      │
            └─────────────────────────────────────────────┘
                   │
            ┌─────────────────────────────────────────────┐
  support   │  _typing.py  ·  exceptions.py  ·  datasets.py │
            └─────────────────────────────────────────────┘
```

## Layers

**Support.** `_typing` centralises the array aliases (everything is `float64`).
`exceptions` defines a single `ResonaError` base so callers can catch the whole
family. `datasets` synthesises audio + ground truth for tests and examples.

**Core DSP.** `windowing` turns a signal into overlapping, tapered frames and
converts between frames, samples, and seconds. `features` builds on it: STFT,
mel filterbanks, log-mel, MFCC, deltas, and spectral descriptors. `io` reads and
writes PCM WAV with the standard library and resamples by linear interpolation.

**Embeddings.** `BaseEmbedder` owns the shared flow — downmix, resample, extract
frame features, pool sliding-window statistics — and concrete embedders only
implement `_frame_features`. A tiny registry maps names to classes.

**Detection.** `events` defines the `Event` dataclass and lossless conversion to
and from boolean activity rolls. `postprocess` smooths, thresholds, and enforces
minimum durations. `energy` and `threshold` are the two built-in detectors.

**Metrics.** `metrics` scores detections two ways: segment-based (rasterise onto
a grid) and event-based (match whole events under onset/offset tolerances using
optimal bipartite matching).

**Pipeline & CLI.** `pipeline` exposes three verbs — embed, detect, evaluate —
selectable by name. `cli` is a thin argparse layer over them.

## Conventions

- **Time-major arrays.** `(n_frames, n_features)` everywhere; no hidden
  transposes between layers.
- **`float64` end to end.** Inputs are coerced on the way in; dtype surprises
  stay out.
- **Pure functions in the core.** DSP routines take arrays and return arrays
  with no global state, which keeps them trivially testable.
- **Names, not classes, at the edges.** Embedders and detectors are chosen by
  string so the CLI and user code never import concrete implementations.
