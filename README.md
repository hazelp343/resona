# resona

[![CI](https://github.com/hazelp343/resona/actions/workflows/ci.yml/badge.svg)](https://github.com/hazelp343/resona/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230)](https://github.com/astral-sh/ruff)

**Audio embeddings and sound-event detection, in plain NumPy.**

resona turns audio into things you can actually use: fixed-length **embeddings**
for similarity, clustering, and retrieval, and frame-level **sound-event
detection** with the metrics to score it. The entire DSP and SED stack is built
on NumPy alone, so it installs in seconds and runs anywhere Python does — no
deep-learning runtime and no system audio libraries required.

## Why

Most audio toolkits sit at one of two extremes: a heavyweight framework that
pulls in a multi-gigabyte deep-learning stack, or a thin FFT wrapper that leaves
you to assemble mel filterbanks, detectors, and evaluation by hand. resona aims
for the middle — batteries-included but lightweight:

- **One dependency.** NumPy. WAV I/O uses the standard library; `soundfile` is an
  optional extra for other formats.
- **Model-free baselines that actually work.** Log-mel, MFCC, and spectral-shape
  embedders are strong starting points for retrieval and clustering before you
  reach for a neural network.
- **Honest evaluation.** Segment- and event-based metrics follow the conventions
  from the sound-event-detection literature, collars and all.

## Features

- STFT, mel / log-mel spectrograms, MFCCs, deltas, and a full set of spectral
  shape descriptors (centroid, bandwidth, roll-off, flatness, flux, ZCR, RMS).
- Three pluggable embedders (`logmel`, `mfcc`, `spectral`) behind a small
  registry, with sliding-window statistics pooling.
- Energy-gate and threshold-based detectors with median smoothing and
  minimum-duration post-processing.
- Segment-based and event-based metrics (F-measure + error-rate decomposition).
- Synthetic scene generator so you can run the whole pipeline end to end with no
  external data.
- A `resona` command-line tool: `info`, `embed`, `detect`, `evaluate`.

## Install

```bash
pip install resona
# with non-WAV decoding via libsndfile:
pip install "resona[soundfile]"
```

## Quickstart

```python
import resona

# A synthetic clip with ground-truth events -- no data files needed.
scene = resona.datasets.synthetic_scene(duration=6.0, sr=22050, n_events=5)

# Embeddings: one vector per sliding window.
emb = resona.extract_embedding(scene.audio, scene.sr, embedder="logmel")
print(emb.vectors.shape)        # (n_windows, 128)
print(emb.pooled().shape)       # (128,)  -- a single clip-level vector

# Detection + evaluation.
events = resona.detect_events(scene.audio, scene.sr, threshold=0.3)
scores = resona.evaluate(scene.events, events, mode="segment")
print(scores["f_measure"], scores["error_rate"])
```

## Command line

```bash
resona info clip.wav
resona embed clip.wav --embedder mfcc --output emb.npy --pooled
resona detect clip.wav --threshold 0.3 --min-duration-on 0.1 -o events.csv
resona evaluate --reference truth.csv --estimated events.csv --mode event
```

## Status

Early days (`0.x`): the API may still shift between minor versions. It is tested
on CPython 3.9–3.12.

## Performance

resona is built for clarity first, but the NumPy core is vectorised and fast
enough for interactive work: extracting log-mel embeddings from a few minutes of
audio is a sub-second operation on a laptop. The spectral-feature embedder
recomputes an STFT per descriptor, so for large batches prefer the `logmel` or
`mfcc` embedders. There are no learned weights to download — everything runs on
CPU from a cold start.

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)

## License

MIT — see [LICENSE](LICENSE).
