# Usage guide

This guide walks through the three things resona does: extracting embeddings,
detecting events, and scoring detections. Every snippet runs against the
built-in synthetic data, so you can paste them as-is.

## Loading audio

```python
from resona import load_audio

signal, sr = load_audio("clip.wav")              # mono float64 in [-1, 1]
signal, sr = load_audio("clip.wav", sr=16000)    # resample on load
```

WAV files are decoded with the standard library. Other formats (FLAC, OGG, ...)
require the optional `soundfile` extra; without it, a clear `AudioIOError` tells
you what to install.

## Features

Everything in `resona.features` takes a 1-D signal and returns a time-major
`(n_frames, ...)` array.

```python
import numpy as np
from resona import features

sig = np.random.default_rng(0).standard_normal(22050)

S = features.melspectrogram(sig, sr=22050, n_mels=64)   # (frames, 64)
logmel = features.power_to_db(S)
mfcc = features.mfcc(sig, sr=22050, n_mfcc=20)           # (frames, 20)
mfcc_d = features.delta(mfcc)                            # same shape
centroid = features.spectral_centroid(sig, sr=22050)    # (frames,)
```

## Embeddings

An embedder pools frame features into fixed-length vectors over sliding windows.

```python
from resona import extract_embedding, available_embedders

print(available_embedders())   # ['logmel', 'mfcc', 'spectral']

emb = extract_embedding(sig, 22050, embedder="logmel")
emb.vectors      # (n_windows, dim)
emb.timestamps   # (n_windows,) -- window start times in seconds
emb.pooled()     # (dim,) -- mean over windows; also "max" / "median"
```

Tune the analysis by passing constructor arguments through:

```python
emb = extract_embedding(sig, 22050, embedder="mfcc", n_mfcc=13, include_deltas=False)
```

## Detection

The energy gate finds active regions in raw audio:

```python
from resona import detect_events

events = detect_events(
    sig, 22050,
    threshold=0.3,        # fraction of the peak energy
    min_duration_on=0.10, # drop blips shorter than 100 ms
    min_duration_off=0.05 # bridge gaps shorter than 50 ms
)
for e in events:
    print(e.onset, e.offset, e.label)
```

If you already have frame-level class scores from a model, use the
`ThresholdDetector` directly:

```python
from resona.detection.threshold import ThresholdDetector

detector = ThresholdDetector(threshold=0.5, median_filter_frames=5)
events = detector.detect(activations, labels=["dog", "cat"], sr=22050, hop_length=512)
```

## Evaluation

```python
from resona import evaluate

segment = evaluate(reference, estimated, mode="segment", time_resolution=1.0)
event = evaluate(reference, estimated, mode="event", t_collar=0.2)
print(segment["f_measure"], event["error_rate"])
```

Both modes return precision, recall, F-measure, and an error rate broken down
into substitution, deletion, and insertion rates.

## Saving and loading events

```python
from resona import save_events, load_events

save_events("events.csv", events)   # or events.json
again = load_events("events.csv")
```
