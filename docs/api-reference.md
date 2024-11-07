# API reference

A map of the public surface. Names listed under `resona` are importable directly
from the package root (`from resona import ...`).

## Top level

| Name | Kind | Summary |
| --- | --- | --- |
| `extract_embedding(signal, sr, *, embedder="logmel", **kw)` | function | Embed a signal with a named embedder. |
| `detect_events(signal, sr, *, detector="energy", **kw)` | function | Detect events in raw audio. |
| `evaluate(reference, estimated, *, mode="segment", **kw)` | function | Score detections (`"segment"` or `"event"`). |
| `available_embedders()` | function | Names of registered embedders. |
| `create_embedder(name, **kw)` / `get_embedder(name)` | function | Instantiate / look up an embedder. |
| `load_audio` · `read_wav` · `write_wav` | function | Audio I/O. |
| `load_events` · `save_events` | function | Event-list I/O (CSV / JSON). |
| `events_to_roll` · `roll_to_events` | function | Convert between events and activity rolls. |
| `Embedding` · `Event` | dataclass | Core containers. |
| `EnergyDetector` · `ThresholdDetector` | class | Built-in detectors. |
| `LogMelStatsEmbedder` · `MFCCEmbedder` · `SpectralEmbedder` | class | Built-in embedders. |
| `ResonaError` and subclasses | exception | Error hierarchy. |

## `resona.features`

Frame-level DSP. All take a 1-D signal and return time-major arrays.

- `stft(signal, *, n_fft, hop_length, window, center) -> complex (frames, bins)`
- `spectrogram(signal, *, power=2.0, ...) -> (frames, bins)`
- `melspectrogram(signal, sr, *, n_mels, fmin, fmax, ...) -> (frames, n_mels)`
- `mel_filterbank(sr, n_fft, *, n_mels, fmin, fmax, norm) -> (n_mels, bins)`
- `power_to_db(spec, *, ref, amin, top_db)`
- `mfcc(signal, sr, *, n_mfcc, ...) -> (frames, n_mfcc)`
- `delta(features, *, width, order)`
- `spectral_centroid` · `spectral_bandwidth` · `spectral_rolloff` ·
  `spectral_flatness` · `spectral_flux` -> `(frames,)`
- `zero_crossing_rate(signal, *, frame_length, hop_length)` · `rms(...)`
- `hz_to_mel` · `mel_to_hz` · `dct_matrix`

## `resona.windowing`

- `get_window(name, length, *, periodic) -> (length,)`
- `frame_signal(signal, frame_length, hop_length, *, pad, pad_mode) -> (frames, frame_length)`
- `num_frames(n_samples, frame_length, hop_length, *, pad) -> int`
- `frames_to_samples` · `samples_to_frames` · `frames_to_time` · `time_to_frames`

## `resona.embeddings`

- `Embedding(vectors, timestamps, sample_rate, name)` with `.dim`,
  `.n_windows`, `.pooled(method="mean"|"max"|"median")`.
- `BaseEmbedder` — subclass and implement `_frame_features(signal)`.
- `register`, `available_embedders`, `get_embedder`, `create_embedder`.

## `resona.detection`

- `events.Event(onset, offset, label="sound", score=1.0)` with `.duration`.
- `events.events_to_roll` · `events.roll_to_events` · `events.find_runs`.
- `postprocess.median_filter` · `binarize` · `apply_min_duration` ·
  `activations_to_events`.
- `energy.EnergyDetector` · `threshold.ThresholdDetector`.
- `eventio.save_events` · `eventio.load_events`.

## `resona.metrics`

- `segment_based_metrics(reference, estimated, *, labels, time_resolution, duration)`
- `event_based_metrics(reference, estimated, *, labels, t_collar, percentage_of_length)`
- `error_rate(reference, estimated, *, mode, **kw)`

Both metric functions return a dict with `precision`, `recall`, `f_measure`,
`error_rate`, `substitution_rate`, `deletion_rate`, `insertion_rate`, and the
raw counts `n_ref`, `n_sys`, `n_correct`.

## `resona.datasets`

- `tone` · `chirp` · `white_noise`
- `synthetic_scene(*, duration, sr, n_events, noise_level, palette, seed) -> Scene`
- `Scene(audio, sr, events)`
