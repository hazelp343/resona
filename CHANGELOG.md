# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Core DSP: framing/windowing, STFT, mel filterbanks, log-mel, MFCC, deltas, and
  spectral descriptors (centroid, bandwidth, roll-off, flatness, flux, ZCR, RMS).
- WAV I/O on the standard library, with optional `soundfile` decoding and a
  linear resampler.
- Three embedders behind a registry: `logmel`, `mfcc`, `spectral`.
- Energy-gate and threshold detectors with median smoothing and minimum-duration
  post-processing.
- Segment-based and event-based metrics with an error-rate decomposition.
- Synthetic scene generator and event-list I/O (CSV / JSON).
- `resona` command-line interface: `info`, `embed`, `detect`, `evaluate`.

[Unreleased]: https://github.com/hazelp343/resona/commits/main
