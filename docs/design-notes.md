# Design notes

A collection of the deliberate choices behind resona, and the trade-offs each
one makes. These are the "why", not the "how" — see the source and `usage.md`
for the latter.

## NumPy only

The core stack is implemented from scratch on NumPy rather than wrapping librosa
or torchaudio. The cost is some duplicated DSP; the benefit is a tiny dependency
surface, fast installs, and code you can read end to end. For a toolkit meant to
be a reproducible baseline, that legibility is worth more than shaving a few
lines.

## Time-major arrays

Feature matrices are `(n_frames, n_features)`. Frameworks disagree on this —
librosa is feature-major `(n_features, n_frames)` — but time-major composes more
naturally with sliding windows, event rolls, and the way models consume
sequences, and it keeps every layer free of defensive transposes.

## `float64` throughout

Inputs are coerced to `float64` at the boundary. Audio ML often runs in
`float32`, but for a reference implementation correctness beats memory: doubles
sidestep the accumulation error that creeps into long FFTs and mel projections,
and they remove a whole class of dtype-promotion surprises from the code.

## HTK mel scale

`hz_to_mel` / `mel_to_hz` use the HTK formula (`2595 * log10(1 + f/700)`) rather
than the piecewise Slaney curve. It is a single closed-form expression in both
directions, which keeps the filterbank construction obvious. The filterbank
itself uses Slaney-style *area* normalisation so that wide high-frequency bands
are not over-weighted.

## Linear resampling

`io.resample` is linear interpolation, not a polyphase / sinc resampler. It is
exact at the endpoints, dependency-free, and entirely adequate for feature
extraction, where the signal is about to be smeared across mel bands anyway. It
is explicitly *not* meant for high-fidelity playback.

## Optimal event matching

Event-based metrics match reference and estimated events with **Kuhn's algorithm
for maximum bipartite matching** rather than a greedy left-to-right pass. Greedy
matching makes the true-positive count depend on the order events happen to be
listed in; the optimal matching does not. The implementation is a few lines of
pure Python and runs comfortably for realistic event counts.

## Label-constrained matching

Event matching requires labels to agree, so a well-timed but mislabelled event
counts as a deletion *and* an insertion rather than a substitution. This keeps
the matching definition simple and unambiguous; the substitution channel is
reported as zero for event-based scores by design.

## Registries over imports

Embedders (and, by extension, detectors) are selected by name through a small
registry. This is what lets the CLI and high-level pipeline stay decoupled from
concrete classes, and it is the seam where a future learned embedder would plug
in without touching call sites.

## Known limitations

- Linear resampling will alias if you upsample aggressively.
- The built-in detectors are unsupervised baselines, not trained models.
- Spectral-feature extraction recomputes the STFT per descriptor; convenient,
  but not the fastest path for large batches.
