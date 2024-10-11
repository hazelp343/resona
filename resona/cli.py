"""Command-line interface for resona."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .detection.eventio import load_events, save_events
from .embeddings import available_embedders
from .io import load_audio
from .pipeline import detect_events, evaluate, extract_embedding


def _cmd_info(args: argparse.Namespace) -> int:
    signal, sr = load_audio(args.audio, sr=args.sr)
    duration = signal.size / sr if sr else 0.0
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    level = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
    print(f"file:     {args.audio}")
    print(f"samples:  {signal.size}")
    print(f"sr:       {sr} Hz")
    print(f"duration: {duration:.3f} s")
    print(f"peak:     {peak:.4f}")
    print(f"rms:      {level:.4f}")
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    signal, sr = load_audio(args.audio, sr=args.sr)
    embedding = extract_embedding(
        signal, sr, embedder=args.embedder, sample_rate=sr
    )
    output = embedding.pooled() if args.pooled else embedding.vectors
    np.save(args.output, output)
    print(f"saved {embedding.name} embedding {output.shape} to {args.output}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    signal, sr = load_audio(args.audio, sr=args.sr)
    events = detect_events(
        signal,
        sr,
        detector=args.detector,
        threshold=args.threshold,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )
    if args.output:
        save_events(args.output, events)
        print(f"detected {len(events)} event(s) -> {args.output}")
    else:
        for event in events:
            print(
                f"{event.onset:.3f}\t{event.offset:.3f}\t{event.label}\t{event.score:.3f}"
            )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    reference = load_events(args.reference)
    estimated = load_events(args.estimated)
    extra = (
        {"time_resolution": args.time_resolution}
        if args.mode == "segment"
        else {"t_collar": args.t_collar}
    )
    scores = evaluate(reference, estimated, mode=args.mode, **extra)
    print(json.dumps(scores, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="resona",
        description="Audio embeddings and sound-event detection.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info = subparsers.add_parser("info", help="Summarise an audio file.")
    info.add_argument("audio", help="Path to an audio file.")
    info.add_argument("--sr", type=int, default=None, help="Resample before analysis.")
    info.set_defaults(func=_cmd_info)

    embed = subparsers.add_parser("embed", help="Extract an embedding to a .npy file.")
    embed.add_argument("audio", help="Path to an audio file.")
    embed.add_argument(
        "--embedder",
        default="logmel",
        choices=available_embedders(),
        help="Which embedder to use.",
    )
    embed.add_argument("--output", "-o", required=True, help="Destination .npy path.")
    embed.add_argument("--sr", type=int, default=None, help="Resample before embedding.")
    embed.add_argument(
        "--pooled",
        action="store_true",
        help="Save a single mean-pooled vector instead of the per-window matrix.",
    )
    embed.set_defaults(func=_cmd_embed)

    detect = subparsers.add_parser("detect", help="Detect sound events in audio.")
    detect.add_argument("audio", help="Path to an audio file.")
    detect.add_argument("--detector", default="energy", help="Detector to use.")
    detect.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    detect.add_argument("--n-fft", type=int, default=2048, dest="n_fft", help="FFT size.")
    detect.add_argument(
        "--hop-length", type=int, default=512, dest="hop_length", help="Frame hop."
    )
    detect.add_argument(
        "--min-duration-on",
        type=float,
        default=0.0,
        dest="min_duration_on",
        help="Discard events shorter than this (seconds).",
    )
    detect.add_argument(
        "--min-duration-off",
        type=float,
        default=0.0,
        dest="min_duration_off",
        help="Bridge gaps shorter than this (seconds).",
    )
    detect.add_argument(
        "--output", "-o", default=None, help="Write events here; otherwise print them."
    )
    detect.add_argument("--sr", type=int, default=None, help="Resample before detecting.")
    detect.set_defaults(func=_cmd_detect)

    evaluate_cmd = subparsers.add_parser(
        "evaluate", help="Score detected events against a reference."
    )
    evaluate_cmd.add_argument("--reference", "-r", required=True, help="Reference events.")
    evaluate_cmd.add_argument("--estimated", "-e", required=True, help="Estimated events.")
    evaluate_cmd.add_argument(
        "--mode", default="segment", choices=("segment", "event"), help="Metric family."
    )
    evaluate_cmd.add_argument(
        "--time-resolution",
        type=float,
        default=1.0,
        dest="time_resolution",
        help="Segment length in seconds (segment mode).",
    )
    evaluate_cmd.add_argument(
        "--t-collar",
        type=float,
        default=0.2,
        dest="t_collar",
        help="Onset collar in seconds (event mode).",
    )
    evaluate_cmd.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))
