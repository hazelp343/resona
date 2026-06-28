"""Command-line interface for resona."""

from __future__ import annotations

import argparse

import numpy as np

from .io import load_audio


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
