"""Command-line interface for resona."""

from __future__ import annotations

import argparse

import numpy as np

from .embeddings import available_embedders
from .io import load_audio
from .pipeline import extract_embedding


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
