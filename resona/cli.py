"""Command-line interface for resona."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="resona",
        description="Audio embeddings and sound-event detection.",
    )
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0
