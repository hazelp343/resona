# Contributing

Thanks for taking the time to contribute to resona. This is a small project, so
the process is light.

## Getting set up

```bash
git clone https://github.com/hazelp343/resona
cd resona
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .
ruff format --check .
mypy resona
pytest
```

`pre-commit` runs the lint/format/type hooks automatically on each commit; CI
runs the full matrix on Python 3.9–3.12.

## Guidelines

- **Keep the dependency surface small.** NumPy is the only required runtime
  dependency, and the goal is to keep it that way. New optional features can
  live behind an extra, the way `soundfile` does.
- **Match the conventions.** Time-major `(n_frames, n_features)` arrays,
  `float64`, type hints on public functions, and a docstring explaining the
  *why* where it is not obvious.
- **Test what you add.** Each core module has a matching `tests/test_*.py`; new
  behaviour should land with a test in the same spirit.
- **Small, focused commits.** They are easier to review and to revert.

## Reporting bugs

Open an issue with a minimal reproduction. The synthetic generators in
`resona.datasets` are a good way to produce a deterministic repro without
attaching audio files.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
