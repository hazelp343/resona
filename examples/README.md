# Examples

Small, self-contained scripts that run against the built-in synthetic data, so
none of them need an audio file to start. Run any of them from the repo root:

```bash
python examples/extract_embeddings.py
python examples/detect_events.py
python examples/evaluate_sed.py
```

| Script | What it shows |
| --- | --- |
| `extract_embeddings.py` | Embeds one clip with each built-in embedder and prints the vector shapes. |
| `detect_events.py` | Runs the energy-gate detector and prints detections next to the ground truth. |
| `evaluate_sed.py` | Scores detector output with both segment- and event-based metrics. |

All three import only `resona` and the standard library.
