"""Extract embeddings from a synthetic clip with each built-in embedder.

Run with::

    python examples/extract_embeddings.py
"""

from __future__ import annotations

import numpy as np
from resona import available_embedders, datasets, extract_embedding


def main() -> None:
    scene = datasets.synthetic_scene(duration=6.0, sr=22050, n_events=5, seed=7)
    print(f"clip: {scene.audio.size / scene.sr:.1f}s @ {scene.sr} Hz")
    for name in available_embedders():
        embedding = extract_embedding(scene.audio, scene.sr, embedder=name, sample_rate=scene.sr)
        pooled = embedding.pooled()
        print(
            f"{name:8s} -> {embedding.n_windows} windows x {embedding.dim} dims "
            f"(pooled L2 norm {np.linalg.norm(pooled):.2f})"
        )


if __name__ == "__main__":
    main()
