import pytest

from resona.datasets import Scene
from resona.embeddings import available_embedders, create_embedder, get_embedder
from resona.exceptions import InvalidParameterError, UnknownComponentError


def test_registry_lists_builtins() -> None:
    assert {"logmel", "mfcc", "spectral"} <= set(available_embedders())


@pytest.mark.parametrize(("name", "expected_dim"), [("logmel", 128), ("spectral", 14)])
def test_embedder_dimensions(scene: Scene, name: str, expected_dim: int) -> None:
    embedding = create_embedder(name, sample_rate=scene.sr).embed(scene.audio, scene.sr)
    assert embedding.dim == expected_dim
    assert embedding.vectors.shape[0] == embedding.n_windows
    assert embedding.timestamps.shape[0] == embedding.n_windows


def test_pooled_vector_shape(scene: Scene) -> None:
    embedding = create_embedder("mfcc", sample_rate=scene.sr).embed(scene.audio, scene.sr)
    assert embedding.pooled().shape == (embedding.dim,)
    assert embedding.pooled("max").shape == (embedding.dim,)
    assert embedding.pooled("median").shape == (embedding.dim,)


def test_callable_alias_matches_embed(scene: Scene) -> None:
    embedder = create_embedder("logmel", sample_rate=scene.sr)
    via_call = embedder(scene.audio, scene.sr)
    via_method = embedder.embed(scene.audio, scene.sr)
    assert via_call.vectors.shape == via_method.vectors.shape


def test_unknown_embedder_raises() -> None:
    with pytest.raises(UnknownComponentError):
        get_embedder("does-not-exist")


def test_unknown_pooling_raises(scene: Scene) -> None:
    embedding = create_embedder("spectral", sample_rate=scene.sr).embed(scene.audio, scene.sr)
    with pytest.raises(InvalidParameterError):
        embedding.pooled("nonsense")
