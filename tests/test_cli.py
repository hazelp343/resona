from resona import detect_events, evaluate, extract_embedding
from resona.cli import build_parser, main
from resona.datasets import Scene
from resona.detection.events import Event
from resona.eventio import save_events
from resona.io import write_wav


def test_build_parser_has_info_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["info", "clip.wav"])
    assert args.command == "info"


def test_pipeline_extract_and_detect(scene: Scene) -> None:
    embedding = extract_embedding(
        scene.audio, scene.sr, embedder="logmel", sample_rate=scene.sr
    )
    assert embedding.n_windows > 0
    events = detect_events(scene.audio, scene.sr, threshold=0.3)
    assert isinstance(events, list)


def test_pipeline_evaluate_modes(scene: Scene) -> None:
    reference = [Event(e.onset, e.offset, "active") for e in scene.events]
    assert evaluate(reference, reference, mode="segment")["f_measure"] == 1.0
    assert evaluate(reference, reference, mode="event")["f_measure"] == 1.0


def test_cli_info_runs(tmp_path, capsys, scene: Scene) -> None:
    clip = tmp_path / "clip.wav"
    write_wav(str(clip), scene.audio, scene.sr)
    assert main(["info", str(clip)]) == 0
    assert "duration" in capsys.readouterr().out


def test_cli_detect_then_evaluate(tmp_path, scene: Scene) -> None:
    clip = tmp_path / "clip.wav"
    write_wav(str(clip), scene.audio, scene.sr)
    reference = tmp_path / "ref.csv"
    save_events(str(reference), [Event(e.onset, e.offset, "active") for e in scene.events])
    estimated = tmp_path / "est.csv"

    rc = main(
        ["detect", str(clip), "--threshold", "0.3", "--min-duration-on", "0.1", "-o", str(estimated)]
    )
    assert rc == 0
    assert estimated.exists()
    assert main(["evaluate", "-r", str(reference), "-e", str(estimated), "--mode", "segment"]) == 0


def test_cli_no_command_returns_error() -> None:
    assert main([]) == 1
