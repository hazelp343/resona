import pytest
from resona.detection.events import Event
from resona.exceptions import InvalidParameterError
from resona.metrics import error_rate, event_based_metrics, segment_based_metrics


def _events(spans: list[tuple[float, float]], label: str = "a") -> list[Event]:
    return [Event(start, end, label) for start, end in spans]


def test_segment_perfect_match() -> None:
    ref = _events([(0.0, 1.0), (2.0, 3.0)])
    scores = segment_based_metrics(ref, ref, time_resolution=0.5)
    assert scores["f_measure"] == pytest.approx(1.0)
    assert scores["error_rate"] == pytest.approx(0.0)


def test_segment_metrics_with_partial_miss() -> None:
    ref = _events([(0.0, 2.0)])
    est = _events([(0.0, 1.0)])
    scores = segment_based_metrics(ref, est, time_resolution=1.0, duration=2.0)
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(0.5)


def test_event_perfect_match() -> None:
    ref = _events([(0.0, 1.0), (2.0, 3.0)])
    scores = event_based_metrics(ref, ref)
    assert scores["f_measure"] == pytest.approx(1.0)
    assert scores["n_correct"] == 2


def test_event_collar_rejects_far_offset() -> None:
    ref = _events([(0.0, 1.0)])
    est = _events([(0.0, 3.0)])
    scores = event_based_metrics(ref, est, t_collar=0.2, percentage_of_length=0.2)
    assert scores["n_correct"] == 0


def test_event_counts_insertions() -> None:
    ref = _events([(0.0, 1.0)])
    est = _events([(0.0, 1.0), (5.0, 6.0)])
    scores = event_based_metrics(ref, est)
    assert scores["n_correct"] == 1
    assert scores["insertion_rate"] == pytest.approx(1.0)


def test_event_optimal_matching_is_one_to_one() -> None:
    ref = _events([(0.0, 1.0), (0.1, 1.1)])
    est = _events([(0.05, 1.05), (0.0, 1.0)])
    scores = event_based_metrics(ref, est, t_collar=0.2)
    assert scores["n_correct"] == 2


def test_error_rate_helper() -> None:
    ref = _events([(0.0, 1.0)])
    assert error_rate(ref, ref, mode="event") == pytest.approx(0.0)
    with pytest.raises(InvalidParameterError):
        error_rate(ref, ref, mode="bogus")
