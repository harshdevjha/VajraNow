import pytest

from eval.scoreboard import evaluate


def test_evaluator_uses_only_whole_requested_events_and_reports_real_counts():
    rows = []
    for event_id in ("train-event", "test-event"):
        rows.extend(
            [
                {"event_id": event_id, "valid_time": "t0", "grid_y": "0", "grid_x": "0", "probability": "0.9", "baseline_probability": "0.8", "observed": "1"},
                {"event_id": event_id, "valid_time": "t0", "grid_y": "0", "grid_x": "1", "probability": "0.1", "baseline_probability": "0.2", "observed": "0"},
            ]
        )
    result = evaluate(rows, {"test-event"})
    assert result["events"] == ["test-event"]
    assert result["grid_samples"] == 2
    assert result["CSI"] == 1
    assert result["Brier"] < 0.02


def test_evaluator_rejects_incomplete_spatial_grid():
    rows = [{"event_id": "e1", "valid_time": "t0", "grid_y": "0", "grid_x": "1", "probability": "0.1", "baseline_probability": "0.2", "observed": "0"}]
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(rows, {"e1"})
